"""
src/trainer.py

Main PPO training loop.

Flow per iteration:
  ┌──────────────────────────────────────────────────────────┐
  │  1. Sample batch_size queries from curriculum dataset    │
  │  2. RolloutCollector runs each query through env         │
  │     → actor generates tool calls step by step           │
  │     → environment executes tools, steps state           │
  │     → reward model scores completed trajectory          │
  │     → Episode(tokens, old_logprobs, values, reward)     │
  │  3. PPOBuffer.compute_advantages() — GAE over episodes  │
  │  4. PPOUpdater.update() — K epochs of PPO               │
  │     → for each mini-batch:                              │
  │         forward(ids) → new_logprobs, new_values         │
  │         ref_forward(ids) → ref_logprobs                 │
  │         policy_loss + value_loss + entropy - KL         │
  │         backward + clip + step                          │
  │  5. Log metrics, checkpoint, advance curriculum         │
  └──────────────────────────────────────────────────────────┘
"""

import os
import json
import time
import logging
import random
import numpy as np
import torch
from datetime import date
from collections import deque
from typing import Optional

from transformers import AutoTokenizer
from src.ppo_core import (
    PPOConfig, ActorCritic, ReferenceModel,
    AdaptiveKLController, PPOBuffer, PPOUpdater
)
from src.rollout import RolloutCollector

logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    Self-contained PPO trainer for the student agent.
    No TRL, no external RL framework.
    """

    def __init__(self, cfg: PPOConfig):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _set_seed(cfg.seed)
        logger.info(f"Device: {self.device}")

        self._build()

    def _build(self):
        cfg = self.cfg

        # ── Tokenizer ──────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # ── Actor-Critic (policy + value head) ────────────────────────
        self.actor_critic = ActorCritic(cfg, self.device)

        # ── Frozen reference model ─────────────────────────────────────
        self.ref_model = ReferenceModel(cfg.model_name, self.device)

        # ── KL controller ──────────────────────────────────────────────
        self.kl_ctrl = AdaptiveKLController(
            cfg.init_kl_coef, cfg.target_kl, cfg.kl_horizon
        )

        # ── PPO updater ────────────────────────────────────────────────
        self.updater = PPOUpdater(
            self.actor_critic, self.ref_model,
            self.kl_ctrl, cfg, self.device
        )

        # ── Reward model ───────────────────────────────────────────────
        from src.reward import RuleBasedRewardModel
        self.reward_model = RuleBasedRewardModel()

        # ── Dataset ────────────────────────────────────────────────────
        from src.dataset import EpisodeDataset
        self.dataset = EpisodeDataset(seed=cfg.seed)

        # ── Rollout collector ──────────────────────────────────────────
        self.collector = RolloutCollector(
            actor_critic = self.actor_critic,
            tokenizer    = self.tokenizer,
            reward_model = self.reward_model,
            cfg          = cfg,
            device       = self.device,
        )

        # ── Metrics tracking ───────────────────────────────────────────
        self.metrics_window = deque(maxlen=50)   # rolling window for logging
        self.global_step    = 0
        self.best_reward    = -float("inf")

    # ──────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────

    def train(self, stage: int = 2):
        cfg = self.cfg
        logger.info(f"Starting PPO training — stage {stage}, device={self.device}")
        logger.info(f"Total episodes: {cfg.total_episodes}, batch_size: {cfg.batch_size}")

        # Stage-specific settings
        dense_w  = 1.0 if stage == 2 else 0.2
        sparse_w = 0.0 if stage == 2 else 1.0
        levels   = [1, 2] if stage == 2 else [1, 2, 3]
        self.dataset.set_levels(levels)

        buffer     = PPOBuffer()
        episodes_done = 0
        t_start    = time.time()

        while episodes_done < cfg.total_episodes:
            print("Allocated:", torch.cuda.memory_allocated()/1e9)
            print("Reserved :", torch.cuda.memory_reserved()/1e9)
            # ── 1. Sample queries ──────────────────────────────────────
            batch   = self.dataset.sample_batch(cfg.batch_size)
            queries = [b[0] for b in batch]
            dbs     = [b[1] for b in batch]
            gts     = [b[2] for b in batch]

            # ── 2. Collect rollouts ────────────────────────────────────
            self.actor_critic.eval()
            episodes = self.collector.collect_batch(
                queries = queries,
                dbs     = dbs,
                gts     = gts,
                today   = date.today(),
            )
            self.actor_critic.train()

            if not episodes:
                logger.warning("Empty episode batch — skipping")
                continue
            
            # ── 3. Apply reward weighting (dense vs sparse) ───────────
            for ep, original_batch in zip(episodes, batch[:len(episodes)]):
                # ep.reward is the terminal reward from reward model
                # For dense mode we could add per-step shaping here
                ep.reward = ep.reward * (dense_w + sparse_w)
                ep.reward = max(-1.0, min(1.0, ep.reward))

            # ── 4. Fill buffer ─────────────────────────────────────────
            buffer.clear()
            print("Allocated:", torch.cuda.memory_allocated()/1e9)
            print("Reserved :", torch.cuda.memory_reserved()/1e9)   
            for ep in episodes:
                buffer.add(ep)
                print("Allocated:", torch.cuda.memory_allocated()/1e9)
                print("Reserved :", torch.cuda.memory_reserved()/1e9)

            # ── 5. PPO update ──────────────────────────────────────────
            update_stats = self.updater.update(buffer)
            print("Allocated:", torch.cuda.memory_allocated()/1e9)
            print("Reserved :", torch.cuda.memory_reserved()/1e9)
            episodes_done += len(episodes)
            self.global_step += 1

            # ── 6. Log ─────────────────────────────────────────────────
            rewards = [ep.reward for ep in episodes]
            self.metrics_window.extend(rewards)
            self._log(update_stats, episodes, episodes_done, t_start)

            # ── 7. Evaluate ────────────────────────────────────────────
            if episodes_done % cfg.eval_every < cfg.batch_size:
                eval_reward = self._evaluate()
                logger.info(f"[eval @ {episodes_done}] mean_reward={eval_reward:.4f}")
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self._save("best")

            # ── 8. Checkpoint ──────────────────────────────────────────
            if episodes_done % cfg.save_every < cfg.batch_size:
                self._save(f"step_{episodes_done}")

        logger.info("Training complete.")
        self._save("final")

    # ──────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────

    def _evaluate(self, n: int = 30) -> float:
        """Run n held-out episodes and return mean reward."""
        eval_ds = self.dataset.__class__(seed=99999)
        eval_ds.set_levels([1, 2, 3])
        self.actor_critic.eval()

        total = 0.0
        for _ in range(n):
            q, db, gt = eval_ds.sample()
            try:
                ep     = self.collector.collect_episode(q, db, gt)
                total += ep.reward
            except Exception as e:
                logger.debug(f"Eval episode error: {e}")

        self.actor_critic.train()
        return total / n

    # ──────────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────────

    def _log(
        self,
        update_stats:  dict,
        episodes:      list,
        episodes_done: int,
        t_start:       float,
    ):
        rewards     = [ep.reward for ep in episodes]
        steps       = [ep.steps_taken for ep in episodes]
        exits       = [ep.exit_type for ep in episodes]

        mean_r      = np.mean(rewards)
        rolling_r   = np.mean(self.metrics_window)
        mean_steps  = np.mean(steps)
        pct_respond = np.mean([1 if e == "respond" else 0 for e in exits])
        elapsed     = time.time() - t_start
        eps_per_s   = episodes_done / max(elapsed, 1)

        log_data = {
            "episode":         episodes_done,
            "reward/mean":     round(mean_r, 4),
            "reward/rolling50":round(rolling_r, 4),
            "steps/mean":      round(mean_steps, 2),
            "respond_pct":     round(pct_respond, 3),
            "ppo/policy_loss": round(update_stats.get("policy_loss", 0), 5),
            "ppo/value_loss":  round(update_stats.get("value_loss", 0), 5),
            "ppo/entropy":     round(update_stats.get("entropy", 0), 5),
            "ppo/kl":          round(update_stats.get("kl", 0), 5),
            "ppo/approx_kl":   round(update_stats.get("approx_kl", 0), 5),
            "ppo/clip_frac":   round(update_stats.get("clip_frac", 0), 5),
            "ppo/kl_coef":     round(update_stats.get("kl_coef", 0), 5),
            "ppo/lr":          round(update_stats.get("lr", 0), 8),
            "eps_per_sec":     round(eps_per_s, 2),
        }

        logger.info(
            f"[{episodes_done:6d}] "
            f"r={mean_r:+.3f} rolling={rolling_r:+.3f} | "
            f"steps={mean_steps:.1f} respond={pct_respond*100:.0f}% | "
            f"pg={log_data['ppo/policy_loss']:.4f} "
            f"vf={log_data['ppo/value_loss']:.4f} "
            f"kl={log_data['ppo/kl']:.4f} "
            f"β={log_data['ppo/kl_coef']:.4f}"
        )

        # Append to JSONL log
        os.makedirs("logs", exist_ok=True)
        with open("logs/train.jsonl", "a") as f:
            f.write(json.dumps(log_data) + "\n")

    # ──────────────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────────────

    def _save(self, tag: str):
        path = os.path.join("checkpoints", tag)
        os.makedirs(path, exist_ok=True)
        # Save LoRA adapter weights only (much smaller than full model)
        self.actor_critic.backbone.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # Save value head separately
        torch.save(self.actor_critic.value_head.state_dict(),
                   os.path.join(path, "value_head.pt"))
        # Save optimizer + scheduler state
        torch.save({
            "optimizer":   self.updater.optimizer.state_dict(),
            "scheduler":   self.updater.scheduler.state_dict(),
            "global_step": self.global_step,
            "kl_coef":     self.kl_ctrl.beta,
            "best_reward": self.best_reward,
        }, os.path.join(path, "training_state.pt"))
        logger.info(f"Checkpoint saved → {path}")

    def load(self, path: str):
        """Resume from a checkpoint."""
        from peft import PeftModel
        self.actor_critic.backbone = PeftModel.from_pretrained(
            self.actor_critic.backbone, path
        )
        self.actor_critic.value_head.load_state_dict(
            torch.load(os.path.join(path, "value_head.pt"), map_location=self.device)
        )
        state = torch.load(os.path.join(path, "training_state.pt"), map_location="cpu")
        self.updater.optimizer.load_state_dict(state["optimizer"])
        self.updater.scheduler.load_state_dict(state["scheduler"])
        self.global_step   = state["global_step"]
        self.kl_ctrl.beta  = state["kl_coef"]
        self.best_reward   = state["best_reward"]
        logger.info(f"Resumed from {path} (step {self.global_step})")


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
