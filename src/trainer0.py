# src/trainer.py
"""
PPOStudentTrainer — main training loop.

Architecture:
  - Actor:   policy LLM (LoRA fine-tuned) — generates tool calls + RESPOND
  - Critic:  value head on top of actor backbone — estimates V(state)
  - Ref:     frozen copy of SFT model — KL penalty anchor
  - Reward:  RuleBasedRewardModel — no neural reward model needed

TRL's PPOTrainer handles:
  - GAE advantage computation
  - Clipped surrogate objective
  - Value function loss
  - KL penalty against reference model
  - Adaptive KL controller
  - Gradient accumulation + clipping

We handle:
  - Episode rollout via RolloutCollector
  - Curriculum-aware query sampling
  - Stage transitions (dense → sparse rewards)
  - Evaluation and checkpointing
"""

import os
import yaml
import torch
import wandb
import logging
from datetime import date
from typing import Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from trl import (
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
)

from src.dataset import EpisodeDataset
from src.rollout import RolloutCollector, experiences_to_ppo_batch
from src.reward import RuleBasedRewardModel
from src.environment import StudentAgentEnvironment

logger = logging.getLogger(__name__)


class PPOStudentTrainer:
    """
    Orchestrates the full PPO training loop for the student agent.

    Usage:
        trainer = PPOStudentTrainer(config_path="configs/ppo_config.yaml", stage=2)
        trainer.train()
    """

    def __init__(self, config_path: str, stage: int = 2):
        self.config  = self._load_config(config_path)
        self.stage   = stage
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initialising PPO trainer — stage {stage} on {self.device}")
        self._setup_logging()
        self._build_components()

    # ─────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────

    def _load_config(self, path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        log_cfg = self.config["training"]
        if log_cfg.get("log_with") == "wandb":
            wandb.init(
                project = log_cfg["project_name"],
                name    = f"ppo-stage{self.stage}",
                config  = self.config,
            )

    def _build_components(self):
        cfg        = self.config
        model_cfg  = cfg["model"]
        ppo_cfg    = cfg["ppo"]

        # ── Tokenizer ─────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["base_model"],
            padding_side = "left",      # left-pad for decoder-only generation
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Actor + value head ────────────────────────────────────────
        # AutoModelForCausalLMWithValueHead wraps the LLM and adds a
        # scalar value head (linear layer) on top of the last hidden state.
        # This is what TRL's PPOTrainer expects.
        sft_ckpt = model_cfg.get("sft_checkpoint", model_cfg["base_model"])
        logger.info(f"Loading actor from {sft_ckpt}")

        base = AutoModelForCausalLM.from_pretrained(
            sft_ckpt,
            torch_dtype  = torch.bfloat16,
            device_map   = "auto",
            load_in_4bit = model_cfg.get("load_in_4bit", False),
        )

        # Apply LoRA if requested
        if model_cfg.get("use_lora", True):
            lora_cfg = LoraConfig(**{
                k: v for k, v in model_cfg["lora"].items()
            })
            base = get_peft_model(base, lora_cfg)
            base.print_trainable_parameters()

        # Wrap with value head — this becomes our actor/critic
        self.model = AutoModelForCausalLMWithValueHead(base)
        self.model.to(self.device)

        # ── Reference model — frozen SFT weights ──────────────────────
        # TRL uses this to compute KL(π_θ || π_ref) per token
        logger.info("Loading frozen reference model")
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            sft_ckpt,
            torch_dtype = torch.bfloat16,
            device_map  = "auto",
        )
        # Freeze all ref model params
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        # ── PPO config ────────────────────────────────────────────────
        self.ppo_config = PPOConfig(
            # Naming
            model_name             = model_cfg["base_model"],
            # Batch sizes
            batch_size             = ppo_cfg["batch_size"],
            mini_batch_size        = ppo_cfg["mini_batch_size"],
            gradient_accumulation_steps = max(
                1, ppo_cfg["batch_size"] // ppo_cfg["mini_batch_size"]
            ),
            # PPO hyperparams
            ppo_epochs             = ppo_cfg["ppo_epochs"],
            learning_rate          = ppo_cfg["learning_rate"],
            gamma                  = ppo_cfg["gamma"],
            lam                    = ppo_cfg["lam"],
            cliprange              = ppo_cfg["cliprange"],
            cliprange_value        = ppo_cfg["cliprange_value"],
            vf_coef                = ppo_cfg["vf_coef"],
            max_grad_norm          = ppo_cfg["max_grad_norm"],
            # KL penalty
            kl_penalty             = ppo_cfg["kl_penalty"],
            init_kl_coef           = ppo_cfg["init_kl_coef"],
            target_kl              = ppo_cfg["target_kl"],
            horizon                = ppo_cfg["horizon"],
            # Logging
            log_with               = self.config["training"].get("log_with"),
            # Reproducibility
            seed                   = self.config["training"]["seed"],
        )

        # ── TRL PPOTrainer ────────────────────────────────────────────
        # PPOTrainer manages: optimizer, scheduler, GAE, loss computation
        self.ppo_trainer = PPOTrainer(
            config    = self.ppo_config,
            model     = self.model,
            ref_model = self.ref_model,
            tokenizer = self.tokenizer,
        )

        # ── Reward model ──────────────────────────────────────────────
        self.reward_model = RuleBasedRewardModel(self.config["reward"])

        # ── Dataset (query sampler) ───────────────────────────────────
        self.dataset = EpisodeDataset(
            config = self.config,
            stage  = self.stage,
            seed   = self.config["training"]["seed"],
        )

        # ── Rollout collector ─────────────────────────────────────────
        self.rollout_collector = RolloutCollector(
            model        = self.model,
            tokenizer    = self.tokenizer,
            reward_model = self.reward_model,
            config       = self.config,
            device       = self.device,
        )

    # ─────────────────────────────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────────────────────────────

    def train(self):
        cfg           = self.config["training"]
        ppo_cfg       = self.config["ppo"]
        reward_cfg    = self.config["reward"]
        batch_size    = ppo_cfg["batch_size"]

        # Stage-specific reward weighting
        dense_w  = reward_cfg["dense_reward_weight"]
        sparse_w = reward_cfg["sparse_reward_weight"]

        total_episodes  = (
            cfg["stage2_episodes"] if self.stage == 2
            else cfg["stage3_episodes"]
        )
        episodes_so_far = 0
        best_eval_score = -float("inf")

        logger.info(
            f"Starting PPO stage {self.stage} — "
            f"{total_episodes} episodes, batch_size={batch_size}, "
            f"dense_w={dense_w}, sparse_w={sparse_w}"
        )

        while episodes_so_far < total_episodes:

            # ── Step 1: Sample a batch of queries ─────────────────────
            batch       = self.dataset.sample_batch(batch_size)
            queries_str = [b[0] for b in batch]
            dbs         = [b[1] for b in batch]
            gts         = [b[2] for b in batch]

            # ── Step 2: Collect rollouts ──────────────────────────────
            # Run the policy in the environment for each query
            experiences = self.rollout_collector.collect(
                queries       = queries_str,
                mock_dbs      = dbs,
                ground_truths = gts,
            )

            # ── Step 3: Convert to PPO batch format ───────────────────
            # queries:   list[Tensor]  — prompt token ids
            # responses: list[Tensor]  — response token ids
            # rewards:   list[Tensor]  — scalar reward per episode
            queries, responses, rewards = experiences_to_ppo_batch(
                experiences  = experiences,
                dense_weight = dense_w,
                sparse_weight= sparse_w,
            )

            # ── Step 4: PPO update ────────────────────────────────────
            # PPOTrainer.step() does:
            #   1. Forward pass on actor → log_probs, values
            #   2. Forward pass on ref   → ref_log_probs
            #   3. KL penalty per token: kl = log_probs - ref_log_probs
            #   4. GAE advantage using values + rewards
            #   5. PPO clip loss + value loss
            #   6. Gradient step × ppo_epochs
            stats = self.ppo_trainer.step(queries, responses, rewards)

            episodes_so_far += batch_size

            # ── Step 5: Log metrics ───────────────────────────────────
            self._log_step(stats, experiences, episodes_so_far)

            # ── Step 6: Evaluate ──────────────────────────────────────
            if episodes_so_far % cfg["eval_every"] == 0:
                eval_score = self._evaluate()
                logger.info(
                    f"[ep {episodes_so_far}] eval_score={eval_score:.3f}"
                )
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    self._save_checkpoint("best")

            # ── Step 7: Periodic checkpoint ───────────────────────────
            if episodes_so_far % cfg["save_every"] == 0:
                self._save_checkpoint(f"ep_{episodes_so_far}")

        logger.info("Training complete.")
        self._save_checkpoint("final")

    # ─────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────

    def _evaluate(self, n_queries: int = 40) -> float:
        """
        Run n_queries held-out episodes, return mean reward.
        Uses a fixed seed dataset so results are comparable across checkpoints.
        """
        eval_dataset = EpisodeDataset(
            config = self.config,
            stage  = self.stage,
            seed   = 9999,          # fixed seed — never seen during training
        )
        self.model.eval()
        total_reward = 0.0
        metrics = {
            "correct":           0,
            "hallucinated":      0,
            "tool_order_ok":     0,
            "early_exit_ok":     0,
            "step_limit_hit":    0,
        }

        for _ in range(n_queries):
            query, db, gt = eval_dataset.sample()
            exps = self.rollout_collector.collect(
                queries       = [query],
                mock_dbs      = [db],
                ground_truths = [gt],
            )
            exp = exps[0]
            total_reward += exp.reward

            if exp.reward >= 0.9:
                metrics["correct"] += 1
            if exp.reward <= -0.8:
                metrics["hallucinated"] += 1
            if exp.exit_type == "step_limit":
                metrics["step_limit_hit"] += 1

        mean_reward = total_reward / n_queries
        metrics     = {k: v / n_queries for k, v in metrics.items()}

        if self.config["training"].get("log_with") == "wandb":
            wandb.log({"eval/mean_reward": mean_reward, **{
                f"eval/{k}": v for k, v in metrics.items()
            }})

        self.model.train()
        return mean_reward

    # ─────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────

    def _log_step(
        self,
        ppo_stats:       dict,
        experiences:     list,
        episodes_so_far: int,
    ):
        rewards      = [e.reward for e in experiences]
        steps        = [e.steps_taken for e in experiences]
        exit_types   = [e.exit_type for e in experiences]

        mean_reward  = sum(rewards) / len(rewards)
        mean_steps   = sum(steps)   / len(steps)
        pct_respond  = sum(1 for e in exit_types if e == "respond") / len(exit_types)
        pct_limit    = sum(1 for e in exit_types if e == "step_limit") / len(exit_types)

        log_dict = {
            "train/mean_reward":          mean_reward,
            "train/mean_steps":           mean_steps,
            "train/pct_clean_respond":    pct_respond,
            "train/pct_step_limit_hit":   pct_limit,
            "train/episodes":             episodes_so_far,
            # PPO internals from TRL
            "ppo/policy_loss":            ppo_stats.get("ppo/loss/policy", 0),
            "ppo/value_loss":             ppo_stats.get("ppo/loss/value", 0),
            "ppo/entropy":                ppo_stats.get("ppo/policy/entropy", 0),
            "ppo/kl":                     ppo_stats.get("ppo/mean_scores", 0),
            "ppo/approx_kl":             ppo_stats.get("ppo/policy/approxkl", 0),
            "ppo/clipfrac":               ppo_stats.get("ppo/policy/clipfrac", 0),
        }

        logger.info(
            f"[ep {episodes_so_far:6d}] "
            f"reward={mean_reward:+.3f} "
            f"steps={mean_steps:.1f} "
            f"respond%={pct_respond*100:.0f}% "
            f"kl={log_dict['ppo/approx_kl']:.4f}"
        )

        if self.config["training"].get("log_with") == "wandb":
            wandb.log(log_dict, step=episodes_so_far)

    # ─────────────────────────────────────────────────────────────────
    # Checkpointing
    # ─────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, tag: str):
        path = os.path.join("checkpoints", tag)
        os.makedirs(path, exist_ok=True)
        self.ppo_trainer.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Saved checkpoint → {path}")
