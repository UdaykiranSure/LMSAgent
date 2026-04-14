"""
src/grpo_trainer.py

GRPO (Group Relative Policy Optimisation) trainer for the student agent.
Replaces the custom PPO stack (ppo_core.py + rollout.py + trainer.py) with
TRL's GRPOTrainer.

Key differences from PPO:
  - No value head / critic needed — advantage is computed from group statistics.
  - For each prompt we sample G completions (group_size), score each with the
    rule-based reward model, normalise within the group, and use those as
    advantages directly.
  - KL penalty against a frozen reference model is handled internally by TRL.

Architecture
───────────────────────────────────────────────────────────────────────────
  GRPOTrainer (TRL)
    │
    ├── Policy model  : Qwen2.5-2B-Instruct + LoRA  (trained)
    ├── Ref model     : same weights, frozen          (KL anchor)
    │
    └── reward_fn()   : runs the StudentAgentEnvironment for each completion
                        → returns a scalar via RuleBasedRewardModel.score()

How completions are generated:
  Unlike supervised GRPO (where the model generates a full answer in one shot),
  our agent operates in a *multi-step tool loop*.  We handle this by running
  the full environment rollout inside reward_fn() and returning the terminal
  reward.  The "completion" passed to the reward function is the concatenated
  agent output from all steps (tool calls + final respond).

  TRL's GRPOTrainer generates completions internally, so we hook into the
  reward_fn callback that receives (prompts, completions) and can execute
  arbitrary Python — perfect for our environment.

Usage
─────
  python -m src.grpo_trainer              # default config
  python -m src.grpo_trainer --stage 3   # harder curriculum
"""

import os
import re
import json
import random
import logging
import argparse
import numpy as np
from datetime import date
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig

# ── Project imports ────────────────────────────────────────────────────
from src.environment import StudentAgentEnvironment
from src.reward import RuleBasedRewardModel
from src.dataset import EpisodeDataset, MockStudentDB
from src.rollout import format_messages, SYSTEM_PROMPT

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ══════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════

@dataclass
class GRPOStudentConfig:
    """
    Hyperparameters for GRPO training.
    Split into two halves:
      - GRPOStudentConfig  : our domain-level settings
      - GRPOConfig (TRL)   : passed directly to GRPOTrainer
    """

    # ── Model ──────────────────────────────────────────────────────────
    model_name:          str  = "Qwen/Qwen3.5-0.8B"
    load_in_4bit:        bool = False           # set True for <24 GB VRAM

    # ── LoRA ───────────────────────────────────────────────────────────
    use_lora:            bool  = True
    lora_r:              int   = 16
    lora_alpha:          int   = 32
    lora_dropout:        float = 0.05
    lora_target_modules: list  = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",   # MLP gates in Qwen2.5
    ])

    # ── GRPO core ──────────────────────────────────────────────────────
    group_size:          int   = 8      # G — completions per prompt
    #   Advantage = (r - mean(group)) / (std(group) + eps)

    # ── Generation ─────────────────────────────────────────────────────
    max_prompt_length:   int   = 512
    max_completion_length: int = 1024   # per step; total capped by env
    temperature:         float = 0.8
    top_p:               float = 0.9

    # ── Rollout / environment ──────────────────────────────────────────
    max_env_steps:       int   = 10     # max tool calls before forced stop

    # ── Training ───────────────────────────────────────────────────────
    total_steps:         int   = 2000   # gradient update steps
    batch_size:          int   = 8      # prompts per step
    #   effective_batch = batch_size × group_size = 64 completions
    mini_batch_size:     int   = 2      # per GPU forward pass
    gradient_accum:      int   = 4
    lr:                  float = 5e-7
    weight_decay:        float = 0.01
    max_grad_norm:       float = 0.5
    warmup_steps:        int   = 50

    # ── KL penalty ─────────────────────────────────────────────────────
    kl_coef:             float = 0.04   # β — penalises deviation from ref

    # ── Curriculum ─────────────────────────────────────────────────────
    stage:               int   = 2      # 2 = L1+L2, 3 = L1+L2+L3

    # ── Checkpointing ──────────────────────────────────────────────────
    output_dir:          str   = "checkpoints/grpo"
    save_steps:          int   = 200
    logging_steps:       int   = 10
    eval_steps:          int   = 200
    seed:                int   = 42


# ══════════════════════════════════════════════════════════════════════
# Prompt builder
# ══════════════════════════════════════════════════════════════════════

def build_prompt(query: str) -> str:
    """
    Build the initial prompt string for the model.
    Mirrors the format used in rollout.py so the model sees identical
    formatting during both training and inference.
    """
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": query},
    ]
    return format_messages(messages)


# ══════════════════════════════════════════════════════════════════════
# Multi-step rollout inside reward_fn
# ══════════════════════════════════════════════════════════════════════

def run_environment_episode(
    model,
    tokenizer,
    query:    str,
    mock_db:  MockStudentDB,
    gt,
    cfg:      GRPOStudentConfig,
    device:   torch.device,
) -> tuple[str, float]:
    """
    Run one full multi-step episode.

    Because GRPO's reward_fn receives *already-generated* completions from
    TRL's generation loop, we need a way to use the environment's multi-step
    nature.  We solve this by:

      1. Using the *first* completion that TRL generated as step 1's output.
      2. Continuing to generate subsequent steps ourselves (with no_grad) using
         the current policy model until the environment is done.
      3. Returning the concatenated full agent text + the terminal reward.

    In practice we call this function *instead* of TRL's generation loop by
    providing a custom data_collator + generate override — see GRPOStudentTrainer.

    Returns
    -------
    full_response : str   — all agent outputs concatenated (for logging)
    reward        : float — scalar reward in [-1, 1]
    """
    reward_model = RuleBasedRewardModel()
    env          = StudentAgentEnvironment(mock_db, gt, query, date.today())

    all_outputs = []

    with torch.no_grad():
        for _ in range(cfg.max_env_steps):
            if env.state.done:
                break

            # Format current context
            ctx_text = format_messages(env.state.context_window)
            inputs   = tokenizer(
                ctx_text,
                return_tensors = "pt",
                truncation     = True,
                max_length     = cfg.max_prompt_length + cfg.max_completion_length,
                padding        = False,
            ).to(device)

            # Generate one step
            output_ids = model.generate(
                **inputs,
                max_new_tokens = cfg.max_completion_length,
                temperature    = cfg.temperature,
                top_p          = cfg.top_p,
                do_sample      = True,
                pad_token_id   = tokenizer.eos_token_id,
                eos_token_id   = tokenizer.eos_token_id,
            )

            # Decode only the new tokens
            new_ids   = output_ids[0, inputs["input_ids"].shape[1]:]
            step_text = tokenizer.decode(new_ids, skip_special_tokens=True)
            all_outputs.append(step_text)

            obs = env.step(step_text)
            if obs["done"]:
                break

    trajectory = env.build_trajectory()
    score      = reward_model.score(trajectory, gt)
    reward     = float(score["total"])

    full_response = "\n".join(all_outputs)
    return full_response, reward


# ══════════════════════════════════════════════════════════════════════
# Dataset adapter — converts EpisodeDataset to HuggingFace Dataset
# ══════════════════════════════════════════════════════════════════════

def build_hf_dataset(episode_dataset: EpisodeDataset, n_samples: int):
    """
    Generate n_samples query/db/gt triples and wrap them in a
    datasets.Dataset that GRPOTrainer can iterate over.

    Each row stores:
      - "prompt"  : the formatted initial prompt string
      - "query"   : raw query (for env construction in reward_fn)
      - "db_seed" : reproducible seed to reconstruct the MockStudentDB
      - "gt_json" : serialised GroundTruth fields needed by reward model
    """
    from datasets import Dataset

    rows = []
    today = date.today()

    for _ in range(n_samples):
        query, db, gt = episode_dataset.sample(today)
        prompt        = build_prompt(query)

        # Serialise only what the reward function needs
        gt_payload = {
            "correct_answer":   gt.correct_answer,
            "answer_type":      gt.answer_type,
            "intent":           gt.intent,
            "required_tools":   gt.required_tools,
            "required_facts":   gt.required_facts,
            "expected_writes":  gt.expected_writes,
            "optimal_steps":    gt.optimal_steps,
            "max_steps":        gt.max_steps,
        }

        # Serialise the DB state so we can reconstruct it in reward_fn
        db_payload = {
            "subjects":    db.get_subjects(),
            "assignments": {s: db.get_assignments(s) for s in db.get_subjects()},
            "grades":      {s: db.get_grades(s)      for s in db.get_subjects()},
            "schedule":    {s: db.get_schedule(s)    for s in db.get_subjects()},
            "notes":       {s: db.get_notes(s)       for s in db.get_subjects()},
        }

        rows.append({
            "prompt":   prompt,
            "query":    query,
            "gt_json":  json.dumps(gt_payload),
            "db_json":  json.dumps(db_payload),
        })

    return Dataset.from_list(rows)


# ══════════════════════════════════════════════════════════════════════
# Reward function — the core GRPO callback
# ══════════════════════════════════════════════════════════════════════

class RewardFunctionFactory:
    """
    Wraps a stateful reward function that can be passed to GRPOTrainer.

    GRPOTrainer calls:
        rewards = reward_fn(prompts, completions, **kwargs)

    We reconstruct the environment from the serialised db/gt payloads
    stored in the batch and score each completion.

    NOTE: completions here are *single-step* strings generated by TRL's
    internal generate loop.  For the multi-step tool-calling case we use
    the "full rollout" approach (see GRPOStudentTrainer below).
    For simple single-step scoring (e.g. does the final response look
    correct?) this is used directly.
    """

    def __init__(self, cfg: GRPOStudentConfig):
        self.cfg          = cfg
        self.reward_model = RuleBasedRewardModel()

    def __call__(
        self,
        prompts:     list[str],
        completions: list[str],
        gt_json:     list[str] | None = None,
        db_json:     list[str] | None = None,
        query:       list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        """
        Score each (prompt, completion) pair.

        For multi-step environments the completion is the concatenation of
        all agent steps.  We parse tool calls and the final <respond> from
        it to reconstruct a trajectory, then score with RuleBasedRewardModel.
        """
        rewards = []

        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            try:
                gt_data = json.loads(gt_json[i])  if gt_json else {}
                db_data = json.loads(db_json[i])  if db_json else {}
                q       = query[i]                 if query   else ""

                # Reconstruct a lightweight GT object
                gt = _dict_to_gt(gt_data)

                # Reconstruct a lightweight DB object
                db = _dict_to_db(db_data)

                # Build a trajectory from the completion text
                trajectory = _parse_trajectory(completion, q, gt)

                score  = self.reward_model.score(trajectory, gt)
                reward = float(score["total"])

            except Exception as e:
                logger.warning(f"reward_fn error for sample {i}: {e}")
                reward = 0.0

            rewards.append(reward)

        return rewards


def _dict_to_gt(d: dict):
    """Reconstruct a minimal GroundTruth-compatible object from a dict."""
    from src.dataset import GroundTruth
    return GroundTruth(
        correct_answer        = d.get("correct_answer", ""),
        answer_type           = d.get("answer_type", ""),
        intent                = d.get("intent", ""),
        required_tools        = d.get("required_tools", []),
        required_tool_families= d.get("required_tool_families", []),
        optimal_steps         = d.get("optimal_steps", 5),
        max_steps             = d.get("max_steps", 10),
        required_facts        = d.get("required_facts", []),
        expected_writes       = d.get("expected_writes", []),
    )


class _ReconstructedDB:
    """Minimal read-only DB reconstructed from a serialised snapshot."""
    def __init__(self, data: dict):
        self._data = data

    def get_subjects(self):          return self._data.get("subjects", [])
    def get_assignments(self, s=""):
        d = self._data.get("assignments", {})
        return d.get(s, []) if s else d
    def get_grades(self, s):         return self._data.get("grades", {}).get(s, [])
    def get_schedule(self, s=None):
        d = self._data.get("schedule", {})
        return d.get(s, {}) if s else d
    def get_notes(self, s):          return self._data.get("notes", {}).get(s, [])
    def get_todos(self):             return []
    def add_todo(self, *a, **kw):   return {}
    def add_notes(self, *a, **kw):  return {}
    def get_allnotes(self, s):      return []
    def get_announcements(self, s): return []
    def get_assignment_details(self, s):
        return self._data.get("assignments", {}).get(s, [])


def _dict_to_db(d: dict) -> _ReconstructedDB:
    return _ReconstructedDB(d)


def _parse_trajectory(completion: str, query: str, gt) -> dict:
    """
    Parse a raw completion string into the trajectory dict that
    RuleBasedRewardModel.score() expects.

    The completion may contain multiple tool calls and a final <respond>.
    """
    tool_calls = []

    for m in re.finditer(r'<tool_call>(.*?)</tool_call>', completion, re.DOTALL):
        try:
            call   = json.loads(m.group(1).strip())
            tool_calls.append({
                "tool":   call.get("tool", ""),
                "params": call.get("params", {}),
                "result": {},   # results not embedded in the completion text
            })
        except json.JSONDecodeError:
            pass

    final_response = ""
    respond_m = re.search(r'<respond>(.*?)</respond>', completion, re.DOTALL)
    if respond_m:
        final_response = respond_m.group(1).strip()

    return {
        "query":          query,
        "tool_calls":     tool_calls,
        "final_response": final_response,
        "steps_used":     len(tool_calls) + (1 if final_response else 0),
        "max_steps":      getattr(gt, "max_steps", 10),
        "intent":         getattr(gt, "intent", ""),
    }


# ══════════════════════════════════════════════════════════════════════
# Main trainer class
# ══════════════════════════════════════════════════════════════════════

class GRPOStudentTrainer:
    """
    Thin wrapper that wires together:
      - the EpisodeDataset / Oracle
      - TRL's GRPOTrainer
      - our RuleBasedRewardModel (via RewardFunctionFactory)

    Full multi-step rollout strategy
    ──────────────────────────────────
    TRL's GRPOTrainer supports a reward_fn that is called with the model's
    completions.  For a multi-step tool-calling agent the "completion" is
    the *entire* sequence of tool calls + final respond.

    We handle this by setting max_new_tokens to a large value (≈1024) and
    prompting the model to emit the full chain in one shot — which is valid
    because the model is trained to predict token-by-token regardless.
    The environment is then reconstructed inside reward_fn by parsing the
    emitted text.

    For true interactive multi-step training (where tool results feed back
    into the context), use `run_environment_episode()` directly and generate
    your own dataset offline.  See --mode offline below.
    """

    def __init__(self, cfg: GRPOStudentConfig):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _set_seed(cfg.seed)

        logger.info(f"Initialising GRPO trainer | model={cfg.model_name} | device={self.device}")
        self._build()

    # ──────────────────────────────────────────────────────────────────
    def _build(self):
        cfg = self.cfg
        self._rope_reset_hooks = []

        # ── Tokenizer ─────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"   # required for batch generation

        # ── LoRA config ────────────────────────────────────────────────
        peft_config = None
        if cfg.use_lora:
            peft_config = LoraConfig(
                r               = cfg.lora_r,
                lora_alpha      = cfg.lora_alpha,
                lora_dropout    = cfg.lora_dropout,
                target_modules  = cfg.lora_target_modules,
                task_type       = TaskType.CAUSAL_LM,
                bias            = "none",
            )

        # ── TRL GRPOConfig ─────────────────────────────────────────────
        # Maps our config fields to TRL's naming conventions.
        self.trl_cfg = GRPOConfig(
            output_dir                  = cfg.output_dir,
            num_train_epochs            = 1,            # we control via max_steps
            max_steps                   = cfg.total_steps,
            per_device_train_batch_size = cfg.batch_size,
            gradient_accumulation_steps = cfg.gradient_accum,
            learning_rate               = cfg.lr,
            weight_decay                = cfg.weight_decay,
            max_grad_norm               = cfg.max_grad_norm,
            warmup_steps                = cfg.warmup_steps,
            logging_steps               = cfg.logging_steps,
            save_steps                  = cfg.save_steps,
            eval_steps                  = cfg.eval_steps,
            seed                        = cfg.seed,
            bf16                        = torch.cuda.is_bf16_supported(),
            fp16                        = not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
            # ── GRPO-specific ──────────────────────────────────────────
            num_generations             = cfg.group_size,      # G
            max_completion_length       = cfg.max_completion_length,
            temperature                 = cfg.temperature,
            top_p                       = cfg.top_p,
            beta                        = cfg.kl_coef,         # KL coefficient
            # ── Misc ──────────────────────────────────────────────────
            report_to                   = "wandb",              # swap to "wandb" if desired
            remove_unused_columns       = False,               # we need gt_json, db_json
            dataloader_num_workers      = 0,
        )

        # ── Dataset ────────────────────────────────────────────────────
        levels = [1, 2] if cfg.stage == 2 else [1, 2, 3]
        ep_ds  = EpisodeDataset(seed=cfg.seed, stage=cfg.stage)
        ep_ds.set_levels(levels)

        # We pre-generate a large dataset; GRPOTrainer will shuffle + repeat.
        n_samples = max(cfg.total_steps * cfg.batch_size, 2000)
        logger.info(f"Generating {n_samples} training samples …")
        self.train_dataset = build_hf_dataset(ep_ds, n_samples)

        # Eval set (always all levels)
        eval_ds = EpisodeDataset(seed=99999, stage=3)
        eval_ds.set_levels([1, 2, 3])
        self.eval_dataset = build_hf_dataset(eval_ds, 200)

        logger.info(f"Train: {len(self.train_dataset)} rows | Eval: {len(self.eval_dataset)} rows")

        # ── Reward function ────────────────────────────────────────────
        self.reward_fn = RewardFunctionFactory(cfg)

        # ── GRPOTrainer ────────────────────────────────────────────────
        self.trainer = GRPOTrainer(
            model           = cfg.model_name,
            args            = self.trl_cfg,
            train_dataset   = self.train_dataset,
            eval_dataset    = self.eval_dataset,
            processing_class= self.tokenizer,
            reward_funcs    = self._wrapped_reward_fn,
            peft_config     = peft_config,
        )

        # Qwen3.5 caches rope_deltas on the module. During GRPO generation and
        # scoring, effective batch sizes can differ, which can leave a stale
        # cache with incompatible shape for the next forward pass.
        self._rope_reset_hooks.append(_install_qwen_rope_reset_hook(self.trainer.model))
        self._rope_reset_hooks.append(_install_qwen_rope_reset_hook(getattr(self.trainer, "ref_model", None)))

        logger.info("GRPOTrainer ready.")

    # ──────────────────────────────────────────────────────────────────
    def _wrapped_reward_fn(self, prompts, completions, **batch):
        """
        Adapter between TRL's reward_fn signature and our RewardFunctionFactory.

        TRL passes the *entire batch dict* as **kwargs, so gt_json / db_json /
        query columns are available here automatically.
        """
        return self.reward_fn(
            prompts     = prompts,
            completions = completions,
            gt_json     = batch.get("gt_json"),
            db_json     = batch.get("db_json"),
            query       = batch.get("query"),
        )

    # ──────────────────────────────────────────────────────────────────
    def train(self):
        logger.info("Starting GRPO training …")
        self.trainer.train()
        logger.info("Training complete — saving final checkpoint …")
        self.trainer.save_model(os.path.join(self.cfg.output_dir, "final"))
        self.tokenizer.save_pretrained(os.path.join(self.cfg.output_dir, "final"))
        logger.info(f"Model saved → {self.cfg.output_dir}/final")

    # ──────────────────────────────────────────────────────────────────
    def evaluate(self, n: int = 50) -> float:
        """
        Run n evaluation episodes with full multi-step rollouts and return
        mean reward.  Uses run_environment_episode() so tool results feed
        back into context (closer to real deployment).
        """
        model = self.trainer.model
        model.eval()

        eval_ds  = EpisodeDataset(seed=99999, stage=3)
        eval_ds.set_levels([1, 2, 3])
        rewards  = []

        for _ in range(n):
            query, db, gt = eval_ds.sample()
            try:
                _, reward = run_environment_episode(
                    model, self.tokenizer, query, db, gt,
                    self.cfg, self.device
                )
                rewards.append(reward)
            except Exception as e:
                logger.debug(f"Eval error: {e}")
                rewards.append(0.0)

        mean_r = float(np.mean(rewards)) if rewards else 0.0
        logger.info(f"Evaluation over {n} episodes — mean reward: {mean_r:.4f}")
        model.train()
        return mean_r


# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reset_qwen_rope_deltas(module: torch.nn.Module):
    """Clear cached Qwen rope_deltas recursively."""
    for submodule in module.modules():
        if hasattr(submodule, "rope_deltas"):
            setattr(submodule, "rope_deltas", None)


def _install_qwen_rope_reset_hook(module: Optional[torch.nn.Module]):
    """Install a pre-forward hook that resets Qwen rope_deltas caches."""
    if module is None:
        return None

    def _hook(mod, _inputs):
        _reset_qwen_rope_deltas(mod)

    return module.register_forward_pre_hook(_hook)


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════

def parse_args() -> GRPOStudentConfig:
    parser = argparse.ArgumentParser(description="GRPO trainer for student agent")
    parser.add_argument("--model",        default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--stage",        type=int,   default=2)
    parser.add_argument("--group_size",   type=int,   default=8)
    parser.add_argument("--total_steps",  type=int,   default=2000)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=5e-7)
    parser.add_argument("--kl_coef",      type=float, default=0.04)
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--output_dir",   default="checkpoints/grpo2")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    return GRPOStudentConfig(
        model_name    = args.model,
        stage         = args.stage,
        group_size    = args.group_size,
        total_steps   = args.total_steps,
        batch_size    = args.batch_size,
        lr            = args.lr,
        kl_coef       = args.kl_coef,
        lora_r        = args.lora_r,
        lora_alpha    = args.lora_r * 2,
        output_dir    = args.output_dir,
        load_in_4bit  = args.load_in_4bit,
        seed          = args.seed,
    )


if __name__ == "__main__":
    cfg     = parse_args()
    trainer = GRPOStudentTrainer(cfg)
    trainer.train()
    trainer.evaluate(n=50)