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

from __future__ import annotations

import os
import re
import json
import random
import logging
import argparse
import sys
import numpy as np
from datetime import date, datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, TrainerState, TrainerControl
from peft import LoraConfig, TaskType, PeftModel
from trl import GRPOTrainer, GRPOConfig

# Allow running this script from either project root or scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Project imports ────────────────────────────────────────────────────
from src.environment import StudentAgentEnvironment
from src.reward import RuleBasedRewardModel
from src.dataset import EpisodeDataset, MockStudentDB
from src.rollout import format_messages, SYSTEM_PROMPT


class _DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("grpo2.log"), # Saves to file
        logging.StreamHandler()         # Prints live to console
    ]
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
    sft_adapter_path:    Optional[str] = None   # path to SFT-warmed LoRA adapter

    # ── LoRA ───────────────────────────────────────────────────────────
    use_lora:            bool  = True
    lora_r:              int   = 8
    lora_alpha:          int   = 32
    lora_dropout:        float = 0.05
    lora_target_modules: list  = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",   # MLP gates in Qwen2.5
    ])

    # ── GRPO core ──────────────────────────────────────────────────────
    # Interactive rollout cost = batch_size × group_size × max_env_steps × generate()
    # Keep batch_size × group_size ≤ 16 for practical wall-clock time.
    group_size:          int   = 4      # G — completions per prompt (was 8)

    # ── Generation ─────────────────────────────────────────────────────
    max_prompt_length:   int   = 1024   # must fit system prompt (~861 tok) + user query
    max_completion_length: int = 256    # per step; tool calls are short JSON
    temperature:         float = 0.9    # slightly higher → more within-group diversity
    top_p:               float = 0.9

    # ── Rollout / environment ──────────────────────────────────────────
    max_env_steps:       int   = 10     # matches GT max_steps for L1/L2 queries

    # ── Training ───────────────────────────────────────────────────────
    total_steps:         int   = 2000   # gradient update steps
    batch_size:          int   = 4      # prompts per step (was 8); 4×4=16 episodes/step
    #   effective_batch = batch_size × group_size = 64 completions
    mini_batch_size:     int   = 2      # per GPU forward pass
    gradient_accum:      int   = 4
    lr:                  float = 5e-7
    weight_decay:        float = 0.01
    max_grad_norm:       float = 0.5
    warmup_steps:        int   = 50

    # ── KL penalty ─────────────────────────────────────────────────────
    kl_coef:             float = 0.1    # β — higher to prevent KL explosion

    # ── Curriculum ─────────────────────────────────────────────────────
    stage:               int   = 1      # 1 = L1 only, 2 = L1+L2 (weighted), 3 = all

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

    def __init__(self, cfg: GRPOStudentConfig, tracker: MetricsTracker | None = None):
        self.cfg          = cfg
        self.reward_model = RuleBasedRewardModel()
        self.tracker      = tracker

    def __call__(
        self,
        prompts:         list[str],
        completions:     list[str],
        trajectory_json: list[str] | None = None,   # from interactive rollout_fn
        gt_json:         list[str] | None = None,
        db_json:         list[str] | None = None,
        query:           list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        rewards      = []
        breakdowns   = []
        n_tool_calls = []
        comp_lens    = []

        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            try:
                if trajectory_json and i < len(trajectory_json) and trajectory_json[i]:
                    # Interactive rollout path: trajectory includes real tool results
                    traj    = json.loads(trajectory_json[i])
                    gt_data = traj.pop("_gt_data", {})
                    gt      = _dict_to_gt(gt_data)
                else:
                    # Single-shot fallback: parse completion text (no real tool results)
                    gt_data = json.loads(gt_json[i]) if gt_json else {}
                    q       = query[i] if query else ""
                    gt      = _dict_to_gt(gt_data)
                    traj    = _parse_trajectory(completion, q, gt)

                score  = self.reward_model.score(traj, gt)
                reward = float(score["total"])
                breakdowns.append(score["breakdown"])
                n_tool_calls.append(len(traj.get("tool_calls", [])))

            except Exception as e:
                logger.warning(f"reward_fn error for sample {i}: {e}")
                reward = 0.0
                breakdowns.append({})
                n_tool_calls.append(0)

            rewards.append(reward)
            comp_lens.append(len(completion))

        if self.tracker is not None:
            self.tracker.record(
                rewards      = rewards,
                breakdowns   = breakdowns,
                n_tool_calls = n_tool_calls,
                comp_lens    = comp_lens,
                group_size   = self.cfg.group_size,
            )

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


# ══════════════════════════════════════════════════════════════════════
# Interactive rollout — runs the real StudentAgentEnvironment step-by-step
# ══════════════════════════════════════════════════════════════════════

def _run_interactive_episode(
    model,
    tokenizer,
    prompt_ids: list[int],
    query:      str,
    db,
    gt,
    cfg:        "GRPOStudentConfig",
    device:     torch.device,
) -> tuple[list[int], list[float], dict]:
    """
    Run one full interactive episode with the real DB.

    The model generates one step at a time; tool results are injected back
    into the context window between steps.  Only the model's generated tokens
    are returned in completion_ids — tool-result tokens are not included.

    Log probs are computed per step on the actual context the model saw when
    generating each token (including interleaved tool results), giving accurate
    importance weights for the policy-gradient loss.

    Returns
    -------
    completion_ids : list[int]   — model-generated token IDs
    logprobs       : list[float] — per-token log probs under the actual generation context
    trajectory     : dict        — full trajectory with real tool results + embedded gt
    """
    env = StudentAgentEnvironment(db, gt, query, date.today())

    all_comp_ids:  list[int]   = []
    all_logprobs:  list[float] = []

    for _ in range(cfg.max_env_steps):
        if env.state.done:
            break

        ctx_text = format_messages(env.state.context_window)
        ctx_ids  = tokenizer(
            ctx_text,
            return_tensors  = "pt",
            truncation      = True,
            max_length      = cfg.max_prompt_length + cfg.max_completion_length,
            add_special_tokens = False,
        ).input_ids.to(device)

        # Fix Issue 5: reset stale rope cache on the unwrapped model before each generate
        _reset_qwen_rope_deltas(model)
        out = model.generate(
            ctx_ids,
            attention_mask          = torch.ones_like(ctx_ids),
            max_new_tokens          = 256,
            temperature             = cfg.temperature,
            top_p                   = cfg.top_p,
            do_sample               = True,
            pad_token_id            = tokenizer.eos_token_id,
            return_dict_in_generate = True,
            output_scores           = True,   # needed for per-step log probs
        )

        new_ids   = out.sequences[0, ctx_ids.shape[1]:].tolist()
        step_text = tokenizer.decode(new_ids, skip_special_tokens=True)

        all_comp_ids.extend(new_ids)
        # Fix Issue 1: log probs from the actual per-step context, not flat concatenation
        for tok_id, score in zip(new_ids, out.scores):
            all_logprobs.append(torch.log_softmax(score[0], dim=-1)[tok_id].item())

        obs = env.step(step_text)
        if obs["done"]:
            break

    if not all_comp_ids:
        all_comp_ids = [tokenizer.eos_token_id]
        all_logprobs = [0.0]

    trajectory = env.build_trajectory()
    return all_comp_ids, all_logprobs, trajectory


def make_interactive_rollout_fn(cfg: "GRPOStudentConfig"):
    """
    Returns a TRL 1.0 rollout_func that runs full interactive episodes with
    the real MockStudentDB instead of letting TRL do single-shot generation.

    For each prompt in the batch, G episodes are run.  The resulting token IDs
    and log probs are returned in the format TRL expects.  The full trajectory
    (including real tool results) is passed as the extra field `trajectory_json`
    which TRL forwards to the reward function.
    """
    def rollout_fn(prompts: list, trainer) -> dict:
        model     = trainer.accelerator.unwrap_model(trainer.model)
        tokenizer = trainer.processing_class
        device    = trainer.accelerator.device

        # _current_inputs has B entries (raw batch); TRL expands prompts to B*G before
        # calling rollout_func. Use integer division to map back to the source row.
        inputs = getattr(trainer, "_current_inputs", None) or []
        n_src  = len(inputs)
        G      = cfg.group_size

        prompt_ids_out:     list[list[int]]   = []
        completion_ids_out: list[list[int]]   = []
        logprobs_out:       list[list[float]] = []
        trajectory_out:     list[str]         = []

        model.eval()
        try:
            with torch.no_grad():
                for i, prompt in enumerate(prompts):
                    # Fix Issue 2/6: prompts has B*G entries; inputs has B entries
                    src_idx = (i // G) if n_src > 0 and G > 0 else 0
                    row     = inputs[src_idx] if src_idx < n_src else {}
                    gt_data = json.loads(row.get("gt_json", "{}"))
                    db_data = json.loads(row.get("db_json", "{}"))
                    query   = row.get("query", "")

                    gt = _dict_to_gt(gt_data)
                    db = _dict_to_db(db_data)

                    p_ids = tokenizer(
                        prompt,
                        return_tensors     = "pt",
                        truncation         = True,
                        max_length         = cfg.max_prompt_length,
                        add_special_tokens = False,
                    ).input_ids[0].tolist()

                    comp_ids, lps, traj = _run_interactive_episode(
                        model, tokenizer, p_ids, query, db, gt, cfg, device
                    )
                    traj["_gt_data"] = gt_data

                    prompt_ids_out.append(p_ids)
                    completion_ids_out.append(comp_ids)
                    logprobs_out.append(lps)
                    trajectory_out.append(json.dumps(traj, cls=_DateEncoder))
        finally:
            model.train()

        # Fix Issue 4: pad to uniform length and convert to tensors for TRL's loss step
        pad_id       = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        max_p_len    = max(len(p) for p in prompt_ids_out)
        max_comp_len = max(len(c) for c in completion_ids_out)

        def _pad(seqs, pad_val, length):
            return [s + [pad_val] * (length - len(s)) for s in seqs]

        return {
            "prompt_ids":      torch.tensor(_pad(prompt_ids_out,     pad_id, max_p_len),    dtype=torch.long,    device=device),
            "completion_ids":  torch.tensor(_pad(completion_ids_out, pad_id, max_comp_len), dtype=torch.long,    device=device),
            "logprobs":        torch.tensor(_pad(logprobs_out,       0.0,    max_comp_len), dtype=torch.float32, device=device),
            "trajectory_json": trajectory_out,
        }

    return rollout_fn


class MultiStepGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer subclass that caches the current batch's raw inputs on
    self._current_inputs before generation, so that rollout_func can
    read db_json / gt_json / query for environment construction.
    """

    def _generate_and_score_completions(self, inputs):
        self._current_inputs = inputs
        return super()._generate_and_score_completions(inputs)


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
# Metrics tracking
# ══════════════════════════════════════════════════════════════════════

class MetricsTracker:
    """
    Accumulates per-call reward statistics from reward_fn and flushes
    them to wandb + a JSONL log file every `log_every` calls.
    """

    _BREAKDOWN_KEYS = ("correctness", "grounding", "tool_path", "step_limit", "write_ops")

    def __init__(self, log_every: int = 10, log_file: str = "grpo_metrics.jsonl"):
        self.log_every = log_every
        self.log_file  = log_file
        self._call     = 0
        self._buf: dict[str, list] = {k: [] for k in (
            "rewards", "n_tool_calls", "completion_len", "group_reward_std",
            *self._BREAKDOWN_KEYS,
        )}

    def record(
        self,
        rewards:      list[float],
        breakdowns:   list[dict],
        n_tool_calls: list[int],
        comp_lens:    list[int],
        group_size:   int = 1,
    ):
        self._buf["rewards"].extend(rewards)
        for bd in breakdowns:
            for k in self._BREAKDOWN_KEYS:
                v = bd.get(k)
                if v is not None:
                    self._buf[k].append(float(v))
        self._buf["n_tool_calls"].extend(n_tool_calls)
        self._buf["completion_len"].extend(comp_lens)

        # Within-group reward std — proxy for within-group exploration
        if group_size > 1 and len(rewards) % group_size == 0:
            r_groups = np.array(rewards).reshape(-1, group_size)
            self._buf["group_reward_std"].extend(r_groups.std(axis=1).tolist())

        self._call += 1
        if self._call % self.log_every == 0:
            self._flush()

    def _flush(self):
        r = np.array(self._buf["rewards"])
        if len(r) == 0:
            return

        metrics: dict[str, float] = {
            "reward/mean":          float(r.mean()),
            "reward/std":           float(r.std()),
            "reward/min":           float(r.min()),
            "reward/max":           float(r.max()),
            "reward/positive_frac": float((r > 0).mean()),
        }
        for k in self._BREAKDOWN_KEYS:
            arr = self._buf[k]
            if arr:
                metrics[f"reward_breakdown/{k}"] = float(np.mean(arr))
        if self._buf["n_tool_calls"]:
            metrics["agent/mean_tool_calls"]     = float(np.mean(self._buf["n_tool_calls"]))
        if self._buf["completion_len"]:
            metrics["agent/mean_completion_len"] = float(np.mean(self._buf["completion_len"]))
        if self._buf["group_reward_std"]:
            metrics["grpo/group_reward_std"]     = float(np.mean(self._buf["group_reward_std"]))

        if wandb.run is not None:
            wandb.log(metrics)

        with open(self.log_file, "a") as fh:
            fh.write(json.dumps({"reward_call": self._call, **metrics}) + "\n")

        logger.info(
            f"[reward] call={self._call} "
            f"r={metrics['reward/mean']:.3f}±{metrics['reward/std']:.3f} "
            f"pos={metrics['reward/positive_frac']:.1%} "
            f"correct={metrics.get('reward_breakdown/correctness', 0):.3f} "
            f"tool_path={metrics.get('reward_breakdown/tool_path', 0):.3f}"
        )
        for lst in self._buf.values():
            lst.clear()


class GRPOLoggingCallback(TrainerCallback):
    """
    Captures per-step training diagnostics (grad_norm, KL, entropy, LR)
    from the Transformers/TRL trainer and writes them to wandb + JSONL.

    Policy entropy is estimated via a forward pass on a small held-out
    batch every `entropy_every` global steps.
    """

    _TRL_METRIC_MAP = {
        "grad_norm":    "train/grad_norm",
        "learning_rate":"train/lr",
        "kl":           "train/kl",
        "loss":         "train/loss",
        "clip_ratio":   "grpo/clip_ratio",
        "reward":       "reward/trl_mean",
        "reward_std":   "reward/trl_std",
        "entropy":      "train/entropy",
    }

    def __init__(
        self,
        log_file:      str = "grpo_metrics.jsonl",
        entropy_every: int = 50,
        entropy_texts: list | None = None,
    ):
        self.log_file       = log_file
        self.entropy_every  = entropy_every
        self._entropy_texts = entropy_texts or []
        self._model         = None
        self._tokenizer     = None
        self._device        = None

    def set_model_refs(self, model, tokenizer, device):
        self._model     = model
        self._tokenizer = tokenizer
        self._device    = device

    def on_log(self, args, state: TrainerState, control: TrainerControl,
               logs: dict = None, **kwargs):
        if not logs:
            return
        metrics = {dst: logs[src] for src, dst in self._TRL_METRIC_MAP.items() if src in logs}
        if not metrics:
            return
        if wandb.run is not None:
            wandb.log(metrics)
        with open(self.log_file, "a") as fh:
            fh.write(json.dumps({"global_step": state.global_step, **metrics}) + "\n")
        logger.info(
            f"[trainer] step={state.global_step} "
            + "  ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in metrics.items())
        )

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Estimate and log policy entropy every `entropy_every` steps."""
        if (
            self._model is None
            or not self._entropy_texts
            or state.global_step % self.entropy_every != 0
            or state.global_step == 0
        ):
            return
        entropy = self._estimate_entropy()
        metrics = {"train/policy_entropy": entropy}
        if wandb.run is not None:
            wandb.log(metrics)
        with open(self.log_file, "a") as fh:
            fh.write(json.dumps({"global_step": state.global_step, **metrics}) + "\n")
        logger.info(f"[trainer] step={state.global_step} policy_entropy={entropy:.4f}")

    def _estimate_entropy(self) -> float:
        """Mean per-token softmax entropy over a small sample."""
        self._model.eval()
        entropies: list[float] = []
        try:
            with torch.no_grad():
                for text in self._entropy_texts[:4]:
                    inputs = self._tokenizer(
                        text, return_tensors="pt", truncation=True, max_length=256,
                    ).to(self._device)
                    logits = self._model(**inputs).logits        # [1, T, V]
                    probs  = torch.softmax(logits[0], dim=-1)   # [T, V]
                    ent    = -(probs * probs.log().clamp(min=-1e9)).sum(-1).mean()
                    entropies.append(ent.item())
        except Exception as e:
            logger.debug(f"entropy estimation failed: {e}")
        finally:
            self._model.train()
        return float(np.mean(entropies)) if entropies else 0.0


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
            lr_scheduler_type           = "cosine",     # decay LR instead of plateau
            weight_decay                = cfg.weight_decay,
            max_grad_norm               = cfg.max_grad_norm,
            warmup_steps                = cfg.warmup_steps,
            logging_steps               = cfg.logging_steps,
            save_steps                  = cfg.save_steps,
            save_strategy               = "steps",
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
        # EpisodeDataset sets levels and curriculum weights from stage internally.
        ep_ds = EpisodeDataset(seed=cfg.seed, stage=cfg.stage)

        # We pre-generate a large dataset; GRPOTrainer will shuffle + repeat.
        n_samples = max(cfg.total_steps * cfg.batch_size, 2000)
        logger.info(f"Generating {n_samples} training samples …")
        self.train_dataset = build_hf_dataset(ep_ds, n_samples)

        # Eval set (always all levels)
        eval_ds = EpisodeDataset(seed=99999, stage=3)
        eval_ds.set_levels([1, 2, 3])
        self.eval_dataset = build_hf_dataset(eval_ds, 200)

        logger.info(f"Train: {len(self.train_dataset)} rows | Eval: {len(self.eval_dataset)} rows")

        # ── Reward function + metrics tracker ─────────────────────────
        self.metrics_tracker = MetricsTracker(
            log_every = cfg.logging_steps,
            log_file  = os.path.join(cfg.output_dir, "grpo_metrics.jsonl"),
        )
        self.reward_fn = RewardFunctionFactory(cfg, tracker=self.metrics_tracker)

        # Load model eagerly with trust_remote_code=True so custom model types
        # like qwen3_5 do not fail in TRL's AutoConfig path.
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        else:
            model_kwargs["torch_dtype"] = torch.float32
        if cfg.load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        base_model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)

        # ── Load SFT LoRA adapter if provided ─────────────────────────
        if cfg.sft_adapter_path:
            logger.info(f"Loading SFT LoRA adapter from {cfg.sft_adapter_path} …")
            base_model = PeftModel.from_pretrained(
                base_model,
                cfg.sft_adapter_path,
                is_trainable=True,
            )
            # Model is already a PeftModel; don't let TRL add another LoRA on top.
            peft_config = None
            logger.info("SFT adapter loaded — GRPO will continue training the existing LoRA weights.")

        # ── MultiStepGRPOTrainer ───────────────────────────────────────
        # rollout_func replaces TRL's single-shot generation with interactive
        # environment rollouts; the cached _current_inputs give it DB access.
        self.trainer = MultiStepGRPOTrainer(
            model           = base_model,
            args            = self.trl_cfg,
            train_dataset   = self.train_dataset,
            eval_dataset    = self.eval_dataset,
            processing_class= self.tokenizer,
            reward_funcs    = self._wrapped_reward_fn,
            peft_config     = peft_config,
            rollout_func    = make_interactive_rollout_fn(cfg),
        )

        # Qwen3.5 caches rope_deltas on the module. During GRPO generation and
        # scoring, effective batch sizes can differ, which can leave a stale
        # cache with incompatible shape for the next forward pass.
        self._rope_reset_hooks.append(_install_qwen_rope_reset_hook(self.trainer.model))
        self._rope_reset_hooks.append(_install_qwen_rope_reset_hook(getattr(self.trainer, "ref_model", None)))

        # ── Logging callback ───────────────────────────────────────────
        # Use a handful of eval prompts as fixed entropy reference texts.
        entropy_texts = [
            row["prompt"] for row in self.eval_dataset.select(range(min(4, len(self.eval_dataset))))
        ]
        self.logging_cb = GRPOLoggingCallback(
            log_file      = os.path.join(cfg.output_dir, "grpo_metrics.jsonl"),
            entropy_every = max(cfg.logging_steps * 5, 50),
            entropy_texts = entropy_texts,
        )
        self.logging_cb.set_model_refs(self.trainer.model, self.tokenizer, self.device)
        self.trainer.add_callback(self.logging_cb)

        # Ensure output directory exists for log files
        os.makedirs(cfg.output_dir, exist_ok=True)

        logger.info("GRPOTrainer ready.")

    # ──────────────────────────────────────────────────────────────────
    def _wrapped_reward_fn(self, prompts, completions, **batch):
        """
        Adapter between TRL's reward_fn signature and our RewardFunctionFactory.

        TRL passes the *entire batch dict* as **kwargs, so gt_json / db_json /
        query columns are available here automatically.
        """
        return self.reward_fn(
            prompts         = prompts,
            completions     = completions,
            trajectory_json = batch.get("trajectory_json"),   # fix Issue 3: was silently dropped
            gt_json         = batch.get("gt_json"),
            db_json         = batch.get("db_json"),
            query           = batch.get("query"),
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
    parser.add_argument("--model",        default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--stage",        type=int,   default=2)
    parser.add_argument("--group_size",   type=int,   default=8)
    parser.add_argument("--total_steps",  type=int,   default=2000)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=5e-7)
    parser.add_argument("--kl_coef",      type=float, default=0.1)
    parser.add_argument("--lora_r",       type=int,   default=8)
    parser.add_argument("--output_dir",   default="checkpoints/grpo2")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument(
        "--sft_adapter",
        default=None,
        help="Path to an SFT-warmed LoRA adapter directory to warm-start GRPO from.",
    )
    args = parser.parse_args()

    return GRPOStudentConfig(
        model_name        = args.model,
        stage             = args.stage,
        group_size        = args.group_size,
        total_steps       = args.total_steps,
        batch_size        = args.batch_size,
        lr                = args.lr,
        kl_coef           = args.kl_coef,
        lora_r            = args.lora_r,
        lora_alpha        = args.lora_r * 2,
        output_dir        = args.output_dir,
        load_in_4bit      = args.load_in_4bit,
        seed              = args.seed,
        sft_adapter_path  = args.sft_adapter,
    )


if __name__ == "__main__":
    cfg     = parse_args()
    trainer = GRPOStudentTrainer(cfg)
    trainer.train()
    trainer.evaluate(n=50)