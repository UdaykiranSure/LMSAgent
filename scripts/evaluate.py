# scripts/evaluate.py
"""
Comprehensive evaluation against the held-out test suite.

Metrics:
  - mean_reward:          overall score across all queries
  - accuracy_by_level:    per curriculum level
  - accuracy_by_intent:   per intent type (deadline, grades, study_plan …)
  - hallucination_rate:   fraction of episodes with fabricated facts
  - tool_order_correct:   fraction where get_subjects() was first
  - early_exit_precision: correct early exit when no data exists
  - step_efficiency:      mean(optimal_steps / actual_steps)

Usage (checkpoint):
  python scripts/evaluate.py --checkpoint path/to/checkpoint --config configs/ppo_config.yaml

Usage (local API on port 4141):
  python scripts/evaluate.py --local-api [--port 4141] [--model gpt-5.2] [--temperature 0.2]
"""

import sys
import os
import argparse
import yaml
import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from collections import defaultdict
from typing import Optional

# Add parent directory to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mockdb import MockStudentDB
from src.oracle import Oracle
from src.environment import StudentAgentEnvironment
from src.reward import RuleBasedRewardModel
from src.dataset import TEMPLATES
from scripts.vllm_env_chat import (
    call_vllm_chat,
    get_served_models,
    resolve_model_name,
    to_chat_messages,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Local-API episode result ───────────────────────────────────────────

@dataclass
class EvalEpisode:
    """Minimal episode record for evaluation — no tensors required."""
    reward:        float
    steps_taken:   int
    exit_type:     str
    tool_sequence: list = field(default_factory=list)
    trajectory_steps: list = field(default_factory=list)
    final_response: str = ""


# ── Local-API rollout collector ────────────────────────────────────────

class LocalAPIRolloutCollector:
    """
    Runs episodes against a locally served OpenAI-compatible model
    (e.g. vLLM on port 4141) without loading model weights locally.
    """

    def __init__(self, base_url: str, model: str, reward_model, temperature: float = 0.2):
        self.base_url     = base_url.rstrip("/")
        self.model        = model
        self.reward_model = reward_model
        self.temperature  = temperature

    def collect_episode(
        self,
        query:    str,
        mock_db,
        gt,
        today:    Optional[date] = None,
    ) -> EvalEpisode:
        today = today or date.today()
        env   = StudentAgentEnvironment(mock_db, gt, query, today)

        while not env.state.done:
            messages       = to_chat_messages(env.state.context_window)
            assistant_text = call_vllm_chat(
                base_url    = self.base_url,
                model       = self.model,
                messages    = messages,
                temperature = self.temperature,
            )
            env.step(assistant_text)

        trajectory   = env.build_trajectory()
        score_result = self.reward_model.score(trajectory, gt)
        reward       = float(score_result["total"])

        steps      = env.state.steps
        exit_type  = "respond" if (steps and steps[-1].is_terminal) else "step_limit"
        tool_seq   = [c["tool"] for c in env.state.tool_calls_made]

        return EvalEpisode(
            reward        = reward,
            steps_taken   = len(steps),
            exit_type     = exit_type,
            tool_sequence = tool_seq,
            trajectory_steps = [
                {
                    "step_num": s.step_num,
                    "tool_call": s.tool_call,
                    "tool_result": s.tool_result,
                    "response": s.response,
                    "is_terminal": s.is_terminal,
                }
                for s in env.state.steps
            ],
            final_response = env.state.final_response or "",
        )

    def collect_batch(
        self,
        queries: list[str],
        dbs:     list,
        gts:     list,
        today:   Optional[date] = None,
    ) -> list[EvalEpisode]:
        episodes = []
        for q, db, gt in zip(queries, dbs, gts):
            try:
                episodes.append(self.collect_episode(q, db, gt, today))
            except Exception as e:
                logger.warning(f"Episode failed: {e}")
        return episodes


def evaluate(
    config_path:  str,
    checkpoint:   Optional[str] = None,
    local_api:    bool          = False,
    port:         int           = 8000,
    model_name:   str           = "qwen/qwen2.5-0.5B-Instruct",
    temperature:  float         = 0.2,
    n_per_intent: int           = 10,
):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    reward_model = RuleBasedRewardModel(config["reward"])
    oracle       = Oracle()

    if local_api:
        base_url  = f"http://localhost:{port}"
        resolved  = resolve_model_name(base_url, model_name)
        collector = LocalAPIRolloutCollector(
            base_url     = base_url,
            model        = resolved,
            reward_model = reward_model,
            temperature  = temperature,
        )
        label = f"local-api:{base_url} ({resolved})"
        logger.info(f"Evaluating via local API at {base_url}, model={resolved}")
    else:
        if not checkpoint:
            raise ValueError("Either --checkpoint or --local-api must be specified")
        import torch
        from src.rollout import RolloutCollector
        from src.ppo_core import ActorCritic, PPOConfig
        from transformers import AutoTokenizer
        logger.info(f"Loading model from {checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        ppo_cfg   = PPOConfig(**{k: v for k, v in config.items() if hasattr(PPOConfig, k)})
        ac        = ActorCritic(ppo_cfg)
        ac.load(checkpoint)
        ac.eval()
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ac        = ac.to(device)
        collector = RolloutCollector(ac, tokenizer, reward_model, ppo_cfg, device)
        label     = checkpoint

    safe_label = label.replace("/", "_").replace(":", "_").replace(" ", "_")
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/trajectory_steps_{safe_label}.log"
    with open(log_path, "w") as log:
        log.write(json.dumps({
            "event": "eval_start",
            "label": label,
            "config_path": config_path,
            "local_api": local_api,
            "port": port,
            "model_name": model_name,
            "temperature": temperature,
            "n_per_intent": n_per_intent,
        }) + "\n")

    # ── Build fixed eval set ─────────────────────────────────────────
    # One episode per intent × n_per_intent, fixed seed
    import random
    rng   = random.Random(42)
    today = date(2025, 9, 15)   # fixed date for reproducibility

    by_intent  = defaultdict(list)
    by_level   = defaultdict(list)
    all_scores = []

    for template in TEMPLATES:
        for _ in range(n_per_intent):
            db      = MockStudentDB(today + timedelta(days=rng.randint(0, 14)))
            subject = rng.choice(db.get_subjects()) if template.needs_subject else None
            query   = template.template
            if subject:
                query = query.replace("{subject}", subject)

            gt = oracle.derive(query, db, today)

            exps = collector.collect_batch(
                queries = [query],
                dbs     = [db],
                gts     = [gt],
            )
            if not exps:
                logger.warning(
                    "Skipping episode with no successful rollout for query=%r intent=%s",
                    query,
                    template.intent,
                )
                continue
            exp = exps[0]


            # Log each trajectory step as a JSON line to the named log file
            step_logs = getattr(exp, "trajectory_steps", [])
            with open(log_path, "a") as log:
                for step in step_logs:
                    log.write(json.dumps({
                        "event": "trajectory_step",
                        "query": query,
                        "intent": template.intent,
                        "level": template.level,
                        "step": step,
                    }, default=str) + "\n")

            record = {
                "query":        query,
                "intent":       template.intent,
                "level":        template.level,
                "reward":       exp.reward,
                "steps_taken":  exp.steps_taken,
                "optimal_steps":gt.optimal_steps,
                "exit_type":    exp.exit_type,
                "tool_calls":   exp.tool_sequence,
                "answer_type":  gt.answer_type,
            }
            by_intent[template.intent].append(record)
            by_level[template.level].append(record)
            all_scores.append(exp.reward)

    # ── Compute metrics ───────────────────────────────────────────────
    def mean(xs): return sum(xs) / len(xs) if xs else 0.0

    results = {
        "overall": {
            "mean_reward":     mean(all_scores),
            "n_episodes":      len(all_scores),
        },
        "by_level":  {},
        "by_intent": {},
        "quality":   {},
    }

    for level, records in sorted(by_level.items()):
        rewards = [r["reward"] for r in records]
        results["by_level"][f"L{level}"] = {
            "mean_reward": round(mean(rewards), 3),
            "n":           len(records),
        }

    for intent, records in sorted(by_intent.items()):
        rewards   = [r["reward"] for r in records]
        # Tool order: get_subjects must be first call
        order_ok  = [r["tool_calls"][0] == "get_subjects"
                     for r in records if r["tool_calls"]]
        # Step efficiency: optimal / actual
        efficiency = [
            min(1.0, r["optimal_steps"] / max(1, r["steps_taken"]))
            for r in records
        ]
        results["by_intent"][intent] = {
            "mean_reward":     round(mean(rewards), 3),
            "tool_order_ok":   round(mean(order_ok), 3),
            "step_efficiency": round(mean(efficiency), 3),
            "n":               len(records),
        }

    # Quality metrics across all episodes
    all_records = [r for recs in by_intent.values() for r in recs]

    hallucinated  = sum(1 for r in all_records if r["reward"] <= -0.8)
    step_limit    = sum(1 for r in all_records if r["exit_type"] == "step_limit")
    clean_respond = sum(1 for r in all_records if r["exit_type"] == "respond")
    order_ok_all  = sum(1 for r in all_records
                        if r["tool_calls"] and r["tool_calls"][0] == "get_subjects")

    n = len(all_records)
    results["quality"] = {
        "hallucination_rate":    round(hallucinated  / n, 3),
        "step_limit_rate":       round(step_limit    / n, 3),
        "clean_respond_rate":    round(clean_respond / n, 3),
        "tool_order_correct":    round(order_ok_all  / n, 3),
        "mean_step_efficiency":  round(mean([
            min(1.0, r["optimal_steps"] / max(1, r["steps_taken"]))
            for r in all_records
        ]), 3),
    }

    # ── Print report ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"EVALUATION REPORT — {label}")
    print("="*60)
    print(f"\nOverall mean reward: {results['overall']['mean_reward']:.3f}")
    print(f"Total episodes:      {results['overall']['n_episodes']}")

    print("\n── By curriculum level ──")
    for lvl, m in results["by_level"].items():
        print(f"  {lvl}: reward={m['mean_reward']:.3f}  (n={m['n']})")

    print("\n── By intent ──")
    for intent, m in results["by_intent"].items():
        print(
            f"  {intent:<25s} reward={m['mean_reward']:.3f}  "
            f"order={m['tool_order_ok']:.2f}  "
            f"efficiency={m['step_efficiency']:.2f}  (n={m['n']})"
        )

    print("\n── Quality metrics ──")
    for k, v in results["quality"].items():
        print(f"  {k:<30s} {v:.3f}")

    # Save full results
    out_path   = f"logs/eval_{safe_label}.json"
    os.makedirs("logs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved → {out_path}")
    print(f"Trajectory step log saved → {log_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a student-agent model against the held-out test suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # local checkpoint
  python scripts/evaluate.py --checkpoint sft_checkpoints/model --config configs/ppo_config.yaml

  # local vLLM / llama.cpp server on port 4141
  python scripts/evaluate.py --local-api --port 4141 --model gpt-5.2
""",
    )
    parser.add_argument("--checkpoint",   default=None,
                        help="Path to a local HuggingFace checkpoint (mutually exclusive with --local-api)")
    parser.add_argument("--local-api",    action="store_true",
                        help="Evaluate against a locally running OpenAI-compatible model server")
    parser.add_argument("--port",         type=int, default=8000,
                        help="Port of the local model server (default: 4141)")
    parser.add_argument("--model",        default="qwen/qwen2.5-0.5B-Instruct",
                        help="Model name served by the local API ")
    parser.add_argument("--temperature",  type=float, default=0.2,
                        help="Sampling temperature for the local API (default: 0.2)")
    parser.add_argument("--config",       default="configs/ppo_config.yaml",
                        help="Path to the reward/training config YAML")
    parser.add_argument("--n",            type=int, default=10,
                        help="Episodes per intent type")
    args = parser.parse_args()

    if not args.local_api and not args.checkpoint:
        parser.error("Specify either --checkpoint <path> or --local-api")

    evaluate(
        config_path  = args.config,
        checkpoint   = args.checkpoint,
        local_api    = args.local_api,
        port         = args.port,
        model_name   = args.model,
        temperature  = args.temperature,
        n_per_intent = args.n,
    )
