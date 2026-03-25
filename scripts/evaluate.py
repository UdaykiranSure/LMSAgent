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
"""

import argparse
import yaml
import json
import logging
from datetime import date, timedelta
from collections import defaultdict

import torch

from src.mock_db import MockStudentDB
from src.oracle import Oracle
from src.environment import StudentAgentEnvironment
from src.reward import RuleBasedRewardModel
from src.dataset import QUERY_TEMPLATES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(checkpoint: str, config_path: str, n_per_intent: int = 10):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import AutoModelForCausalLMWithValueHead

    logger.info(f"Loading model from {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model     = AutoModelForCausalLMWithValueHead.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    from src.rollout import RolloutCollector
    reward_model = RuleBasedRewardModel(config["reward"])
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collector    = RolloutCollector(model, tokenizer, reward_model, config, device)
    oracle       = Oracle()

    # ── Build fixed eval set ─────────────────────────────────────────
    # One episode per intent × n_per_intent, fixed seed
    import random
    rng   = random.Random(42)
    today = date(2025, 9, 15)   # fixed date for reproducibility

    by_intent  = defaultdict(list)
    by_level   = defaultdict(list)
    all_scores = []

    for template in QUERY_TEMPLATES:
        for _ in range(n_per_intent):
            db      = MockStudentDB(today + timedelta(days=rng.randint(0, 14)))
            subject = rng.choice(db.get_subjects()) if template.needs_subject else None
            query   = template.template
            if subject:
                query = query.replace("{subject}", subject)

            gt = oracle.derive(query, db, today)

            exps = collector.collect(
                queries       = [query],
                mock_dbs      = [db],
                ground_truths = [gt],
            )
            exp = exps[0]

            record = {
                "query":        query,
                "intent":       template.intent,
                "level":        template.level,
                "reward":       exp.reward,
                "steps_taken":  exp.steps_taken,
                "optimal_steps":gt.optimal_steps,
                "exit_type":    exp.exit_type,
                "tool_calls":   exp.tool_calls_made,
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
    print(f"EVALUATION REPORT — {checkpoint}")
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
            f"order={m['tool_order_correct']:.2f}  "
            f"efficiency={m['step_efficiency']:.2f}  (n={m['n']})"
        )

    print("\n── Quality metrics ──")
    for k, v in results["quality"].items():
        print(f"  {k:<30s} {v:.3f}")

    # Save full results
    out_path = f"logs/eval_{checkpoint.replace('/', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved → {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/ppo_config.yaml")
    parser.add_argument("--n",          type=int, default=10,
                        help="Episodes per intent type")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.config, args.n)
