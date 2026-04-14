"""
train.py — single entry point for all training stages.

Usage:
  # Stage 1: SFT warm-up (generates 500 demos, fine-tunes with HF Trainer)
  python train.py --stage sft

  # Stage 2: PPO with dense rewards, levels 1+2
  python train.py --stage 2

  # Stage 3: PPO with sparse rewards, all levels
  python train.py --stage 3

  # Resume PPO from checkpoint
  python train.py --stage 2 --resume checkpoints/step_1000

  # Quick smoke test (5 episodes, tiny model settings)
  python train.py --stage 2 --smoke-test

Flags:
  --stage       sft | 2 | 3
  --resume      path to checkpoint directory
  --smoke-test  run 5 episodes to verify everything works
  --batch-size  override batch size
  --lr          override learning rate
  --episodes    override total episodes
  --no-lora     disable LoRA (full fine-tuning, needs more VRAM)
  --model       override model name (default: Qwen/Qwen3.5-2B)
  --device      cpu | cuda | cuda:0 (default: auto-detect)
"""

import argparse
import logging
import sys
import os

# Make src importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT_DIR)

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/train.log", mode="a"),
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Student Agent PPO Trainer")
    p.add_argument("--stage",       default="2",    choices=["sft","2","3"])
    p.add_argument("--resume",      default=None,   help="Checkpoint path to resume from")
    p.add_argument("--smoke-test",  action="store_true")
    p.add_argument("--batch-size",  type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--episodes",    type=int,   default=None)
    p.add_argument("--no-lora",     action="store_true")
    p.add_argument("--model",       default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--device",      default=None)
    return p.parse_args()


def run_sft(args):
    """Stage 1: supervised fine-tuning on expert demos."""
    logger.info("=== Stage 1: SFT warm-up ===")
    from src.dataset import EpisodeDataset, MockStudentDB, Oracle, TEMPLATES
    from datetime import date, timedelta
    import random, json, torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    model_name = args.model
    
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate expert trajectories
    logger.info("Generating 500 expert demonstrations...")
    from scripts.gen_demos import generate_demos
    demos   = generate_demos(n=500 if not args.smoke_test else 10)
    dataset = Dataset.from_list([{"text": d} for d in demos])
    logger.info(f"Dataset: {len(dataset)} demos")

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    if not args.no_lora:
        lora = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj","v_proj","k_proj","o_proj"],
            task_type=TaskType.CAUSAL_LM, bias="none",
        )
        model = get_peft_model(model, lora)

    train_args = TrainingArguments(
        output_dir               = "checkpoints/sft",
        num_train_epochs         = 1 if args.smoke_test else 3,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        learning_rate            = 2e-4,
        lr_scheduler_type        = "cosine",
        warmup_ratio             = 0.1,
        logging_steps            = 5,
        save_strategy            = "epoch",
        report_to                = "none",
        fp16                     = False,
    )

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=1024,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]})

    trainer = Trainer(
        model         = model,
        args          = train_args,
        train_dataset = tokenized,
    )
    trainer.train()
    trainer.save_model("checkpoints/sft")
    tokenizer.save_pretrained("checkpoints/sft")
    logger.info("SFT complete → checkpoints/sft")


def run_ppo(args, stage: int):
    """Stage 2 or 3: PPO training."""
    logger.info(f"=== Stage {stage}: PPO training ===")
    os.makedirs("logs",        exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    from src.ppo_core import PPOConfig
    from src.trainer import PPOTrainer

    # Build config — override from CLI flags
    cfg = PPOConfig(
        model_name     = args.model,
        use_lora       = not args.no_lora,
        batch_size     = args.batch_size or (4 if args.smoke_test else 4),
        mini_batch_size= 2 if args.smoke_test else 4,
        total_episodes = args.episodes or (10 if args.smoke_test else 10_000),
        lr             = args.lr or 1e-5,
        eval_every     = 5 if args.smoke_test else 200,
        save_every     = 5 if args.smoke_test else 500,
    )

    trainer = PPOTrainer(cfg)

    if args.resume:
        trainer.load(args.resume)

    trainer.train(stage=stage)


def main():
    args = parse_args()
    os.makedirs("logs", exist_ok=True)

    logger.info(f"Model: {args.model}")
    logger.info(f"Stage: {args.stage}")
    if args.smoke_test:
        logger.info("SMOKE TEST MODE — 10 episodes only")

    if args.stage == "sft":
        run_sft(args)
    elif args.stage in ("2", "3"):
        run_ppo(args, stage=int(args.stage))
    else:
        logger.error(f"Unknown stage: {args.stage}")
        sys.exit(1)


if __name__ == "__main__":
    main()
