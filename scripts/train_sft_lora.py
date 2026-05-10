"""
LoRA SFT training on trajectories from sft_checkpoints/.

Masking strategy:
  - system, user, and tool turns → labels = -100 (masked)
  - assistant turns              → labels = token IDs (trained)

Trajectory format (ChatML):
  <|im_start|>system\n...<|im_end|>\n
  <|im_start|>user\n...<|im_end|>\n
  <|im_start|>assistant\n<tool_call>...</tool_call>\n
  <|im_start|>tool get_subjects\n...<|im_end|>\n
  <|im_start|>assistant\n<respond>...</respond>\n
"""

import os
import sys
import json
import glob
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL    = "Qwen/Qwen2.5-0.5B-Instruct"
CHECKPOINT_DIR   = "sft_data"
OUTPUT_DIR       = "checkpoints/sft_lora2"
MAX_SEQ_LEN      = 2048


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_trajectories(checkpoint_dir: str) -> list[str]:
    """Load all trajectories from every JSON file in checkpoint_dir."""
    trajectories = []
    json_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {checkpoint_dir}")

    for path in json_files:
        with open(path) as f:
            records = json.load(f)
        for rec in records:
            traj = rec.get("trajectory", "")
            if traj:
                trajectories.append(traj)
        logger.info(f"Loaded {len(records)} trajectories from {os.path.basename(path)}")

    logger.info(f"Total trajectories: {len(trajectories)}")
    return trajectories


def tokenize_and_mask(trajectory: str, tokenizer, max_length: int) -> Optional[dict]:
    """
    Tokenize a single trajectory and build labels where:
      - assistant turn tokens  → keep label (model trains on these)
      - all other turn tokens  → label = -100 (masked / ignored)

    Each '<|im_start|>...<|im_end|>' block is a segment. We split on
    '<|im_start|>', re-attach the marker, tokenize the segment, then
    assign labels based on the role prefix.
    """
    # Split into segments; first element after split will be empty string
    raw_parts = trajectory.split("<|im_start|>")
    segments = [p for p in raw_parts if p]  # drop the leading empty string

    all_input_ids: list[int] = []
    all_labels:    list[int] = []

    for seg in segments:
        text = "<|im_start|>" + seg
        is_assistant = seg.startswith("assistant")

        ids = tokenizer.encode(text, add_special_tokens=False)
        all_input_ids.extend(ids)

        if is_assistant:
            all_labels.extend(ids)
        else:
            all_labels.extend([-100] * len(ids))

    # Truncate
    if len(all_input_ids) > max_length:
        all_input_ids = all_input_ids[:max_length]
        all_labels    = all_labels[:max_length]

    # Skip samples that have no trainable tokens at all
    if all(l == -100 for l in all_labels):
        return None

    return {
        "input_ids":      all_input_ids,
        "labels":         all_labels,
        "attention_mask": [1] * len(all_input_ids),
    }


class TrajectoryDataset(TorchDataset):
    def __init__(self, trajectories: list[str], tokenizer, max_length: int):
        self.samples = []
        skipped = 0
        for traj in trajectories:
            item = tokenize_and_mask(traj, tokenizer, max_length)
            if item is None:
                skipped += 1
                continue
            self.samples.append(item)
        if skipped:
            logger.warning(f"Skipped {skipped} trajectories with no trainable tokens")
        logger.info(f"Dataset size: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {k: torch.tensor(v, dtype=torch.long) for k, v in item.items()}


class PaddingCollator:
    """Pad input_ids, attention_mask (with 0) and labels (with -100)."""
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(item["input_ids"].size(0) for item in batch)
        input_ids      = []
        attention_mask = []
        labels         = []

        for item in batch:
            seq_len = item["input_ids"].size(0)
            pad_len = max_len - seq_len

            input_ids.append(
                torch.cat([item["input_ids"],
                           torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            attention_mask.append(
                torch.cat([item["attention_mask"],
                           torch.zeros(pad_len, dtype=torch.long)])
            )
            labels.append(
                torch.cat([item["labels"],
                           torch.full((pad_len,), -100, dtype=torch.long)])
            )

        return {
            "input_ids":      torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels":         torch.stack(labels),
        }


# ── LoRA config ───────────────────────────────────────────────────────────────

def build_lora_model(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    logger.info(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Target the attention projection layers (works for Qwen2, LLaMA, Mistral, etc.)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Required with gradient checkpointing + LoRA so checkpointed blocks get a
    # differentiable input path even though base model weights are frozen.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Keep cache disabled during training (Trainer also toggles this, but set
    # it explicitly on the model to avoid stale config interactions).
    if hasattr(model, "config"):
        model.config.use_cache = False

    model.print_trainable_parameters()
    return model


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LoRA SFT on LMS agent trajectories")
    p.add_argument("--model",         default=DEFAULT_MODEL,  help="HuggingFace model ID")
    p.add_argument("--checkpoint_dir",default=CHECKPOINT_DIR, help="Directory with .json trajectory files")
    p.add_argument("--output_dir",    default=OUTPUT_DIR,     help="Where to save the trained adapter")
    p.add_argument("--max_seq_len",   type=int, default=MAX_SEQ_LEN)
    p.add_argument("--epochs",        type=int, default=3)
    p.add_argument("--batch_size",    type=int, default=2,    help="Per-device batch size")
    p.add_argument("--grad_accum",    type=int, default=8)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--lora_r",        type=int, default=8)
    p.add_argument("--lora_alpha",    type=int, default=32)
    p.add_argument("--lora_dropout",  type=float, default=0.05)
    p.add_argument("--report_to",     default="wandb",         help="wandb / tensorboard / none")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ────────────────────────────────────────────────────────────────
    trajectories = load_trajectories(args.checkpoint_dir)
    dataset = TrajectoryDataset(trajectories, tokenizer, args.max_seq_len)

    # ── Model + LoRA ───────────────────────────────────────────────────────────
    model = build_lora_model(
        args.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # ── Training args ──────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=args.report_to,
        gradient_checkpointing=True,
    )

    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    logger.info("Starting LoRA SFT training...")
    trainer.train()

    # Save the LoRA adapter (not the full model)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"LoRA adapter saved → {args.output_dir}")


if __name__ == "__main__":
    main()
