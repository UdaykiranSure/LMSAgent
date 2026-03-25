# scripts/train_ppo.py
"""
PPO training entry point.

Stage 2: dense rewards, levels 1+2, smaller query budget
Stage 3: sparse rewards, all levels, harder curriculum

The only difference between stages is the reward weighting and
curriculum level distribution — same code, different config values.
"""

import argparse
import logging
import yaml
import torch

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def patch_config_for_stage(config: dict, stage: int) -> dict:
    """Override config values for the given stage."""
    if stage == 3:
        # Stage 3: mostly sparse rewards, harder curriculum
        config["reward"]["dense_reward_weight"]  = 0.2
        config["reward"]["sparse_reward_weight"] = 1.0
        # Tighten KL — model is already good, don't let it drift
        config["ppo"]["init_kl_coef"] = 0.1
        config["ppo"]["target_kl"]    = 4.0
    return config


def main():
    parser = argparse.ArgumentParser(description="PPO training for student agent")
    parser.add_argument("--config", default="configs/ppo_config.yaml")
    parser.add_argument("--stage", type=int, default=2, choices=[2, 3])
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config = patch_config_for_stage(config, args.stage)

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    from src.trainer import PPOStudentTrainer
    trainer = PPOStudentTrainer(config_path=args.config, stage=args.stage)

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        # TRL PPOTrainer doesn't have native resume, but we can reload weights
        from peft import PeftModel
        trainer.model.pretrained_model = PeftModel.from_pretrained(
            trainer.model.pretrained_model, args.resume
        )

    trainer.train()


if __name__ == "__main__":
    main()
