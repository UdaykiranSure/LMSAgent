# Student Agent — RL Training Pipeline

## Project structure

```
student_agent/
├── configs/
│   └── ppo_config.yaml          # all hyperparameters in one place
├── src/
│   ├── mock_db.py               # MockStudentDB — randomised per episode
│   ├── oracle.py                # GroundTruth derivation (zero-knowledge)
│   ├── environment.py           # StudentAgentEnvironment — step loop
│   ├── reward.py                # RuleBasedRewardModel
│   ├── dataset.py               # EpisodeDataset — query template sampler
│   ├── rollout.py               # collect_rollouts() — runs N episodes
│   └── trainer.py               # PPOStudentTrainer — main training loop
├── scripts/
│   ├── sft_warmup.py            # Stage 1: supervised fine-tuning
│   ├── train_ppo.py             # Stage 2+3: PPO training entry point
│   └── evaluate.py              # Held-out evaluation
├── checkpoints/                 # saved model weights
└── logs/                        # wandb / tensorboard
```

## Quickstart

```bash
# Stage 1 — SFT warm-up (run once)
python scripts/sft_warmup.py --config configs/ppo_config.yaml

# Stage 2 — PPO dense rewards
python scripts/train_ppo.py --config configs/ppo_config.yaml --stage 2

# Stage 3 — PPO sparse rewards, harder curriculum
python scripts/train_ppo.py --config configs/ppo_config.yaml --stage 3

# Evaluation
python scripts/evaluate.py --checkpoint checkpoints/best --config configs/ppo_config.yaml
```

