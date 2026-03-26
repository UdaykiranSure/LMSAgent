# src/rollout.py
"""
collect_rollouts() — the bridge between the environment and TRL's PPO trainer.

For each query in the batch:
  1. Run the agent (policy model) step-by-step through the environment
  2. Collect (token_ids, log_probs, values, rewards) at each step
  3. Compute per-token rewards: 0 everywhere except terminal step
  4. Return lists that PPOTrainer.step() expects

Key design decisions:
  - We treat each FULL episode as one "response" in PPO terms
  - The entire tool-call chain + RESPOND is concatenated into one sequence
  - Terminal reward from the rule-based model is assigned to the last token
  - Intermediate shaping rewards (from environment) are added at each tool step
"""

import torch
import re
import json
from dataclasses import dataclass
from typing import Optional
from datetime import date

from transformers import PreTrainedTokenizer, PreTrainedModel

from src.environment import StudentAgentEnvironment
from src.reward import RuleBasedRewardModel
from src.oracle import GroundTruth
from src.mock_db import MockStudentDB


@dataclass
class EpisodeExperience:
    """
    Everything PPOTrainer.step() needs for one episode.
    TRL expects flat token lists — we produce one per episode.
    """
    query_ids:       torch.Tensor   # tokenised query (the "prompt")
    response_ids:    torch.Tensor   # tokenised full agent output (all steps)
    reward:          float          # scalar terminal reward
    shaping_rewards: list[float]    # per-step intermediate rewards (for logging)
    steps_taken:     int
    exit_type:       str            # "respond" | "step_limit" | "error"
    tool_calls_made: list[str]      # for logging/debugging


class RolloutCollector:
    """
    Runs a batch of episodes, collects experience for PPO.

    Args:
        model:       the policy (actor) model — used for generation
        tokenizer:   shared tokenizer
        reward_model: RuleBasedRewardModel instance
        config:      full config dict
        device:      torch device
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        reward_model: RuleBasedRewardModel,
        config: dict,
        device: torch.device,
    ):
        self.model        = model
        self.tokenizer    = tokenizer
        self.reward_model = reward_model
        self.config       = config
        self.device       = device
        self.gen_config   = config["ppo"]

    @torch.no_grad()
    def collect(
        self,
        queries:         list[str],
        mock_dbs:        list[MockStudentDB],
        ground_truths:   list[GroundTruth],
        today:           Optional[date] = None,
    ) -> list[EpisodeExperience]:
        """
        Run one episode per (query, db, gt) triple.
        Returns a list of EpisodeExperience — one per query.
        """
        today = today or date.today()
        experiences = []

        for query, db, gt in zip(queries, mock_dbs, ground_truths):
            exp = self._run_episode(query, db, gt, today)
            experiences.append(exp)

        return experiences

    def _run_episode(
        self,
        query:  str,
        db:     MockStudentDB,
        gt:     GroundTruth,
        today:  date,
    ) -> EpisodeExperience:
        """
        Run one full episode:
        agent calls tools until RESPOND or step limit.
        """
        env = StudentAgentEnvironment(db, gt, query, today)

        # Tokenise the initial prompt (query + system prompt)
        prompt_text = self._format_prompt(env.state.context_window)
        query_ids   = self._tokenize(prompt_text)

        # Accumulate all response tokens across steps
        all_response_tokens = []
        shaping_rewards     = []
        exit_type           = "step_limit"

        while not env.state.done:
            # Build current context into a string for generation
            context_text = self._format_context(env.state.context_window)

            # Generate one step
            step_tokens, step_text = self._generate_step(context_text)
            all_response_tokens.extend(step_tokens)

            # Step the environment
            obs = env.step(step_text)

            # Collect intermediate shaping reward
            shaping_r = obs.get("intermediate_reward") or 0.0
            shaping_rewards.append(shaping_r)

            if obs["done"]:
                exit_type = obs["info"].get("reason", "respond")
                break

        # Build trajectory object for the reward model
        trajectory = env.build_trajectory()

        # Score with rule-based reward model
        score_result  = self.reward_model.score(trajectory, gt)
        terminal_reward = float(score_result["total"])

        # Tokenise the full response
        full_response_text = self.tokenizer.decode(
            all_response_tokens, skip_special_tokens=True
        )
        response_ids = torch.tensor(
            all_response_tokens, dtype=torch.long
        )

        return EpisodeExperience(
            query_ids       = query_ids,
            response_ids    = response_ids,
            reward          = terminal_reward,
            shaping_rewards = shaping_rewards,
            steps_taken     = len(shaping_rewards),
            exit_type       = exit_type,
            tool_calls_made = [c["tool"] for c in trajectory["tool_calls"]],
        )

    @torch.no_grad()
    def _generate_step(self, context_text: str) -> tuple[list[int], str]:
        """
        Generate one agent step (one tool call or RESPOND).
        Returns (token_ids, decoded_text).
        """
        inputs = self.tokenizer(
            context_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens   = self.gen_config["max_new_tokens"],
            temperature      = self.gen_config["temperature"],
            top_p            = self.gen_config["top_p"],
            do_sample        = self.gen_config["do_sample"],
            pad_token_id     = self.tokenizer.eos_token_id,
            eos_token_id     = self.tokenizer.eos_token_id,
            # Stop at closing tags so we don't bleed across steps
            # stopping_criteria added below
        )

        # Only new tokens (not the prompt)
        prompt_len    = inputs["input_ids"].shape[1]
        new_token_ids = output[0][prompt_len:].tolist()
        decoded       = self.tokenizer.decode(
            new_token_ids, skip_special_tokens=True
        )

        # Trim to the first complete tag
        decoded = self._trim_to_first_tag(decoded)

        return new_token_ids, decoded

    def _trim_to_first_tag(self, text: str) -> str:
        """Keep only up to the first complete </tool_call> or </respond>."""
        for end_tag in ["</tool_call>", "</respond>"]:
            idx = text.find(end_tag)
            if idx != -1:
                return text[:idx + len(end_tag)]
        return text

    def _format_prompt(self, context_window: list[dict]) -> str:
        """Format the initial system+user turns as a string."""
        parts = []
        for msg in context_window:
            role    = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    def _format_context(self, context_window: list[dict]) -> str:
        """Format the full context including tool results so far."""
        parts = []
        for msg in context_window:
            role    = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "tool":
                parts.append(f"<|tool|> [{msg.get('name','')}]\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    def _tokenize(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )["input_ids"][0]
        return ids


def experiences_to_ppo_batch(
    experiences: list[EpisodeExperience],
    dense_weight: float = 1.0,
    sparse_weight: float = 0.0,
) -> tuple[list, list, list]:
    """
    Convert EpisodeExperience list into the three lists
    PPOTrainer.step() expects:
        queries:   list of 1-D LongTensors  (prompt token ids)
        responses: list of 1-D LongTensors  (response token ids)
        rewards:   list of scalar Tensors   (one reward per episode)

    Reward composition:
        dense:  sum of intermediate shaping rewards (per-step guidance)
        sparse: terminal reward from rule-based model
        total:  dense_weight * dense + sparse_weight * sparse
    """
    queries   = []
    responses = []
    rewards   = []

    for exp in experiences:
        dense_r  = sum(exp.shaping_rewards) * dense_weight
        sparse_r = exp.reward               * sparse_weight
        # Clamp to [-1, 1] after combining
        total_r  = max(-1.0, min(1.0, dense_r + sparse_r))

        queries.append(exp.query_ids)
        responses.append(exp.response_ids)
        rewards.append(torch.tensor(total_r, dtype=torch.float))

    return queries, responses, rewards
