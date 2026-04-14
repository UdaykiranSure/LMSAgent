"""
src/rollout.py

Connects the StudentAgentEnvironment to the PPO buffer.

For each episode:
  1. Tokenise the initial prompt
  2. While not done:
     a. Format current context → generate one step
     b. Step the environment
     c. Accumulate tokens
  3. Score the completed trajectory with RuleBasedRewardModel
  4. Package everything into an Episode dataclass

Key design: we collect (input_ids, response_mask, old_log_probs, old_values)
during rollout with no_grad, then recompute them with grad during the PPO update.
This is the standard "collect then update" PPO pattern.
"""

import re
import json
import torch
import logging
from datetime import date
from typing import Optional

from src.ppo_core import ActorCritic, Episode, PPOConfig

logger = logging.getLogger(__name__)

# ── Prompt formatting ──────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a student assistant. You have zero prior knowledge about the student.
You must discover all information through tool calls before answering.

Always start with get_subjects() to confirm the subject exists.
Then call get_assignments(), get_grades(), get_schedule() etc as needed.
End with <respond>your answer here</respond>.

Tool call format: <tool_call>{"tool": "name", "params": {...}}</tool_call>
Final answer format: <respond>answer text here</respond>"""


def format_messages(messages: list[dict]) -> str:
    """
    Convert the context window (list of role/content dicts) to a
        single string in Qwen2.5 chat format.

    Qwen2.5-Instruct uses:
      <|im_start|>system\n...<|im_end|>
      <|im_start|>user\n...<|im_end|>
      <|im_start|>assistant\n
    """
    parts = []
    for msg in messages:
        role    = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            # Model's own prior generated text
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        elif role == "tool":
            # Tool result injected by the environment — clearly distinct from assistant
            # Format: [tool_name]\n{json_result}
            name = msg.get("name", "tool")
            parts.append(f"<|im_start|>tool\n[{name}]\n{content}<|im_end|>")
    # Open the next assistant turn
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)

# ── Rollout collector ──────────────────────────────────────────────────

class RolloutCollector:
    """
    Runs episodes and returns filled Episode objects ready for the PPO buffer.
    All generation is done with no_grad.
    """

    def __init__(
        self,
        actor_critic:  ActorCritic,
        tokenizer,
        reward_model,
        cfg:           PPOConfig,
        device:        torch.device,
    ):
        self.ac           = actor_critic
        self.tokenizer    = tokenizer
        self.reward_model = reward_model
        self.cfg          = cfg
        self.device       = device

    @torch.no_grad()
    def collect_episode(
        self,
        n_episode,
        query:    str,
        mock_db,
        gt,
        today:    Optional[date] = None,
        
    ) -> Episode:
        """Run one full episode and return an Episode."""
        today = today or date.today()

        # Import here to avoid circular imports
        from src.environment import StudentAgentEnvironment
        env = StudentAgentEnvironment(mock_db, gt, query, today)

        # ── Build initial prompt ──────────────────────────────────────
        prompt_text  = format_messages(env.state.context_window)
        prompt_ids   = self.tokenizer(
            prompt_text,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 1024,
        )["input_ids"][0]                             # (P,)
        prompt_len   = len(prompt_ids)

        # Running buffers for the full sequence
        all_ids         = prompt_ids.tolist()
        response_tokens = []                          # tokens added after prompt
        shaping_rewards = []

        # ── Step loop ─────────────────────────────────────────────────
        step_count = 0
        logger.info(f"Episode - {n_episode} - Query {query}")
        while not env.state.done:
            step_count += 1
            logger.info(f"Episode step {step_count} starting - Query ")

            # Tokenise current full context
            ctx_text = format_messages(env.state.context_window)
            ctx_ids  = self.tokenizer(
                ctx_text,
                return_tensors = "pt",
                truncation     = True,
                max_length     = 1024,
            )["input_ids"].to(self.device)           # (1, T)
            ctx_mask = torch.ones_like(ctx_ids)

            # Generate one step (one tool call or respond)
            new_token_ids, step_text = self.ac.generate_step(
                ctx_ids, ctx_mask, self.tokenizer
            )
            logger.info(f"Episode step {step_count} generated: {repr(step_text[:100])}")
            response_tokens.extend(new_token_ids)

            # Step environment
            obs = env.step(step_text)
            logger.info(f"Episode step {step_count} env done: {obs['done']}, reason: {obs.get('info', {}).get('reason', 'ongoing')}")
            shaping_rewards.append(obs.get("intermediate_reward") or 0.0)

            if obs["done"]:
                break

        # ── Build full sequence tensor ────────────────────────────────
        full_ids = prompt_ids.tolist() + response_tokens

        # Pad or truncate to max_length
        max_len  = 1024
        if len(full_ids) > max_len:
            # Keep prompt + truncate response from the right
            full_ids = full_ids[:prompt_len] + full_ids[prompt_len:max_len]

        full_ids_t  = torch.tensor(full_ids,  dtype=torch.long)
        attn_mask_t = torch.ones(len(full_ids), dtype=torch.long)

        # Response mask: 1 for response tokens, 0 for prompt
        resp_mask = torch.zeros(len(full_ids), dtype=torch.long)
        resp_mask[prompt_len:] = 1

        # ── Score trajectory ──────────────────────────────────────────
        trajectory   = env.build_trajectory()
        score_result = self.reward_model.score(trajectory, gt)
        terminal_r   = float(score_result["total"])

        # Combine terminal + shaping (shaping weighted by dense_weight)
        # For now kept simple: just terminal reward
        reward = terminal_r

        # ── Collect old log_probs and values (no_grad) ────────────────
        ids_gpu  = full_ids_t.unsqueeze(0).to(self.device)
        mask_gpu = attn_mask_t.unsqueeze(0).to(self.device)
        rm_gpu   = resp_mask.unsqueeze(0).to(self.device)

        old_lp, old_vals = self.ac.get_logprobs_and_values(
            ids_gpu, mask_gpu, rm_gpu
        )
        old_lp   = old_lp.squeeze(0).cpu()
        old_vals = old_vals.squeeze(0).cpu()

        return Episode(
            input_ids       = full_ids_t,
            attention_mask  = attn_mask_t,
            response_mask   = resp_mask,
            old_log_probs   = old_lp,
            old_values      = old_vals,
            reward          = reward,
            steps_taken     = len(shaping_rewards),
            exit_type       = env.state.steps[-1].is_terminal and "respond" or "step_limit"
                              if env.state.steps else "error",
            tool_sequence   = [c["tool"] for c in trajectory.get("tool_calls", [])],
        )

    def collect_batch(
        self,
        queries:  list[str],
        dbs:      list,
        gts:      list,
        today:    Optional[date] = None,
    ) -> list[Episode]:
        episodes = []
        for n_episode, (q, db, gt) in enumerate(zip(queries, dbs, gts)):
            try:
                ep = self.collect_episode(n_episode, q, db, gt, today)
                episodes.append(ep)
            except Exception as e:
                logger.warning(f"Episode failed: {e}")
        return episodes
