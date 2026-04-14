"""
src/ppo_core.py

Pure PyTorch PPO implementation.
No TRL, no external RL libraries.

What this implements:
  - ActorCritic: Qwen3.5-2B backbone + LoRA + scalar value head
  - ReferenceModel: frozen copy of the SFT weights for KL penalty
  - PPOBuffer: stores one batch of episodes before the update
  - PPOUpdater: runs K epochs of clipped PPO + value loss + KL penalty
  - AdaptiveKLController: adjusts beta to keep KL near target

  
  
PPO objective (per token):
  L = E[ min(r_t * A_t,  clip(r_t, 1-eps, 1+eps) * A_t) ]
    - beta * KL(pi_theta || pi_ref)
    - c1 * (V_theta(s) - V_target)^2

where:
  r_t    = exp(log_prob_new - log_prob_old)   probability ratio
  A_t    = GAE advantage
  beta   = adaptive KL coefficient
  c1     = value function coefficient
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _token_logprobs_from_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute log p(target_ids | logits) without allocating full log_softmax(logits).

    logits:     (B, T, V)
    target_ids: (B, T)
    returns:    (B, T)
    """
    gathered = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    normalizer = torch.logsumexp(logits, dim=-1)
    return gathered - normalizer


# ──────────────────────────────────────────────────────────────────────
# PPO Hyperparameters
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    # Model
    model_name:          str   = "Qwen/Qwen3.5-2B"
    load_in_4bit:        bool  = False

    # LoRA
    use_lora:            bool  = True
    lora_r:              int   = 8
    lora_alpha:          int   = 16
    lora_dropout:        float = 0.05
    lora_target_modules: list  = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj"
    ])

    # PPO core
    lr:                  float = 1e-5
    eps_clip:            float = 0.2       # clamp ratio to [1-eps, 1+eps]
    value_clip:          float = 0.2       # clamp value delta
    gamma:               float = 0.99      # discount
    lam:                 float = 0.95      # GAE lambda
    vf_coef:             float = 0.1       # value loss weight
    entropy_coef:        float = 0.01      # entropy bonus (exploration)
    max_grad_norm:       float = 0.5       # gradient clipping

    # KL penalty against reference model
    init_kl_coef:        float = 0.05
    target_kl:           float = 6.0       # nats — adaptive controller targets this
    kl_horizon:          int   = 10_000    # steps over which controller acts

    # Batch structure
    batch_size:          int   = 16        # episodes per PPO iteration
    mini_batch_size:     int   = 4         # episodes per gradient step
    ppo_epochs:          int   = 4         # gradient epochs per collected batch

    # Generation
    max_new_tokens:      int   = 500
    temperature:         float = 0.8
    top_p:               float = 0.9

    # Training
    total_episodes:      int   = 10_000
    eval_every:          int   = 200
    save_every:          int   = 500
    seed:                int   = 42


# ──────────────────────────────────────────────────────────────────────
# Value Head — appended to the LLM backbone
# ──────────────────────────────────────────────────────────────────────

class ValueHead(nn.Module):
    """
    Scalar value estimator V(s).
    Takes the last hidden state of the LLM → single scalar.

    Architecture: LayerNorm → Linear(hidden, 1024) → GELU → Linear(1024, 1)
    The intermediate layer gives the head capacity without overfitting.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm   = nn.LayerNorm(hidden_size)
        self.dense  = nn.Linear(hidden_size, 1024)
        self.act    = nn.GELU()
        self.out    = nn.Linear(1024, 1)

        # Initialise output near zero — value estimates start neutral
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        hidden: (batch, seq_len, hidden_size)
        returns: (batch, seq_len)  — one value per token position
        """
        x = self.norm(hidden)
        x = self.act(self.dense(x))
        return self.out(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────
# ActorCritic — Qwen3.5-2B + LoRA + ValueHead
# ──────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Wraps Qwen3.5-2B as both actor (policy) and critic (value estimator).

    - Actor:  standard LM head → token log-probabilities
    - Critic: ValueHead on last hidden state → scalar V(s)

    LoRA is applied to the backbone so we train far fewer parameters.
    The value head is always trained in full precision (float32).
    """

    def __init__(self, cfg: PPOConfig, device: torch.device):
        super().__init__()
        self.cfg    = cfg
        self.device = device

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        logger.info(f"Loading backbone: {cfg.model_name}")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype    = torch.bfloat16,   # fp32 for stability on CPU/small GPU
            device_map     = "auto",
        )

        if cfg.use_lora:
            lora_config = LoraConfig(
                r               = cfg.lora_r,
                lora_alpha      = cfg.lora_alpha,
                lora_dropout    = cfg.lora_dropout,
                target_modules  = cfg.lora_target_modules,
                task_type       = TaskType.CAUSAL_LM,
                bias            = "none",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.print_trainable_parameters()

        # Value head: same hidden size as the model.
        # Keep critic math in fp32 for stability and to avoid bf16/float mismatches.
        hidden_size = self.backbone.config.hidden_size
        self.value_head = ValueHead(hidden_size).to(device=device, dtype=torch.float32)

        logger.info(f"Model ready. Hidden size: {hidden_size}")

    def forward(
        self,
        input_ids:      torch.Tensor,           # (B, T)
        attention_mask: torch.Tensor,           # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          logits: (B, T, vocab_size)   — for computing log-probs
          values: (B, T)               — V(s_t) for each position
        """
        out = self.backbone(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            output_hidden_states = True,
            return_dict    = True,
        )
        logits         = out.logits                     # (B, T, V)
        last_hidden    = out.hidden_states[-1]          # (B, T, H)
        # Explicitly align hidden states to the value head dtype (fp32 by design).
        values         = self.value_head(last_hidden.to(torch.float32))   # (B, T)
        return logits, values

    @torch.no_grad()
    def get_logprobs_and_values(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask:  torch.Tensor,   # 1 where response tokens are (not prompt)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Efficient forward for rollout collection (no grad).
        Returns per-token log-probs and values, masked to response tokens only.
        Tensors are padded to match input sequence length.
        """
        logits, values = self.forward(input_ids, attention_mask)

        # log-prob of the actual token at each position, memory-efficiently
        token_ids = input_ids[:, 1:]                            # (B, T-1) next tokens
        token_lp  = _token_logprobs_from_logits(logits[:, :-1, :], token_ids)

        # Align values with response positions (drop last position)
        values_shifted = values[:, :-1]                         # (B, T-1)

        # Apply response mask (shift by 1 to match T-1 predictions)
        resp_mask_shifted = response_mask[:, 1:]                # (B, T-1)

        masked_lp = token_lp * resp_mask_shifted               # (B, T-1)
        masked_vals = values_shifted * resp_mask_shifted        # (B, T-1)

        # Pad back to full sequence length T by prepending zeros at position 0
        # (position 0 has no "previous context" to predict from, so log-prob is undefined)
        batch_size = masked_lp.shape[0]
        pad_lp = torch.cat(
            [torch.zeros(batch_size, 1, device=masked_lp.device, dtype=masked_lp.dtype), masked_lp],
            dim=1
        )
        pad_vals = torch.cat(
            [torch.zeros(batch_size, 1, device=masked_vals.device, dtype=masked_vals.dtype), masked_vals],
            dim=1
        )

        return pad_lp, pad_vals

    @torch.no_grad()
    def generate_step(
        self,
        input_ids:      torch.Tensor,   # (1, T) — single episode
        attention_mask: torch.Tensor,
        tokenizer,
    ) -> tuple[list[int], str]:
        """
        Autoregressively generate one agent step.
        Stops at </tool_call> or </respond> end tag.
        Returns (new_token_ids, decoded_text).
        """
        cfg     = self.cfg
        out_ids = []
        cur_ids = input_ids
        cur_msk = attention_mask

        stop_strings = ["</tool_call>", "</respond>", "<|im_end|>"]

        for _ in range(cfg.max_new_tokens):
            with torch.no_grad():
                outputs = self.backbone(
                    input_ids      = cur_ids,
                    attention_mask = cur_msk,
                    return_dict    = True,
                )
            next_logits = outputs.logits[:, -1, :]   # (1, V)

            # Temperature scaling + top-p sampling
            next_logits = next_logits / cfg.temperature
            next_logits = _top_p_filter(next_logits, cfg.top_p)
            probs       = F.softmax(next_logits, dim=-1)
            next_token  = torch.multinomial(probs, num_samples=1)  # (1,1)

            out_ids.append(next_token.item())

            # Append to running context
            cur_ids = torch.cat([cur_ids, next_token], dim=1)
            cur_msk = torch.cat([cur_msk, torch.ones(1, 1, device=cur_msk.device)], dim=1)

            # Check for stop string
            partial = tokenizer.decode(out_ids, skip_special_tokens=True)
            for stop in stop_strings:
                if stop in partial:
                    # Trim to end of stop string
                    idx = partial.find(stop) + len(stop)
                    partial = partial[:idx]
                    return out_ids, partial

            # EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

        decoded = tokenizer.decode(out_ids, skip_special_tokens=True)
        return out_ids, decoded


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering — zero out tokens outside the top-p probability mass."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative prob above top_p (shift right to keep first token above threshold)
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[sorted_indices_to_remove] = -float("Inf")
    # Scatter back to original ordering
    logits.scatter_(1, sorted_indices, sorted_logits)
    return logits


# ──────────────────────────────────────────────────────────────────────
# Reference Model — frozen SFT weights for KL penalty
# ──────────────────────────────────────────────────────────────────────

class ReferenceModel:
    """
    Frozen copy of the initial model weights.
    Used only to compute KL(pi_theta || pi_ref) per token.
    Never updated — loaded once and kept frozen.
    """

    def __init__(self, model_name: str, device: torch.device):
        from transformers import AutoModelForCausalLM
        logger.info(f"Loading frozen reference model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype = torch.bfloat16,
            device_map  = "auto",
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def get_logprobs(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask:  torch.Tensor,
    ) -> torch.Tensor:
        """Returns per-token log-probs under the reference policy."""
        out       = self.model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            return_dict    = True,
        )
        token_ids = input_ids[:, 1:]
        token_lp  = _token_logprobs_from_logits(out.logits[:, :-1, :], token_ids)
        # Ensure mask is on the same device as model outputs.
        resp_mask = response_mask[:, 1:].to(token_lp.device)
        return token_lp * resp_mask


# ──────────────────────────────────────────────────────────────────────
# PPO Experience Buffer
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Episode:
    """
    Everything collected during one rollout episode.
    Stored before the PPO update.

    All sequence tensors must have the same length T (including padding at position 0).
    Position 0 has zero log-prob and value since there's no prior context to predict from.
    """
    # Tokenised full sequence: [prompt tokens | response tokens]
    input_ids:      torch.Tensor   # (T,) — full sequence
    attention_mask: torch.Tensor   # (T,) — full sequence
    response_mask:  torch.Tensor   # (T,) — 1 for response tokens, 0 for prompt

    # Collected during rollout (no_grad) — all padded to length T
    old_log_probs:  torch.Tensor   # (T,) per-token, masked; position 0 is zero
    old_values:     torch.Tensor   # (T,) per-token, masked; position 0 is zero

    # Scalar reward from reward model (assigned to last response token)
    reward:         float

    # Computed after batch is complete
    advantages:     Optional[torch.Tensor] = None   # (T,)
    returns:        Optional[torch.Tensor] = None   # (T,) = V_target

    # For logging
    steps_taken:    int  = 0
    exit_type:      str  = "unknown"
    tool_sequence:  list = field(default_factory=list)


class PPOBuffer:
    """
    Collects episodes for one PPO iteration, then computes advantages.
    Cleared after each update.
    """

    def __init__(self):
        self.episodes: list[Episode] = []

    def add(self, ep: Episode):
        self.episodes.append(ep)

    def __len__(self):
        return len(self.episodes)

    def compute_advantages(self, gamma: float, lam: float):
        """
        Compute GAE advantages and returns for every episode in the buffer.

        For a single-reward episode (reward at the end):
          rewards[t] = shaping_reward[t] + (terminal_reward if last step else 0)

        GAE:
          delta_t  = r_t + gamma * V(s_{t+1}) - V(s_t)
          A_t      = sum_{k=0}^{T-t} (gamma * lam)^k * delta_t+k
        """
        for ep in self.episodes:
            T        = ep.response_mask.sum().item()
            if T == 0:
                ep.advantages = torch.zeros_like(ep.old_values)
                ep.returns    = torch.zeros_like(ep.old_values)
                continue

            # Build per-position reward vector
            # reward is scalar — assign to the last response token position
            rewards_vec = torch.zeros_like(ep.old_values)
            # Find last response token index
            resp_indices = ep.response_mask.nonzero(as_tuple=True)[0]
            if len(resp_indices) > 0:
                last_idx = resp_indices[-1]
                rewards_vec[last_idx] = ep.reward

            # GAE computation over response positions only
            values   = ep.old_values.float()
            adv      = torch.zeros_like(values)
            last_gae = 0.0

            for t in reversed(range(len(values))):
                if ep.response_mask[t] == 0:
                    last_gae = 0.0
                    continue
                next_val  = values[t + 1].item() if t + 1 < len(values) else 0.0
                delta     = rewards_vec[t].item() + gamma * next_val - values[t].item()
                last_gae  = delta + gamma * lam * last_gae
                adv[t]    = last_gae

            ep.advantages = adv
            ep.returns    = adv + values   # V_target = A + V_old

        # Normalise advantages across the whole buffer (zero mean, unit std)
        all_adv = torch.cat([
            ep.advantages[ep.response_mask.bool()]
            for ep in self.episodes
            if ep.advantages is not None and ep.response_mask.sum() > 0
        ])
        if len(all_adv) > 1:
            adv_mean = all_adv.mean()
            adv_std  = all_adv.std().clamp(min=1e-8)
            for ep in self.episodes:
                if ep.advantages is not None:
                    ep.advantages = (ep.advantages - adv_mean) / adv_std

    def mini_batches(self, mini_batch_size: int):
        """Yield mini-batches of episodes (shuffled each call)."""
        import random
        indices = list(range(len(self.episodes)))
        random.shuffle(indices)
        for start in range(0, len(indices), mini_batch_size):
            batch_idx = indices[start:start + mini_batch_size]
            yield [self.episodes[i] for i in batch_idx]

    def clear(self):
        self.episodes.clear()


# ──────────────────────────────────────────────────────────────────────
# Adaptive KL Controller
# ──────────────────────────────────────────────────────────────────────

class AdaptiveKLController:
    """
    Adjusts KL penalty coefficient beta to keep mean KL near target_kl.

    If current KL > target_kl * 1.5:  increase beta (penalise more)
    If current KL < target_kl / 1.5:  decrease beta (allow more exploration)

    This prevents:
      - KL collapse: policy drifts too far from reference
      - KL stagnation: policy barely moves from reference
    """

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: int):
        self.beta       = init_kl_coef
        self.target_kl  = target_kl
        self.horizon    = horizon

    def update(self, current_kl: float):
        """Call once per PPO iteration with mean KL of the batch."""
        proportional_error = (current_kl - self.target_kl) / self.target_kl
        # Clamp adjustment to ±20% per step
        adjustment = 1.0 + max(-0.2, min(0.2, proportional_error))
        self.beta *= adjustment
        # Clamp beta to reasonable range
        self.beta = max(0.001, min(self.beta, 1.0))
        return self.beta


# ──────────────────────────────────────────────────────────────────────
# PPO Updater — core optimisation step
# ──────────────────────────────────────────────────────────────────────

class PPOUpdater:
    """
    Runs K epochs of clipped PPO updates on a filled PPOBuffer.

    Each mini-batch:
      1. Forward pass through ActorCritic → new log_probs, new values
      2. Forward pass through ReferenceModel → ref log_probs
      3. Compute per-token KL penalty
      4. Compute clipped policy loss
      5. Compute value function loss (clipped)
      6. Entropy bonus
      7. Backprop and clip gradients
    """

    def __init__(
        self,
        actor_critic:  ActorCritic,
        ref_model:     ReferenceModel,
        kl_controller: AdaptiveKLController,
        cfg:           PPOConfig,
        device:        torch.device,
    ):
        self.ac          = actor_critic
        self.ref         = ref_model
        self.kl_ctrl     = kl_controller
        self.cfg         = cfg
        self.device      = device

        # Separate learning rates: backbone (LoRA) lower, value head higher
        self.optimizer = AdamW([
            {"params": self.ac.backbone.parameters(),   "lr": cfg.lr},
            {"params": self.ac.value_head.parameters(), "lr": cfg.lr * 5},
        ], weight_decay=0.01)

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max  = cfg.total_episodes,
            eta_min= cfg.lr * 0.1,
        )

    def update(self, buffer: PPOBuffer) -> dict:
        """
        Run ppo_epochs over the buffer. Returns dict of mean losses.
        """
        buffer.compute_advantages(self.cfg.gamma, self.cfg.lam)

        stats = {
            "policy_loss": [], "value_loss":  [],
            "entropy":     [], "kl":          [],
            "approx_kl":   [], "clip_frac":   [],
        }

        for epoch in range(self.cfg.ppo_epochs):
            for mini_batch in buffer.mini_batches(self.cfg.mini_batch_size):
                step_stats = self._update_step(mini_batch)
                for k, v in step_stats.items():
                    stats[k].append(v)

        # Update KL controller with mean KL of this batch
        mean_kl = np.mean(stats["kl"]) if stats["kl"] else 0.0
        new_beta = self.kl_ctrl.update(mean_kl)

        self.scheduler.step()

        return {k: float(np.mean(v)) for k, v in stats.items()} | {
            "kl_coef": new_beta,
            "lr":      self.optimizer.param_groups[0]["lr"],
        }

    def _update_step(self, mini_batch: list[Episode]) -> dict:
        """One gradient step on a mini-batch of episodes."""
        cfg = self.cfg
        beta = self.kl_ctrl.beta

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        total_kl          = 0.0
        n_tokens          = 0
        approx_kl_list    = []
        clip_frac_list    = []

        total_batch_tokens = sum(int(ep.response_mask[1:].sum().item()) for ep in mini_batch)
        if total_batch_tokens == 0:
            return {"policy_loss":0,"value_loss":0,"entropy":0,"kl":0,"approx_kl":0,"clip_frac":0}

        self.optimizer.zero_grad(set_to_none=True)

        for ep in mini_batch:
            ids   = ep.input_ids.unsqueeze(0).to(self.device)       # (1, T)
            mask  = ep.attention_mask.unsqueeze(0).to(self.device)  # (1, T)
            rmask = ep.response_mask.to(self.device)                # (T,)

            if rmask.sum() == 0:
                continue

            # ── Forward: actor-critic ──────────────────────────────────
            logits, values = self.ac(ids, mask)
            # log-probs of actual next tokens
            tgt_ids  = ids[:, 1:]                                   # (1,T-1)
            new_lp   = _token_logprobs_from_logits(logits[:, :-1, :], tgt_ids).squeeze(0)  # (T-1,)
            new_vals = values[:, :-1].squeeze(0)                    # (T-1,)
            rmask_s  = rmask[1:]                                    # (T-1,) aligned

            # ── Forward: reference model for KL ───────────────────────
            ref_lp = self.ref.get_logprobs(ids, mask, ep.response_mask.unsqueeze(0))
            ref_lp = ref_lp.squeeze(0)                             # (T-1,)

            # Select only response token positions
            resp_pos = rmask_s.bool()
            if resp_pos.sum() == 0:
                continue
            resp_pos_cpu = resp_pos.cpu()

            new_lp_resp  = new_lp[resp_pos]
            new_val_resp = new_vals[resp_pos]
            ref_lp_resp  = ref_lp[resp_pos]
            old_lp_resp  = ep.old_log_probs[1:][resp_pos_cpu].to(self.device)
            old_val_resp = ep.old_values[1:][resp_pos_cpu].to(self.device)
            adv_resp     = ep.advantages[1:][resp_pos_cpu].to(self.device)
            ret_resp     = ep.returns[1:][resp_pos_cpu].to(self.device)

            n = resp_pos.sum().item()
            n_tokens += n

            # ── Policy loss (clipped PPO) ──────────────────────────────
            # ratio = pi_new(a|s) / pi_old(a|s)
            log_ratio  = new_lp_resp - old_lp_resp
            ratio      = torch.exp(log_ratio)

            # Approx KL for diagnostics: E[ratio - 1 - log_ratio]
            approx_kl  = ((ratio - 1) - log_ratio).mean().item()
            approx_kl_list.append(approx_kl)

            # Clipping
            clipped    = torch.clamp(ratio, 1 - cfg.eps_clip, 1 + cfg.eps_clip)
            clip_frac  = (ratio != clipped).float().mean().item()
            clip_frac_list.append(clip_frac)

            pg_loss1   = -adv_resp * ratio
            pg_loss2   = -adv_resp * clipped
            policy_loss= torch.max(pg_loss1, pg_loss2).mean()

            # ── Value loss (clipped) ───────────────────────────────────
            # Clipping prevents large value updates from destabilising training
            val_clipped  = old_val_resp + torch.clamp(
                new_val_resp - old_val_resp, -cfg.value_clip, cfg.value_clip
            )
            vf_loss1     = (new_val_resp - ret_resp).pow(2)
            vf_loss2     = (val_clipped  - ret_resp).pow(2)
            value_loss   = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

            # ── Entropy bonus ──────────────────────────────────────────
            # Encourages exploration — prevents premature convergence
            probs_resp = F.softmax(
                logits[:, :-1, :].squeeze(0)[resp_pos], dim=-1
            )
            entropy = -(probs_resp * torch.log(probs_resp.clamp(min=1e-8))).sum(-1).mean()

            # ── KL penalty against reference ───────────────────────────
            # per-token KL: p_new * (log_p_new - log_p_ref)
            kl_per_token = (new_lp_resp - ref_lp_resp).mean()
            kl           = kl_per_token.clamp(min=0)   # KL is non-negative

            # ── Total loss ─────────────────────────────────────────────
            total_policy_loss += policy_loss.detach().item() * n
            total_value_loss  += value_loss.detach().item() * n
            total_entropy     += entropy.detach().item() * n
            total_kl          += kl.detach().item() * n

            # Backprop per episode to avoid keeping full mini-batch graphs in memory.
            ep_loss = (
                  policy_loss
                + cfg.vf_coef      * value_loss
                - cfg.entropy_coef * entropy
                + beta             * kl
            )
            (ep_loss * (n / total_batch_tokens)).backward()

        if n_tokens == 0:
            self.optimizer.zero_grad(set_to_none=True)
            return {"policy_loss":0,"value_loss":0,"entropy":0,"kl":0,"approx_kl":0,"clip_frac":0}

        nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": (total_policy_loss / n_tokens),
            "value_loss":  (total_value_loss  / n_tokens),
            "entropy":     (total_entropy     / n_tokens),
            "kl":          (total_kl          / n_tokens),
            "approx_kl":   float(np.mean(approx_kl_list)) if approx_kl_list else 0.0,
            "clip_frac":   float(np.mean(clip_frac_list)) if clip_frac_list else 0.0,
        }
