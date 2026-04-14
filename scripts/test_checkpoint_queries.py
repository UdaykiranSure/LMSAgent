import argparse
import json
from pathlib import Path
from datetime import date

import torch
from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset import MockStudentDB, Oracle
from src.environment import StudentAgentEnvironment
from src.reward import RuleBasedRewardModel
from src.rollout import format_messages


DEFAULT_QUERIES = [
    "What subjects am I taking this semester?",
    "What is my grade in RL?",
    "Do I have any deadlines this week?",
]


def _resolve_dtype(name: str) -> torch.dtype | str:
    name = name.lower()
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _load_model_and_tokenizer(checkpoint: str, device_map: str, dtype_name: str):
    checkpoint_path = Path(checkpoint)
    adapter_cfg_path = checkpoint_path / "adapter_config.json"
    dtype = _resolve_dtype(dtype_name)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if adapter_cfg_path.exists():
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                checkpoint,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
        except Exception:
            adapter_cfg = json.loads(adapter_cfg_path.read_text())
            base_model_name = adapter_cfg.get("base_model_name_or_path")
            if not base_model_name:
                raise ValueError("adapter_config.json is missing base_model_name_or_path")

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def _build_inputs(tokenizer, query: str, model):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": query}]
        chat_encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(chat_encoded, torch.Tensor):
            return {"input_ids": chat_encoded.to(next(model.parameters()).device)}
        return {k: v.to(next(model.parameters()).device) for k, v in chat_encoded.items()}

    encoded = tokenizer(query, return_tensors="pt")
    return {k: v.to(next(model.parameters()).device) for k, v in encoded.items()}


def run_queries(
    checkpoint: str,
    queries: list[str],
    max_new_tokens: int,
    max_prompt_length: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    device_map: str,
    dtype_name: str,
    in_environment: bool,
    max_env_steps: int | None,
    db_seed: int | None,
):
    model, tokenizer = _load_model_and_tokenizer(checkpoint, device_map, dtype_name)

    if in_environment:
        _run_queries_in_environment(
            model=model,
            tokenizer=tokenizer,
            queries=queries,
            max_new_tokens=max_new_tokens,
            max_prompt_length=max_prompt_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            max_env_steps=max_env_steps,
            db_seed=db_seed,
        )
        return

    for i, query in enumerate(queries, start=1):
        print("=" * 88)
        print(f"Query {i}: {query}")

        model_inputs = _build_inputs(tokenizer, query, model)
        prompt_len = model_inputs["input_ids"].shape[-1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            out = model.generate(**model_inputs, **gen_kwargs)

        new_tokens = out[0, prompt_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print("Response:")
        print(response if response else "<empty>")


def _generate_step_text(
    model,
    tokenizer,
    context_text: str,
    max_new_tokens: int,
    max_prompt_length: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> str:
    device = next(model.parameters()).device
    model_inputs = tokenizer(
        context_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    )
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    prompt_len = model_inputs["input_ids"].shape[-1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**model_inputs, **gen_kwargs)

    new_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _run_queries_in_environment(
    model,
    tokenizer,
    queries: list[str],
    max_new_tokens: int,
    max_prompt_length: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    max_env_steps: int | None,
    db_seed: int | None,
):
    oracle = Oracle()
    reward_model = RuleBasedRewardModel()

    for i, query in enumerate(queries, start=1):
        print("=" * 88)
        print(f"Environment Query {i}: {query}")

        today = date.today()
        this_seed = None if db_seed is None else db_seed + (i - 1)
        db = MockStudentDB(today=today, seed=this_seed)
        gt = oracle.derive(query, db, today)

        if max_env_steps is not None:
            gt.max_steps = min(gt.max_steps, max_env_steps)

        env = StudentAgentEnvironment(db, gt, query, today)

        print(f"Intent: {gt.intent}")
        print(f"Step budget: {gt.max_steps}")
        print(f"Subjects in this episode: {db.get_subjects()}")

        while not env.state.done:
            step_no = len(env.state.steps) + 1
            ctx_text = format_messages(env.state.context_window)
            step_text = _generate_step_text(
                model=model,
                tokenizer=tokenizer,
                context_text=ctx_text,
                max_new_tokens=max_new_tokens,
                max_prompt_length=max_prompt_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            obs = env.step(step_text)

            print(f"\n[Step {step_no}] model output:")
            print(step_text if step_text else "<empty>")
            print(f"[Step {step_no}] env info: {obs.get('info', {})}")

            if obs.get("done"):
                break

        trajectory = env.build_trajectory()
        score = reward_model.score(trajectory, gt)

        print("\nFinal response:")
        print(trajectory.get("final_response") or "<none>")
        print("Tool sequence:")
        print([c.get("tool") for c in trajectory.get("tool_calls", [])])
        print("Reward summary:")
        print(score)


def _read_queries_from_file(path: str) -> list[str]:
    lines = Path(path).read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def parse_args():
    parser = argparse.ArgumentParser(description="Run test queries against a checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/grpo/final")
    parser.add_argument("--query", action="append", help="Can be provided multiple times.")
    parser.add_argument("--queries_file", help="Optional text file with one query per line.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--in_environment", action="store_true",
                        help="Run full multi-step episodes in StudentAgentEnvironment.")
    parser.add_argument("--max_env_steps", type=int,
                        help="Optional cap for environment steps per query.")
    parser.add_argument("--db_seed", type=int,
                        help="Seed for MockStudentDB generation (incremented per query).")
    parser.add_argument("--device_map", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    queries = []
    if args.queries_file:
        queries.extend(_read_queries_from_file(args.queries_file))
    if args.query:
        queries.extend(args.query)
    if not queries:
        queries = DEFAULT_QUERIES

    run_queries(
        checkpoint=args.checkpoint,
        queries=queries,
        max_new_tokens=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        device_map=args.device_map,
        dtype_name=args.dtype,
        in_environment=args.in_environment,
        max_env_steps=args.max_env_steps,
        db_seed=args.db_seed,
    )