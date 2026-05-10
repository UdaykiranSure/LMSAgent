#!/usr/bin/env python3
"""
Run a user query through StudentAgentEnvironment using a locally served vLLM model.

Example:
python scripts/vllm_env_chat.py \
  --query "What are the requirements for RL assignment?" \
  --model "gpt-4o-mini" \
  --base-url "http://localhost:8000"
"""

import argparse
import json
import random
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

# Allow direct execution via `python scripts/vllm_env_chat.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environment import StudentAgentEnvironment
from src.mockdb import MockStudentDB
from src.oracle import Oracle


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} at {url}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach vLLM server at {url}: {e}") from e


def get_served_models(base_url: str) -> list[str]:
    obj = fetch_json(f"{base_url.rstrip('/')}/v1/models")
    data = obj.get("data", [])
    model_ids = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
    return model_ids


def resolve_model_name(base_url: str, requested_model: str) -> str:
    served = get_served_models(base_url)
    if not served:
        raise RuntimeError("No served models found at /v1/models")

    if requested_model in served:
        return requested_model

    # Case-insensitive exact match
    lowered_map = {m.lower(): m for m in served}
    if requested_model.lower() in lowered_map:
        matched = lowered_map[requested_model.lower()]
        print(f"Requested model '{requested_model}' not found exactly; using '{matched}'")
        return matched

    # Fallback to first served model to keep the script usable
    fallback = served[0]
    print(f"Requested model '{requested_model}' is not served.")
    print(f"Available models: {', '.join(served)}")
    print(f"Using fallback model: {fallback}")
    return fallback


def to_chat_messages(context_window: list[dict]) -> list[dict]:
    """
    Convert environment context into OpenAI-compatible chat messages.

    vLLM chat endpoint may not accept role='tool' in this project setup,
    so tool outputs are represented as assistant messages with a prefix.
    """
    messages = []
    for m in context_window:
        role = m.get("role", "assistant")
        content = m.get("content", "")

        if role in ("system", "user", "assistant"):
            messages.append({"role": role, "content": content})
        elif role == "tool":
            name = m.get("name", "tool")
            messages.append({"role": "assistant", "content": f"[{name}] {content}"})
        else:
            messages.append({"role": "assistant", "content": content})

    return messages


def call_vllm_chat(base_url: str, model: str, messages: list[dict], temperature: float) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach vLLM server: {e}") from e

    obj = json.loads(body)
    try:
        return obj["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected response format: {obj}") from e


def run_query(query: str, model: str, base_url: str, seed: int, temperature: float):
    random.seed(seed)
    today = date.today()

    model = resolve_model_name(base_url, model)

    db = MockStudentDB(today=today)
    gt = Oracle().derive(query, db, today)
    env = StudentAgentEnvironment(db=db, gt=gt, query=query, today=today)

    print(f"Query: {query}")
    print(f"Model: {model}")
    print(f"Max steps: {gt.max_steps}")
    print("-" * 72)

    while not env.state.done:
        messages = to_chat_messages(env.state.context_window)
        assistant_text = call_vllm_chat(
            base_url=base_url,
            model=model,
            messages=messages,
            temperature=temperature,
        )
        print(f"Assistant: {assistant_text}")

        obs = env.step(assistant_text)
        reason = obs.get("info", {}).get("reason", "ongoing")
        print(f"Env: done={obs.get('done')} reason={reason}")
        print("-" * 72)

    print("Final response:")
    print(env.state.final_response or "<no final response>")
    print("\nTrajectory summary:")
    print(json.dumps(env.build_trajectory(), indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description="Run environment loop with local vLLM chat endpoint")
    parser.add_argument("--query", required=True, help="User query to answer")
    parser.add_argument("--model", default="gpt-5.2", help="Model name served by vLLM")
    parser.add_argument("--base-url", default="http://localhost:8000", help="vLLM server base URL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for mock DB")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--list-models", action="store_true", help="Print served models and exit")
    args = parser.parse_args()

    if args.list_models:
        models = get_served_models(args.base_url)
        if not models:
            print("No models found")
            return
        print("Served models:")
        for m in models:
            print(f"- {m}")
        return

    run_query(
        query=args.query,
        model=args.model,
        base_url=args.base_url,
        seed=args.seed,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
