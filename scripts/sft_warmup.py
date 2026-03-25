# scripts/sft_warmup.py
"""
Stage 1 — Supervised Fine-Tuning warm-up.

Why this comes before PPO:
  Without SFT, the base model has no idea what a <tool_call> tag is,
  or that it should call get_subjects() first. RL exploration from a
  cold start wastes thousands of episodes learning basic formatting.
  ~500 expert demonstrations give the model a strong prior in ~1-2 hours.

What the demos look like:
  Each demo is a complete trajectory:
    [system prompt]
    User: "when is DSA assignment deadline"
    <tool_call>{"tool":"get_subjects","params":{}}</tool_call>
    <tool>[result]</tool>
    <tool_call>{"tool":"get_assignments","params":{"subject":"DSA"}}</tool_call>
    <tool>[result]</tool>
    <tool_call>{"tool":"get_assignment_details","params":{"subject":"DSA"}}</tool_call>
    <tool>[result]</tool>
    <respond>The DSA assignment "Sorting Algorithms HW" is due on 2025-09-22.</respond>

We generate these programmatically from mock DBs — no human labelling needed.
"""

import os
import yaml
import json
import argparse
import logging
from datetime import date, timedelta
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset

from src.mock_db import MockStudentDB
from src.oracle import Oracle
from src.dataset import QUERY_TEMPLATES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Expert trajectory generator
# Writes ideal tool-call chains based on oracle ground truth
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a student assistant agent. You have ZERO prior knowledge about the student.
You must call tools to discover all information before answering.

Always follow this discovery order:
  1. get_subjects()              — confirm subject exists
  2. get_assignments(subject)    — check assignments
  3. get_assignment_details(subject) — get full details if needed
  4. [other tools as needed]
  5. <respond>...</respond>

Emit tool calls as: <tool_call>{"tool": "...", "params": {...}}</tool_call>
Emit your final answer as: <respond>Your answer here</respond>"""


def generate_expert_trajectory(query: str, db: MockStudentDB, gt, today: date) -> str:
    """
    Build a complete expert trajectory string for one episode.
    This is what the model learns to imitate in SFT.
    """
    lines = []

    # System + user
    lines.append(f"<|system|>\n{SYSTEM_PROMPT}")
    lines.append(f"<|user|>\n{query}")
    lines.append("<|assistant|>")

    # Step 1: always get_subjects
    subjects = db.get_subjects()
    lines.append('<tool_call>{"tool": "get_subjects", "params": {}}</tool_call>')
    lines.append(f'<|tool|> [get_subjects]\n{json.dumps({"subjects": subjects})}')

    # Check if subject is enrolled
    from src.oracle import parse_intent_and_conditions, extract_subject
    ctx     = parse_intent_and_conditions(query, db)
    subject = ctx["subject"]
    intent  = ctx["intent"]

    if subject and subject not in subjects:
        lines.append(f'<respond>You are not enrolled in {subject}.</respond>')
        return "\n".join(lines)

    # Step 2: get_assignments (if intent needs it)
    needs_assignments = intent in (
        "deadline_lookup", "due_this_week", "study_plan_single",
        "study_plan_all", "resources_lookup", "add_deadline_todo",
        "add_all_todos"
    )

    assignments = []
    if needs_assignments:
        if subject:
            assignments = db.get_assignments(subject)
            params = json.dumps({"subject": subject})
            lines.append(f'<tool_call>{{"tool": "get_assignments", "params": {params}}}</tool_call>')
            lines.append(f'<|tool|> [get_assignments]\n{json.dumps({"assignments": assignments})}')

            if not assignments:
                lines.append(f'<respond>No assignments have been posted for {subject} yet.</respond>')
                return "\n".join(lines)
        else:
            # All subjects
            all_assigns = {}
            for s in subjects:
                a = db.get_assignments(s)
                params = json.dumps({"subject": s})
                lines.append(f'<tool_call>{{"tool": "get_assignments", "params": {params}}}</tool_call>')
                lines.append(f'<|tool|> [get_assignments]\n{json.dumps({"assignments": a})}')
                if a:
                    all_assigns[s] = a

    # Step 3: intent-specific steps
    if intent == "deadline_lookup" and assignments:
        params = json.dumps({"subject": subject})
        details = {
            "deadline":     assignments[0]["deadline"],
            "name":         assignments[0]["name"],
            "requirements": f"Complete all {subject} exercises",
        }
        lines.append(f'<tool_call>{{"tool": "get_assignment_details", "params": {params}}}</tool_call>')
        lines.append(f'<|tool|> [get_assignment_details]\n{json.dumps({"details": [details]})}')

        name     = assignments[0]["name"]
        deadline = assignments[0]["deadline"]
        lines.append(
            f'<respond>The {subject} assignment "{name}" is due on {deadline}.</respond>'
        )

    elif intent == "grades_lookup":
        grades = db.get_grades(subject)
        params = json.dumps({"subject": subject})
        lines.append(f'<tool_call>{{"tool": "get_grades", "params": {params}}}</tool_call>')
        lines.append(f'<|tool|> [get_grades]\n{json.dumps({"grades": grades})}')
        if grades:
            last = grades[-1]
            lines.append(
                f'<respond>Your last {subject} exam was "{last["exam"]}" '
                f'with a score of {last["marks"]}/{last["max"]}.</respond>'
            )
        else:
            lines.append(f'<respond>No exam grades have been recorded for {subject} yet.</respond>')

    elif intent == "due_this_week":
        due = {}
        for s, assigns in (all_assigns if not subject else {subject: assignments}).items():
            for a in assigns:
                d = date.fromisoformat(a["deadline"])
                if 0 <= (d - today).days <= 7:
                    due[s] = a["deadline"]
        if due:
            items = ", ".join(f"{s} ({d})" for s, d in due.items())
            lines.append(f'<respond>Assignments due this week: {items}.</respond>')
        else:
            lines.append('<respond>No assignments are due this week.</respond>')

    elif intent == "schedule_all_today":
        weekday = today.strftime("%A")
        today_classes = {}
        for s in subjects:
            sched = db.get_schedule(s)
            params = json.dumps({"subject": s})
            lines.append(f'<tool_call>{{"tool": "get_schedule", "params": {params}}}</tool_call>')
            lines.append(f'<|tool|> [get_schedule]\n{json.dumps({"schedule": sched})}')
            if weekday in sched:
                today_classes[s] = sched[weekday]["time"]
        if today_classes:
            items = ", ".join(f"{s} at {t}" for s, t in today_classes.items())
            lines.append(f'<respond>Today ({weekday}) you have: {items}.</respond>')
        else:
            lines.append(f'<respond>You have no classes today ({weekday}).</respond>')

    elif intent == "schedule_specific":
        sched  = db.get_schedule(subject)
        params = json.dumps({"subject": subject})
        lines.append(f'<tool_call>{{"tool": "get_schedule", "params": {params}}}</tool_call>')
        lines.append(f'<|tool|> [get_schedule]\n{json.dumps({"schedule": sched})}')
        weekday = today.strftime("%A")
        if weekday in sched:
            t = sched[weekday]["time"]
            lines.append(f'<respond>Yes, you have {subject} today ({weekday}) at {t}.</respond>')
        else:
            lines.append(f'<respond>No {subject} class today ({weekday}).</respond>')

    elif intent == "study_plan_single" and assignments:
        notes  = db.get_notes(subject)
        sched  = db.get_schedule(subject)

        params = json.dumps({"subject": subject})
        lines.append(f'<tool_call>{{"tool": "get_assignment_details", "params": {params}}}</tool_call>')
        lines.append(f'<|tool|> [get_assignment_details]\n{json.dumps({"details": assignments})}')
        lines.append(f'<tool_call>{{"tool": "get_notes", "params": {params}}}</tool_call>')
        lines.append(f'<|tool|> [get_notes]\n{json.dumps({"notes": notes})}')
        lines.append(f'<tool_call>{{"tool": "get_schedule", "params": {params}}}</tool_call>')
        lines.append(f'<|tool|> [get_schedule]\n{json.dumps({"schedule": sched})}')

        deadline  = assignments[0]["deadline"]
        file_list = ", ".join(n["file_name"] for n in notes[:3]) or "none posted"
        class_days= ", ".join(sched.keys()) or "no fixed schedule"
        lines.append(
            f'<respond>Study plan for {subject}:\n'
            f'- Assignment deadline: {deadline}\n'
            f'- Available materials: {file_list}\n'
            f'- Class days: {class_days}\n'
            f'Focus on reviewing materials before class days and complete '
            f'the assignment well before {deadline}.</respond>'
        )

    elif intent == "material_sync":
        notes  = db.get_notes(subject)
        params = json.dumps({"subject": subject})
        lines.append(f'<tool_call>{{"tool": "get_notes", "params": {params}}}</tool_call>')
        lines.append(f'<|tool|> [get_notes]\n{json.dumps({"notes": notes})}')
        lines.append(f'<tool_call>{{"tool": "get_allnotes", "params": {params}}}</tool_call>')
        lines.append(f'<|tool|> [get_allnotes]\n{json.dumps({"local_notes": []})}')

        for n in notes:
            add_params = json.dumps({"subject": subject, "file_name": n["file_name"]})
            lines.append(f'<tool_call>{{"tool": "add_notes", "params": {add_params}}}</tool_call>')
            lines.append(f'<|tool|> [add_notes]\n{json.dumps({"saved": n})}')

        count = len(notes)
        lines.append(f'<respond>Synced {count} file(s) for {subject} to your local files.</respond>')

    elif intent == "add_deadline_todo" and assignments:
        deadline = assignments[0]["deadline"]
        name     = assignments[0]["name"]
        lines.append('<tool_call>{"tool": "get_todos", "params": {}}</tool_call>')
        lines.append(f'<|tool|> [get_todos]\n{json.dumps({"todos": []})}')
        add_params = json.dumps({
            "subject":  subject,
            "title":    f"Complete {name}",
            "deadline": deadline,
            "priority": "high",
        })
        lines.append(f'<tool_call>{{"tool": "add_todo", "params": {add_params}}}</tool_call>')
        lines.append(f'<|tool|> [add_todo]\n{json.dumps({"created": {"subject": subject, "deadline": deadline}})}')
        lines.append(f'<respond>Added a reminder for {subject} assignment "{name}" due {deadline}.</respond>')

    else:
        # Generic fallback
        lines.append(f'<respond>I found information about {subject}. '
                     f'Please let me know what specific details you need.</respond>')

    return "\n".join(lines)


def build_sft_dataset(n_demos: int, config: dict, seed: int = 42) -> Dataset:
    """Generate n_demos expert trajectories across all levels."""
    random.seed(seed)
    today    = date.today()
    oracle   = Oracle()
    records  = []
    templates = QUERY_TEMPLATES  # all levels

    for i in range(n_demos):
        template  = random.choice(templates)
        db        = MockStudentDB(today + timedelta(days=random.randint(0, 30)))
        subject   = random.choice(db.get_subjects()) if template.needs_subject else None
        query     = template.template
        if subject:
            query = query.replace("{subject}", subject)

        gt   = oracle.derive(query, db, today)
        text = generate_expert_trajectory(query, db, gt, today)

        records.append({"text": text})
        if (i + 1) % 50 == 0:
            logger.info(f"Generated {i+1}/{n_demos} demos")

    return Dataset.from_list(records)


def run_sft(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]

    logger.info("Generating SFT demo dataset...")
    dataset = build_sft_dataset(n_demos=500, config=config)
    logger.info(f"Dataset: {len(dataset)} trajectories")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        torch_dtype = torch.bfloat16,
        device_map  = "auto",
    )

    lora_cfg = LoraConfig(**model_cfg["lora"])
    model    = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir              = "checkpoints/sft",
        num_train_epochs        = 3,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,
        learning_rate           = 2e-4,
        lr_scheduler_type       = "cosine",
        warmup_ratio            = 0.1,
        logging_steps           = 10,
        save_strategy           = "epoch",
        bf16                    = True,
        dataloader_num_workers  = 0,
        report_to               = config["training"].get("log_with", "none"),
    )

    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = dataset,
        args            = training_args,
        max_seq_length  = 4096,
        dataset_text_field = "text",
    )

    logger.info("Starting SFT training...")
    trainer.train()
    trainer.save_model("checkpoints/sft")
    tokenizer.save_pretrained("checkpoints/sft")
    logger.info("SFT complete → checkpoints/sft")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo_config.yaml")
    args = parser.parse_args()
    run_sft(args.config)
