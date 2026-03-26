"""
scripts/gen_demos.py

Generates expert trajectory strings for SFT warm-up.
Each demo is a complete episode in Qwen2.5 chat format:
  system prompt → user query → tool calls → final answer

Called by train.py --stage sft
Can also be run standalone: python scripts/gen_demos.py --n 500
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import random
import argparse
from datetime import date, timedelta

from src.dataset import MockStudentDB, Oracle, TEMPLATES, _parse_intent


SYSTEM_PROMPT = """You are a student assistant. You have zero knowledge about the student.
Call get_subjects() first to verify enrollment, then the appropriate tools.
Use <tool_call>{"tool":"...","params":{...}}</tool_call> for tool calls.
Use <respond>answer</respond> for your final answer."""


def make_trajectory(query: str, db: MockStudentDB, gt, today: date) -> str:
    intent, subject = _parse_intent(query, db)
    subjects        = db.get_subjects()
    lines           = []

    lines.append(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>")
    lines.append(f"<|im_start|>user\n{query}<|im_end|>")
    lines.append("<|im_start|>assistant")

    def tool_call(name, params=None):
        p = params or {}
        lines.append(f'<tool_call>{{"tool":"{name}","params":{json.dumps(p)}}}</tool_call>')

    def tool_result(name, result):
        lines.append(f"<|im_end|>\n<|im_start|>tool [{name}]")
        lines.append(json.dumps(result))
        lines.append("<|im_end|>\n<|im_start|>assistant")

    def respond(text):
        lines.append(f"<respond>{text}</respond>")
        lines.append("<|im_end|>")

    # Step 1: always get_subjects
    tool_call("get_subjects")
    tool_result("get_subjects", {"subjects": subjects})

    if subject and subject not in subjects:
        respond(f"You are not enrolled in {subject}.")
        return "\n".join(lines)

    assigns = db.get_assignments(subject) if subject else {}

    # Step 2: get_assignments (for intents that need it)
    needs_assigns = intent in (
        "deadline_lookup","study_plan_single","resources_lookup",
        "add_deadline_todo","due_this_week"
    )

    if needs_assigns:
        if subject:
            tool_call("get_assignments", {"subject": subject})
            tool_result("get_assignments", {"assignments": assigns})
            if not assigns:
                respond(f"No assignments have been posted for {subject} yet.")
                return "\n".join(lines)
        else:
            all_a = {}
            for s in subjects:
                a = db.get_assignments(s)
                tool_call("get_assignments", {"subject": s})
                tool_result("get_assignments", {"assignments": a})
                if a: all_a[s] = a

    # Step 3+: intent-specific
    if intent == "deadline_lookup" and assigns:
        tool_call("get_assignment_details", {"subject": subject})
        tool_result("get_assignment_details", {
            "details": db.get_assignment_details(subject)
        })
        name     = assigns[0]["name"]
        deadline = assigns[0]["deadline"]
        respond(f'The {subject} assignment "{name}" is due on {deadline}.')

    elif intent == "grades_lookup":
        tool_call("get_grades", {"subject": subject})
        grades = db.get_grades(subject)
        tool_result("get_grades", {"grades": grades})
        if grades:
            last = grades[-1]
            respond(f'Your last {subject} exam was "{last["exam"]}" '
                    f'with a score of {last["marks"]}/{last["max"]}.')
        else:
            respond(f"No exam grades have been recorded for {subject} yet.")

    elif intent == "schedule_specific":
        tool_call("get_schedule", {"subject": subject})
        sched = db.get_schedule(subject)
        tool_result("get_schedule", {"schedule": sched})
        weekday = today.strftime("%A")
        if weekday in sched:
            t = sched[weekday]["time"]
            respond(f"Yes, you have {subject} today ({weekday}) at {t}.")
        else:
            respond(f"No {subject} class today ({weekday}).")

    elif intent == "schedule_all_today":
        weekday   = today.strftime("%A")
        today_cls = {}
        for s in subjects:
            tool_call("get_schedule", {"subject": s})
            sched = db.get_schedule(s)
            tool_result("get_schedule", {"schedule": sched})
            if weekday in sched:
                today_cls[s] = sched[weekday]["time"]
        if today_cls:
            items = ", ".join(f"{s} at {t}" for s, t in today_cls.items())
            respond(f"Today ({weekday}) you have: {items}.")
        else:
            respond(f"You have no classes today ({weekday}).")

    elif intent == "due_this_week":
        from datetime import date as ddate
        due = {}
        for s, a_list in (all_a if not subject else {subject: assigns}).items():
            for a in a_list:
                d = ddate.fromisoformat(a["deadline"])
                if 0 <= (d - today).days <= 7:
                    due[s] = a["deadline"]
        if due:
            items = ", ".join(f"{s} (due {d})" for s, d in due.items())
            respond(f"Assignments due this week: {items}.")
        else:
            respond("No assignments are due this week.")

    elif intent == "notes_lookup":
        tool_call("get_notes", {"subject": subject})
        notes = db.get_notes(subject)
        tool_result("get_notes", {"notes": notes})
        if notes:
            files = ", ".join(n["file_name"] for n in notes)
            respond(f"Available materials for {subject}: {files}.")
        else:
            respond(f"No materials have been posted for {subject} yet.")

    elif intent == "study_plan_single" and assigns:
        tool_call("get_assignment_details", {"subject": subject})
        tool_result("get_assignment_details", {"details": db.get_assignment_details(subject)})
        tool_call("get_notes", {"subject": subject})
        notes = db.get_notes(subject)
        tool_result("get_notes", {"notes": notes})
        tool_call("get_schedule", {"subject": subject})
        sched = db.get_schedule(subject)
        tool_result("get_schedule", {"schedule": sched})
        deadline  = assigns[0]["deadline"]
        file_list = ", ".join(n["file_name"] for n in notes[:3]) or "none"
        days      = ", ".join(sched.keys()) or "no fixed schedule"
        respond(
            f"Study plan for {subject}: Assignment due {deadline}. "
            f"Review materials ({file_list}). Classes on {days}."
        )

    elif intent == "material_sync":
        tool_call("get_notes", {"subject": subject})
        notes = db.get_notes(subject)
        tool_result("get_notes", {"notes": notes})
        tool_call("get_allnotes", {"subject": subject})
        tool_result("get_allnotes", {"local_notes": []})
        for n in notes:
            tool_call("add_notes", {"subject": subject, "file_name": n["file_name"]})
            tool_result("add_notes", {"saved": n})
        respond(f"Synced {len(notes)} file(s) for {subject} to your local files.")

    elif intent == "add_deadline_todo" and assigns:
        tool_call("get_todos")
        tool_result("get_todos", {"todos": []})
        deadline = assigns[0]["deadline"]
        name     = assigns[0]["name"]
        tool_call("add_todo", {
            "subject": subject, "title": f"Complete {name}",
            "deadline": deadline, "priority": "high"
        })
        tool_result("add_todo", {"created": {"subject": subject, "deadline": deadline}})
        respond(f'Added reminder for {subject} assignment "{name}" due {deadline}.')

    else:
        respond(f"I found information about {subject}. Let me know what specific details you need.")

    return "\n".join(lines)


def generate_demos(n: int = 500, seed: int = 42) -> list[str]:
    rng   = random.Random(seed)
    today = date.today()
    demos = []
    for i in range(n):
        tmpl    = rng.choice(TEMPLATES)
        day_off = timedelta(days=rng.randint(0, 30))
        db      = MockStudentDB(today + day_off, seed=rng.randint(0, 99999))
        oracle  = Oracle()
        subject = rng.choice(db.get_subjects()) if tmpl.needs_subject else None
        query   = tmpl.template
        if subject:
            query = query.replace("{subject}", subject)
        gt   = oracle.derive(query, db, today)
        text = make_trajectory(query, db, gt, today)
        demos.append(text)
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n} demos")
    return demos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=500)
    parser.add_argument("--out",  default="data/sft_demos.jsonl")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    demos = generate_demos(args.n)
    with open(args.out, "w") as f:
        for d in demos:
            f.write(json.dumps({"text": d}) + "\n")
    print(f"Saved {len(demos)} demos → {args.out}")
