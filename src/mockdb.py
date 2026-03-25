from dataclasses import dataclass, field
from typing import Optional
from datetime import date, timedelta
import random, json

# ─────────────────────────────────────────────
# Mock database — this IS the ground truth source
# ─────────────────────────────────────────────

class MockStudentDB:
    """
    Randomly generated per episode. 
    Calling methods on this is how the oracle derives ground truth.
    The agent also calls these same methods via the tool API.
    """
    def __init__(self, today: date, n_subjects: int = 5):
        self.today = today
        all_subjects = ["RL", "SSL", "Algorithms", "DNN", "NLP", "CV",
                         "Deep Learning", "Software Systems"]
        self.subjects = random.sample(all_subjects, min(n_subjects, len(all_subjects)))
        self._assignments = self._gen_assignments()
        self._grades      = self._gen_grades()
        self._schedule    = self._gen_schedule()
        self._notes       = self._gen_notes()
        self._todos       = []

    def get_subjects(self):
        return self.subjects

    def get_assignments(self, subject=None):
        if subject:
            return self._assignments.get(subject, [])
        return self._assignments

    def get_grades(self, subject):
        return self._grades.get(subject, [])

    def get_schedule(self, subject=None):
        if subject:
            return self._schedule.get(subject, {})
        return self._schedule

    def get_notes(self, subject):
        return self._notes.get(subject, [])

    def get_todos(self):
        return self._todos

    def add_todo(self, subject, title, deadline, priority="medium"):
        entry = {"subject": subject, "title": title,
                 "deadline": deadline, "priority": priority}
        self._todos.append(entry)
        return entry

    def add_notes(self, subject, file_name, content):
        if subject not in self._notes:
            self._notes[subject] = []
        entry = {"file_name": file_name, "content": content}
        self._notes[subject].append(entry)
        return entry

    def _gen_assignments(self):
        assignments = {}
        for subj in self.subjects:
            # ~30% chance of no assignment
            if random.random() < 0.3:
                assignments[subj] = []
            else:
                offset = random.randint(-3, 20)  # -3 = overdue
                deadline = (self.today + timedelta(days=offset)).isoformat()
                assignments[subj] = [{
                    "name": f"{subj} Assignment",
                    "deadline": deadline,
                    "description": f"Complete {subj} project"
                }]
        return assignments

    def _gen_grades(self):
        grades = {}
        for subj in self.subjects:
            n_exams = random.randint(0, 3)
            grades[subj] = [
                {"exam": f"Exam {i+1}", "marks": random.randint(40, 100),
                 "max": 100, "date": (self.today - timedelta(days=30*(i+1))).isoformat()}
                for i in range(n_exams)
            ]
        return grades

    def _gen_schedule(self):
        days = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
        schedule = {}
        for subj in self.subjects:
            # Each subject has 2 classes/week on random days
            class_days = random.sample(days, 2)
            schedule[subj] = {
                day: {"time": f"{random.randint(8,17):02d}:00",
                      "duration": 60, "room": f"Room {random.randint(100,400)}"}
                for day in class_days
            }
        return schedule

    def _gen_notes(self):
        notes = {}
        for subj in self.subjects:
            n_files = random.randint(0, 4)
            notes[subj] = [
                {"file_name": f"{subj}_lecture_{i+1}.pdf",
                 "uploaded": (self.today - timedelta(days=i*3)).isoformat(),
                 "size_kb": random.randint(100, 2000)}
                for i in range(n_files)
            ]
        return notes


# ─────────────────────────────────────────────
# Intent parser — maps query → intent type + conditions
# ─────────────────────────────────────────────

INTENT_MAP = {
    "deadline_lookup":    ["get_assignments"],
    "grades_lookup":      ["get_grades"],
    "schedule_today":     ["get_schedule"],
    "schedule_tomorrow":  ["get_schedule"],
    "due_this_week":      ["get_subjects", "get_assignments"],
    "notes_lookup":       ["get_notes"],
    "announcements":      ["get_announcements"],
    "assignment_details": ["get_assignment_details"],
    "study_plan_single":  ["get_notes", "get_schedule", "get_assignment_details"],
    "study_plan_all":     ["get_subjects", "get_notes", "get_schedule", "get_assignments"],
    "material_sync":      ["get_notes", "get_allnotes", "add_notes"],
    "add_deadline_todo":  ["get_assignments", "get_todos", "add_todo"],
    "resources_lookup":   ["get_assignment_details", "get_notes"],
}

# Subjects that are genuinely ambiguous short-forms in this domain
AMBIGUOUS_SUBJECTS = {"DL", "ML", "AI", "NN", "CV"}

def parse_intent_and_conditions(query: str, db: MockStudentDB) -> dict:
    """
    Deterministically parse query into intent + runtime conditions.
    Returns the intent string and any modifier flags.
    """
    q = query.lower()
    subject = extract_subject(query, db)

    if "deadline" in q or "due date" in q:
        intent = "deadline_lookup"
    elif "marks" in q or "grade" in q or "exam" in q and "syllabus" not in q:
        intent = "grades_lookup"
    elif "today" in q and ("class" in q or "schedule" in q):
        intent = "schedule_today"
    elif "tomorrow" in q:
        intent = "schedule_tomorrow"
    elif "this week" in q and ("assignment" in q or "due" in q):
        intent = "due_this_week"
    elif "notes" in q or "material" in q and "add" not in q:
        intent = "notes_lookup"
    elif "study plan" in q and subject:
        intent = "study_plan_single"
    elif "study plan" in q and "mid" in q:
        intent = "study_plan_all"
    elif "add" in q and ("material" in q or "file" in q):
        intent = "material_sync"
    elif "add" in q and "todo" in q:
        intent = "add_deadline_todo"
    elif "resource" in q or "requirement" in q:
        intent = "resources_lookup"
    elif "syllabus" in q:
        intent = "assignment_details"
    else:
        intent = "notes_lookup"

    # Detect scope: named subject, all subjects, or ambiguous
    scope = "named" if subject else "all"
    ambiguous = subject and subject.upper() in AMBIGUOUS_SUBJECTS

    # Detect if subject is enrolled
    enrolled = subject in db.get_subjects() if subject else True

    return {
        "intent": intent,
        "subject": subject,
        "scope": scope,
        "ambiguous": ambiguous,
        "enrolled": enrolled,
    }

def extract_subject(query: str, db: MockStudentDB) -> Optional[str]:
    for subj in db.get_subjects():
        if subj.lower() in query.lower():
            return subj
    # Check abbreviations
    abbr_map = {"rl": "RL", "ssl": "SSL", "dnn": "DNN", "nlp": "NLP",
                "cv": "CV", "dl": "Deep Learning", "ss": "Software Systems"}
    for abbr, full in abbr_map.items():
        if abbr in query.lower().split():
            return full
    return None
