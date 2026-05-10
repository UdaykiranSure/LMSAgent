"""
src/dataset.py

Query template sampler with curriculum levels.
Self-contained — includes its own MockStudentDB and Oracle
so this file works standalone.
"""

import random
import json
from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# Mock Student Database
# ──────────────────────────────────────────────────────────────────────

class MockStudentDB:
    SUBJECTS = ["RL","SSL","DSA","DNN","NLP","CV","Deep Learning","Software Systems"]

    def __init__(self, today: date, seed: Optional[int] = None):
        self.today = today
        self._rng  = random.Random(seed)
        self._assignments = self._gen_assignments()
        self._grades      = self._gen_grades()
        self._schedule    = self._gen_schedule()
        self._notes       = self._gen_notes()
        self._todos: list = []

    def get_subjects(self):
        return self.SUBJECTS

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

    def get_allnotes(self, subject):
        return []  # local files always start empty

    def get_todos(self):
        return self._todos

    def get_announcements(self, subject):
        return [{"text": f"No new announcements for {subject}"}]

    def get_assignment_details(self, subject):
        assigns = self._assignments.get(subject, [])
        return [{**a,
                 "requirements": f"Complete all {subject} tasks",
                 "materials":    self._notes.get(subject, [])[:2]}
                for a in assigns]

    def add_todo(self, subject, title, deadline, priority="medium"):
        entry = {"subject": subject, "title": title,
                 "deadline": deadline, "priority": priority}
        self._todos.append(entry)
        return entry

    def add_notes(self, subject, file_name, content=""):
        entry = {"file_name": file_name, "content": content}
        self._notes.setdefault(subject, []).append(entry)
        return entry

    def _gen_assignments(self):
        out = {}
        for s in self.SUBJECTS:
            if self._rng.random() < 0.3:
                out[s] = []
            else:
                offset   = self._rng.randint(-3, 20)
                deadline = (self.today + timedelta(days=offset)).isoformat()
                out[s]   = [{"name": f"{s} Assignment",
                              "deadline": deadline,
                              "description": f"Complete {s} project"}]
        return out

    def _gen_grades(self):
        out = {}
        for s in self.SUBJECTS:
            n = self._rng.randint(0, 3)
            out[s] = [{"exam": f"Exam {i+1}",
                       "marks": self._rng.randint(40,100), "max": 100,
                       "date": (self.today - timedelta(days=30*(i+1))).isoformat()}
                      for i in range(n)]
        return out

    def _gen_schedule(self):
        days = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
        out  = {}
        for s in self.SUBJECTS:
            chosen = self._rng.sample(days, 2)
            out[s] = {d: {"time": f"{self._rng.randint(8,17):02d}:00",
                          "duration": 60,
                          "room": f"Room {self._rng.randint(100,400)}"}
                      for d in chosen}
        return out

    def _gen_notes(self):
        out = {}
        for s in self.SUBJECTS:
            n    = self._rng.randint(0, 4)
            out[s] = [{"file_name": f"{s}_lecture_{i+1}.pdf",
                       "uploaded": (self.today - timedelta(days=i*3)).isoformat()}
                      for i in range(n)]
        return out


# ──────────────────────────────────────────────────────────────────────
# Minimal Oracle  (inline — avoids circular imports)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class GroundTruth:
    correct_answer:         str
    answer_type:            str
    intent:                 str
    required_tools:         list
    required_tool_families: list
    optimal_steps:          int
    max_steps:              int
    exit_paths:             dict  = field(default_factory=dict)
    required_facts:         list  = field(default_factory=list)
    expected_writes:        list  = field(default_factory=list)
    oracle_data:            dict  = field(default_factory=dict)


def _parse_intent(query: str, db: MockStudentDB) -> tuple[str, Optional[str]]:
    """Returns (intent, subject_or_None)."""
    q = query.lower()
    # Subject matching
    subject = None
    for s in db.get_subjects():
        if s.lower() in q:
            subject = s
            break

    if "deadline" in q or "due date" in q:    return "deadline_lookup",    subject
    if "marks" in q or "grade" in q:          return "grades_lookup",      subject
    if "today" in q and "class" in q:         return "schedule_all_today", subject
    if "today" in q:                           return "schedule_specific",  subject
    if "tomorrow" in q:                        return "schedule_tomorrow",  subject
    if "this week" in q:                       return "due_this_week",      None
    if "study plan" in q and "mid" in q:       return "study_plan_all",     None
    if "study plan" in q or "study schedule" in q: return "study_plan_single", subject
    if "sync" in q or ("add" in q and "material" in q): return "material_sync", subject
    if "todo" in q or "remind" in q:           return "add_deadline_todo",  subject
    if "notes" in q or "material" in q:        return "notes_lookup",       subject
    if "announcement" in q:                    return "announcements_lookup", subject
    if "resource" in q or "requirement" in q:  return "resources_lookup",   subject
    if "syllabus" in q:                        return "resources_lookup",   subject
    return "notes_lookup", subject


class Oracle:
    BUFFER = 3

    def derive(self, query: str, db: MockStudentDB, today: date) -> GroundTruth:
        intent, subject = _parse_intent(query, db)
        subjects        = db.get_subjects()

        # Not enrolled
        if subject and subject not in subjects:
            return GroundTruth(
                correct_answer="NOT_ENROLLED", answer_type="NOT_ENROLLED",
                intent=intent,
                required_tools=["get_subjects"],
                required_tool_families=["lms"],
                optimal_steps=2, max_steps=5,
            )

        assignments = db.get_assignments(subject) if subject else {}

        # No assignments
        if subject and not assignments and intent in (
            "deadline_lookup","study_plan_single","resources_lookup","add_deadline_todo"
        ):
            return GroundTruth(
                correct_answer="NO_ASSIGNMENTS", answer_type="NO_ASSIGNMENTS",
                intent=intent,
                required_tools=["get_subjects","get_assignments"],
                required_tool_families=["lms"],
                optimal_steps=3, max_steps=6,
            )

        N = len(subjects)

        if intent == "deadline_lookup":
            deadline = assignments[0]["deadline"] if assignments else "NO_DATA"
            return GroundTruth(
                correct_answer=deadline, answer_type="exact",
                intent=intent,
                required_tools=["get_subjects","get_assignments","get_assignment_details"],
                required_tool_families=["lms"],
                optimal_steps=4, max_steps=7,
                required_facts=[deadline, assignments[0]["name"]] if assignments else [],
                oracle_data={"assignments": assignments},
            )

        if intent == "grades_lookup":
            grades = db.get_grades(subject) if subject else []
            if not grades:
                return GroundTruth(
                    correct_answer="NO_DATA", answer_type="NO_DATA",
                    intent=intent,
                    required_tools=["get_subjects","get_grades"],
                    required_tool_families=["lms"],
                    optimal_steps=3, max_steps=6,
                )
            last = grades[-1]
            ans  = f"{last['marks']}/{last['max']}"
            return GroundTruth(
                correct_answer=ans, answer_type="exact",
                intent=intent,
                required_tools=["get_subjects","get_grades"],
                required_tool_families=["lms"],
                optimal_steps=3, max_steps=6,
                required_facts=[ans],
                oracle_data={"grades": grades},
            )

        if intent in ("schedule_specific","schedule_today","schedule_tomorrow","schedule_all_today"):
            target  = today if "tomorrow" not in intent else today + timedelta(days=1)
            weekday = target.strftime("%A")
            if subject:
                sched = db.get_schedule(subject)
                cls   = sched.get(weekday)
                return GroundTruth(
                    correct_answer=cls["time"] if cls else "NO_CLASS",
                    answer_type="exact" if cls else "NO_CLASS",
                    intent=intent,
                    required_tools=["get_subjects","get_schedule"],
                    required_tool_families=["lms"],
                    optimal_steps=3, max_steps=6,
                    required_facts=[weekday, cls["time"]] if cls else [],
                )
            return GroundTruth(
                correct_answer="SYNTHESIS", answer_type="SYNTHESIS",
                intent=intent,
                required_tools=["get_subjects"] + ["get_schedule"]*N,
                required_tool_families=["lms"],
                optimal_steps=N+2, max_steps=N+5,
                required_facts=[weekday],
            )

        if intent == "due_this_week":
            due = {}
            for s in subjects:
                for a in db.get_assignments(s):
                    d = date.fromisoformat(a["deadline"])
                    if 0 <= (d - today).days <= 7:
                        due[s] = a["deadline"]
            return GroundTruth(
                correct_answer=json.dumps(due) if due else "NO_DATA",
                answer_type="set" if due else "NO_DATA",
                intent=intent,
                required_tools=["get_subjects"]+["get_assignments"]*N,
                required_tool_families=["lms"],
                optimal_steps=N+2, max_steps=N+5,
                required_facts=list(due.values()),
            )

        if intent == "study_plan_single" and subject:
            notes  = db.get_notes(subject)
            sched  = db.get_schedule(subject)
            facts  = (
                [a["deadline"] for a in assignments] +
                [n["file_name"] for n in notes[:3]] +
                list(sched.keys())
            )
            return GroundTruth(
                correct_answer="SYNTHESIS", answer_type="SYNTHESIS",
                intent=intent,
                required_tools=["get_subjects","get_assignments",
                                "get_assignment_details","get_notes","get_schedule"],
                required_tool_families=["lms"],
                optimal_steps=6, max_steps=9,
                required_facts=facts,
            )

        if intent == "material_sync" and subject:
            notes  = db.get_notes(subject)
            writes = [{"tool":"add_notes","subject":subject,"file_name":n["file_name"]}
                      for n in notes]
            return GroundTruth(
                correct_answer=f"synced {len(notes)} files",
                answer_type="state_change",
                intent=intent,
                required_tools=["get_subjects","get_notes","get_allnotes","add_notes"],
                required_tool_families=["lms","files"],
                optimal_steps=len(notes)+4, max_steps=len(notes)+7,
                required_facts=[n["file_name"] for n in notes],
                expected_writes=writes,
            )

        if intent == "add_deadline_todo" and subject and assignments:
            deadline = assignments[0]["deadline"]
            return GroundTruth(
                correct_answer=f"todo added",
                answer_type="state_change",
                intent=intent,
                required_tools=["get_subjects","get_assignments","get_todos","add_todo"],
                required_tool_families=["lms","planner"],
                optimal_steps=5, max_steps=8,
                required_facts=[deadline],
                expected_writes=[{"tool":"add_todo","subject":subject,"deadline":deadline}],
            )

        # Fallback
        return GroundTruth(
            correct_answer="NO_DATA", answer_type="NO_DATA",
            intent=intent,
            required_tools=["get_subjects"],
            required_tool_families=["lms"],
            optimal_steps=2, max_steps=5,
        )


# ──────────────────────────────────────────────────────────────────────
# Query templates
# ──────────────────────────────────────────────────────────────────────

@dataclass
class QueryTemplate:
    template:      str
    level:         int
    intent:        str
    needs_subject: bool


TEMPLATES = [
    # Level 1
    QueryTemplate("when is {subject} assignment deadline", 1, "deadline_lookup", True),
    QueryTemplate("what is the deadline for {subject} homework", 1, "deadline_lookup", True),
    QueryTemplate("what are my marks in {subject} last exam", 1, "grades_lookup", True),
    QueryTemplate("show me my {subject} grades", 1, "grades_lookup", True),
    QueryTemplate("get notes for {subject}", 1, "notes_lookup", True),
    QueryTemplate("what study materials are available for {subject}", 1, "notes_lookup", True),
    QueryTemplate("is there a {subject} class today", 1, "schedule_specific", True),
    QueryTemplate("when is {subject} class", 1, "schedule_specific", True),
    QueryTemplate("what are the requirements for {subject} assignment", 1, "resources_lookup", True),
    QueryTemplate("are there any announcements for {subject}", 1, "announcements_lookup", True),
    # Level 2
    QueryTemplate("are there any assignments due this week", 2, "due_this_week", False),
    QueryTemplate("what assignments do I have due this week", 2, "due_this_week", False),
    QueryTemplate("what are today's classes", 2, "schedule_all_today", False),
    QueryTemplate("what classes do I have today", 2, "schedule_all_today", False),
    QueryTemplate("what are tomorrow's classes", 2, "schedule_tomorrow", False),
    QueryTemplate("what is the syllabus for {subject} exam", 2, "resources_lookup", True),
    QueryTemplate("get me resources for {subject} exam", 2, "resources_lookup", True),
    QueryTemplate("show me all my pending assignments", 2, "due_this_week", False),
    # Level 3
    QueryTemplate("give me a study plan for {subject} exam", 3, "study_plan_single", True),
    QueryTemplate("create a study schedule for {subject}", 3, "study_plan_single", True),
    QueryTemplate("make me a study plan for mid examinations", 3, "study_plan_all", False),
    QueryTemplate("add the newly posted materials for {subject} to my files", 3, "material_sync", True),
    QueryTemplate("sync {subject} lecture notes to my local files", 3, "material_sync", True),
    QueryTemplate("add a todo for {subject} assignment deadline", 3, "add_deadline_todo", True),
    QueryTemplate("remind me about the {subject} deadline", 3, "add_deadline_todo", True),
]




LEVEL_WEIGHTS = {
    1: {1: 1.0, 2: 0.0, 3: 0.0},   # stage 1: L1 only — build fundamentals first
    2: {1: 0.7, 2: 0.3, 3: 0.0},   # stage 2: mostly L1, introduce some L2
    3: {1: 0.2, 2: 0.3, 3: 0.5},   # stage 3: full curriculum
}

# L2 templates that require sweeping all subjects (N+2 optimal steps) — these are
# much harder than the step budget allows for a small model early in training.
_SWEEP_INTENTS = {"due_this_week", "schedule_all_today", "schedule_tomorrow"}


class EpisodeDataset:
    def __init__(self, seed: int = 42, stage: int = 1):
        self._rng    = random.Random(seed)
        self._oracle = Oracle()
        self._stage  = stage
        # Pre-group templates by level for weighted sampling
        self._by_level = {
            lvl: [t for t in TEMPLATES if t.level == lvl]
            for lvl in (1, 2, 3)
        }
        levels = {1: [1], 2: [1, 2], 3: [1, 2, 3]}.get(stage, [1, 2])
        self.set_levels(levels)

    def set_levels(self, levels: list[int]):
        self._active_levels = levels
        self._templates = [t for t in TEMPLATES if t.level in levels]

    def sample(self, today: Optional[date] = None) -> tuple[str, MockStudentDB, GroundTruth]:
        today = today or date.today()
        db    = MockStudentDB(today, seed=self._rng.randint(0, 99999))

        # Pick a level according to curriculum weights, then a template within it
        weights_map = LEVEL_WEIGHTS.get(self._stage, {})
        active      = [l for l in self._active_levels if weights_map.get(l, 0) > 0]
        if active:
            level_weights = [weights_map[l] for l in active]
            chosen_level  = self._rng.choices(active, weights=level_weights, k=1)[0]
            pool = self._by_level[chosen_level]
            # For stage ≤ 2, skip all-subject sweep templates so the agent
            # isn't penalised for hitting step limits on tasks it can't yet do.
            if self._stage <= 2:
                pool = [t for t in pool if t.intent not in _SWEEP_INTENTS] or pool
        else:
            pool = self._templates

        tmpl = self._rng.choice(pool)
        if tmpl.needs_subject:
            subject = self._rng.choice(db.get_subjects())
            query   = tmpl.template.replace("{subject}", subject)
        else:
            query = tmpl.template
        gt = self._oracle.derive(query, db, today)
        return query, db, gt

    def sample_batch(self, n: int, today: Optional[date] = None) -> list:
        return [self.sample(today) for _ in range(n)]
