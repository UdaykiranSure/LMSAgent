from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional
import json

# ─────────────────────────────────────────────────────────────────
# GroundTruth — updated dataclass
# New fields: intent, exit_path, step_budget_by_path
# ─────────────────────────────────────────────────────────────────

@dataclass
class GroundTruth:
    # What the final response must contain
    correct_answer: str
    answer_type: str          # "exact" | "set" | "NO_DATA" | "NOT_ENROLLED"
                              # "NO_ASSIGNMENTS" | "NO_CLASS" | "SYNTHESIS"
                              # "state_change"

    # Intent stored directly — reward model reads this, no re-parsing
    intent: str

    # The FULL ordered chain — position matters for step-order reward
    # e.g. ["get_subjects", "get_assignments", "get_assignment_details"]
    required_tools: list[str]

    # Which tool families must be touched
    required_tool_families: list[str]

    # Dynamic budget — set from actual DB state, not hardcoded
    optimal_steps: int         # minimum steps for the happy path
    max_steps: int             # ceiling before penalty (optimal + buffer)

    # Early exit metadata
    # Maps exit_condition → (optimal_steps, required_tools_for_that_path)
    # e.g. {"NOT_ENROLLED": (2, ["get_subjects"]),
    #        "NO_ASSIGNMENTS": (3, ["get_subjects","get_assignments"])}
    exit_paths: dict = field(default_factory=dict)

    # Facts that must appear in the response (used for scoring)
    required_facts: list[str] = field(default_factory=list)

    # For write operations (add_todo, add_notes)
    expected_writes: list[dict] = field(default_factory=list)

    # The actual DB values the oracle computed — reward model uses these
    # to verify correctness without re-querying the DB
    oracle_data: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────
# Step-order constants — defines the canonical discovery chain
# Every intent type maps to an ordered list of (tool, condition)
# ─────────────────────────────────────────────────────────────────

# ALWAYS_REQUIRED: these must appear in the trajectory for every intent
UNIVERSAL_FIRST_STEP = "get_subjects"

# Canonical tool chains per intent — ORDER MATTERS
# The reward model checks that tools appear in this order
TOOL_CHAINS = {

    # ── Simple lookups ─────────────────────────────────────────────
    "deadline_lookup": [
        "get_subjects",           # step 1: discover + verify subject
        "get_assignments",        # step 2: get assignment list + deadlines
        "get_assignment_details", # step 3: get full details (requirements, materials)
    ],

    "grades_lookup": [
        "get_subjects",
        "get_grades",
    ],

    "notes_lookup": [
        "get_subjects",
        "get_notes",
    ],

    "announcements_lookup": [
        "get_subjects",
        "get_announcements",
    ],

    # ── Schedule queries ───────────────────────────────────────────
    "schedule_specific": [        # "is there a DNN class today"
        "get_subjects",
        "get_schedule",
    ],

    "schedule_all_today": [       # "what are today's classes"
        "get_subjects",
        # get_schedule called N times (one per subject) — handled dynamically
        "get_schedule",           # placeholder; reward checks N calls
    ],

    "schedule_tomorrow": [
        "get_subjects",
        "get_schedule",
    ],

    # ── Multi-subject queries ──────────────────────────────────────
    "due_this_week": [
        "get_subjects",
        "get_assignments",        # called N times — one per subject
    ],

    # ── Synthesis ─────────────────────────────────────────────────
    "study_plan_single": [
        "get_subjects",
        "get_assignments",
        "get_assignment_details",
        "get_notes",
        "get_schedule",
    ],

    "study_plan_all": [
        "get_subjects",
        # then for each subject: get_assignments, get_notes, get_schedule
        "get_assignments",
        "get_notes",
        "get_schedule",
    ],

    "resources_lookup": [
        "get_subjects",
        "get_assignments",
        "get_assignment_details",
        "get_notes",
    ],

    # ── Write operations ───────────────────────────────────────────
    "material_sync": [
        "get_subjects",
        "get_notes",        # LMS: what exists
        "get_allnotes",     # Files: what is already saved locally
        "add_notes",        # write each new file (called M times)
    ],

    "add_deadline_todo": [
        "get_subjects",
        "get_assignments",
        "get_todos",        # check for duplicates first
        "add_todo",
    ],

    "add_all_todos": [
        "get_subjects",
        "get_assignments",  # N times
        "get_todos",
        "add_todo",         # M times
    ],
}

# Early exit paths — what the reward considers CORRECT at each step
# If the agent exits at this step for the right reason, it gets full reward
EXIT_CONDITIONS = {
    "NOT_ENROLLED": {
        "after_step": "get_subjects",
        "optimal_steps": 2,         # get_subjects + RESPOND
        "required_tools": ["get_subjects"],
    },
    "NO_ASSIGNMENTS": {
        "after_step": "get_assignments",
        "optimal_steps": 3,         # get_subjects + get_assignments + RESPOND
        "required_tools": ["get_subjects", "get_assignments"],
    },
    "NO_CLASS": {
        "after_step": "get_schedule",
        "optimal_steps": 3,
        "required_tools": ["get_subjects", "get_schedule"],
    },
    "NO_DATA": {
        "after_step": "get_notes",
        "optimal_steps": 3,
        "required_tools": ["get_subjects", "get_notes"],
    },
    "NO_NEW_FILES": {
        "after_step": "get_allnotes",
        "optimal_steps": 4,         # get_subjects + get_notes + get_allnotes + RESPOND
        "required_tools": ["get_subjects", "get_notes", "get_allnotes"],
    },
}


# ─────────────────────────────────────────────────────────────────
# Oracle — fully rewritten for zero-knowledge multi-step
# ─────────────────────────────────────────────────────────────────

class Oracle:
    """
    Derives GroundTruth by running the same queries the agent will run,
    against the mock DB, before the episode starts.

    Key design:
    - Simulates all 3 discovery steps to know which exit path applies
    - Sets optimal_steps and max_steps dynamically from DB state
    - Stores oracle_data so reward model never needs to re-query DB
    - Encodes all valid exit paths so reward model can score early exits correctly
    """

    BUFFER = 3   # steps added above optimal before penalty kicks in

    def derive(self, query: str, db, today: date) -> GroundTruth:
        ctx     = parse_intent_and_conditions(query, db)
        intent  = ctx["intent"]
        subject = ctx["subject"]       # may be None for "all" queries
        N       = len(db.get_subjects())  # total enrolled subjects

        # ── Simulate step 1: get_subjects ──────────────────────────────
        all_subjects = db.get_subjects()   # oracle runs this just like agent will

        # ── NOT_ENROLLED exit path ─────────────────────────────────────
        if subject and subject not in all_subjects:
            return GroundTruth(
                correct_answer        = "NOT_ENROLLED",
                answer_type           = "NOT_ENROLLED",
                intent                = intent,
                required_tools        = EXIT_CONDITIONS["NOT_ENROLLED"]["required_tools"],
                required_tool_families= ["lms"],
                optimal_steps         = EXIT_CONDITIONS["NOT_ENROLLED"]["optimal_steps"],
                max_steps             = EXIT_CONDITIONS["NOT_ENROLLED"]["optimal_steps"] + self.BUFFER,
                exit_paths            = {"NOT_ENROLLED": EXIT_CONDITIONS["NOT_ENROLLED"]},
                required_facts        = [],
                oracle_data           = {"all_subjects": all_subjects},
            )

        # ── Simulate step 2: get_assignments ──────────────────────────
        if subject:
            assignments = db.get_assignments(subject)
        else:
            assignments = {s: db.get_assignments(s) for s in all_subjects}

        # ── NO_ASSIGNMENTS exit path ───────────────────────────────────
        if subject and not assignments and intent in (
            "deadline_lookup", "study_plan_single",
            "resources_lookup", "add_deadline_todo"
        ):
            return GroundTruth(
                correct_answer        = "NO_ASSIGNMENTS",
                answer_type           = "NO_ASSIGNMENTS",
                intent                = intent,
                required_tools        = EXIT_CONDITIONS["NO_ASSIGNMENTS"]["required_tools"],
                required_tool_families= ["lms"],
                optimal_steps         = EXIT_CONDITIONS["NO_ASSIGNMENTS"]["optimal_steps"],
                max_steps             = EXIT_CONDITIONS["NO_ASSIGNMENTS"]["optimal_steps"] + self.BUFFER,
                exit_paths            = {
                    "NOT_ENROLLED":   EXIT_CONDITIONS["NOT_ENROLLED"],
                    "NO_ASSIGNMENTS": EXIT_CONDITIONS["NO_ASSIGNMENTS"],
                },
                required_facts        = [],
                oracle_data           = {
                    "all_subjects": all_subjects,
                    "assignments":  assignments,
                },
            )

        # ── Simulate step 3: get_assignment_details (if needed) ────────
        details = {}
        if intent in ("deadline_lookup", "study_plan_single",
                      "resources_lookup", "add_deadline_todo"):
            if assignments:
                details = self._build_details(subject, assignments, db)

        # ── Now branch by intent ───────────────────────────────────────

        if intent == "deadline_lookup":
            return self._deadline_lookup(
                subject, assignments, details, all_subjects, today
            )

        if intent == "grades_lookup":
            return self._grades_lookup(subject, db, all_subjects)

        if intent == "notes_lookup":
            return self._notes_lookup(subject, db, all_subjects)

        if intent in ("schedule_specific", "schedule_tomorrow"):
            return self._schedule_specific(subject, db, all_subjects, today, intent)

        if intent == "schedule_all_today":
            return self._schedule_all_today(db, all_subjects, today, N)

        if intent == "due_this_week":
            return self._due_this_week(db, all_subjects, today, N)

        if intent == "study_plan_single":
            return self._study_plan_single(
                subject, assignments, details, db, all_subjects
            )

        if intent == "study_plan_all":
            return self._study_plan_all(db, all_subjects, N)

        if intent == "resources_lookup":
            return self._resources_lookup(
                subject, assignments, details, db, all_subjects
            )

        if intent == "material_sync":
            return self._material_sync(subject, db, all_subjects)

        if intent == "add_deadline_todo":
            return self._add_deadline_todo(
                subject, assignments, db, all_subjects
            )

        if intent == "announcements_lookup":
            return self._announcements_lookup(subject, db, all_subjects)

        # Fallback
        return self._fallback(intent, all_subjects)

    # ─────────────────────────────────────────────────────────────
    # Per-intent derivation methods
    # Each one simulates ALL steps the agent will take, then seals
    # the ground truth from the actual DB values
    # ─────────────────────────────────────────────────────────────

    def _deadline_lookup(self, subject, assignments, details,
                         all_subjects, today) -> GroundTruth:
        # Simulate step 2 result
        first = assignments[0]
        deadline = first["deadline"]
        name     = first["name"]

        # Simulate step 3 result — details add requirements, materials
        requirements = details.get("requirements", "")
        materials    = details.get("materials", [])

        # required_facts: every value that MUST appear in the response
        required_facts = [deadline, name]
        if requirements:
            required_facts.append(requirements[:30])  # prefix match ok

        # All valid exit paths this episode could legally take
        exit_paths = {
            "NOT_ENROLLED":   EXIT_CONDITIONS["NOT_ENROLLED"],
            "NO_ASSIGNMENTS": EXIT_CONDITIONS["NO_ASSIGNMENTS"],
        }

        chain         = TOOL_CHAINS["deadline_lookup"]
        optimal_steps = len(chain) + 1   # 3 tool calls + RESPOND = 4

        return GroundTruth(
            correct_answer        = deadline,
            answer_type           = "exact",
            intent                = "deadline_lookup",
            required_tools        = chain,
            required_tool_families= ["lms"],
            optimal_steps         = optimal_steps,
            max_steps             = optimal_steps + self.BUFFER,
            exit_paths            = exit_paths,
            required_facts        = required_facts,
            oracle_data           = {
                "all_subjects":   all_subjects,
                "assignments":    assignments,
                "details":        details,
                "deadline":       deadline,
                "assignment_name":name,
            },
        )

    def _grades_lookup(self, subject, db, all_subjects) -> GroundTruth:
        grades = db.get_grades(subject)

        if not grades:
            return GroundTruth(
                correct_answer        = "NO_DATA",
                answer_type           = "NO_DATA",
                intent                = "grades_lookup",
                required_tools        = ["get_subjects", "get_grades"],
                required_tool_families= ["lms"],
                optimal_steps         = 3,
                max_steps             = 3 + self.BUFFER,
                exit_paths            = {
                    "NOT_ENROLLED": EXIT_CONDITIONS["NOT_ENROLLED"],
                    "NO_DATA":      EXIT_CONDITIONS["NO_DATA"],
                },
                required_facts        = [],
                oracle_data           = {"all_subjects": all_subjects, "grades": []},
            )

        last   = grades[-1]
        answer = f"{last['marks']}/{last['max']}"
        chain  = TOOL_CHAINS["grades_lookup"]

        return GroundTruth(
            correct_answer        = answer,
            answer_type           = "exact",
            intent                = "grades_lookup",
            required_tools        = chain,
            required_tool_families= ["lms"],
            optimal_steps         = len(chain) + 1,
            max_steps             = len(chain) + 1 + self.BUFFER,
            exit_paths            = {"NOT_ENROLLED": EXIT_CONDITIONS["NOT_ENROLLED"]},
            required_facts        = [answer, last.get("exam","")],
            oracle_data           = {
                "all_subjects": all_subjects,
                "grades":       grades,
                "last_grade":   last,
            },
        )

    def _notes_lookup(self, subject, db, all_subjects) -> GroundTruth:
        notes = db.get_notes(subject)
        chain = TOOL_CHAINS["notes_lookup"]

        if not notes:
            return GroundTruth(
                correct_answer        = "NO_DATA",
                answer_type           = "NO_DATA",
                intent                = "notes_lookup",
                required_tools        = chain,
                required_tool_families= ["lms"],
                optimal_steps         = len(chain) + 1,
                max_steps             = len(chain) + 1 + self.BUFFER,
                exit_paths            = {"NOT_ENROLLED": EXIT_CONDITIONS["NOT_ENROLLED"]},
                required_facts        = [],
                oracle_data           = {"all_subjects": all_subjects, "notes": []},
            )

        file_names = [n["file_name"] for n in notes]
        return GroundTruth(
            correct_answer        = json.dumps(file_names),
            answer_type           = "set",
            intent                = "notes_lookup",
            required_tools        = chain,
            required_tool_families= ["lms"],
            optimal_steps         = len(chain) + 1,
            max_steps             = len(chain) + 1 + self.BUFFER,
            exit_paths            = {"NOT_ENROLLED": EXIT_CONDITIONS["NOT_ENROLLED"]},
            required_facts        = file_names,
            oracle_data           = {"all_subjects": all_subjects, "notes": notes},
        )

    def _schedule_specific(self, subject, db, all_subjects,
                           today, intent) -> GroundTruth:
        schedule = db.get_schedule(subject)
        target   = today if intent == "schedule_specific" \
                   else today + timedelta(days=1)
        weekday  = target.strftime("%A")
        class_   = schedule.get(weekday)
        chain    = TOOL_CHAINS["schedule_specific"]

        if not class_:
            return GroundTruth(
                correct_answer        = "NO_CLASS",
                answer_type           = "NO_CLASS",
                intent                = intent,
                required_tools        = chain,
                required_tool_families= ["lms"],
                optimal_steps         = len(chain) + 1,
                max_steps             = len(chain) + 1 + self.BUFFER,
                exit_paths            = {
                    "NOT_ENROLLED": EXIT_CONDITIONS["NOT_ENROLLED"],
                    "NO_CLASS":     EXIT_CONDITIONS["NO_CLASS"],
                },
                required_facts        = [],
                oracle_data           = {
                    "all_subjects": all_subjects,
                    "schedule":     schedule,
                    "weekday":      weekday,
                },
            )

        answer = class_["time"]
        return GroundTruth(
            correct_answer        = answer,
            answer_type           = "exact",
            intent                = intent,
            required_tools        = chain,
            required_tool_families= ["lms"],
            optimal_steps         = len(chain) + 1,
            max_steps             = len(chain) + 1 + self.BUFFER,
            exit_paths            = {"NOT_ENROLLED": EXIT_CONDITIONS["NOT_ENROLLED"]},
            required_facts        = [answer, weekday],
            oracle_data           = {
                "all_subjects": all_subjects,
                "schedule":     schedule,
                "weekday":      weekday,
                "class":        class_,
            },
        )

    def _schedule_all_today(self, db, all_subjects, today, N) -> GroundTruth:
        weekday   = today.strftime("%A")
        # Oracle simulates N get_schedule calls
        today_map = {}
        for s in all_subjects:
            c = db.get_schedule(s).get(weekday)
            if c:
                today_map[s] = c["time"]

        # optimal: 1 (get_subjects) + N (get_schedule each) + 1 (RESPOND)
        optimal = N + 2

        if not today_map:
            return GroundTruth(
                correct_answer        = "NO_CLASS",
                answer_type           = "NO_CLASS",
                intent                = "schedule_all_today",
                required_tools        = ["get_subjects"] + ["get_schedule"] * N,
                required_tool_families= ["lms"],
                optimal_steps         = optimal,
                max_steps             = optimal + self.BUFFER,
                exit_paths            = {"NO_CLASS": EXIT_CONDITIONS["NO_CLASS"]},
                required_facts        = [],
                oracle_data           = {
                    "all_subjects": all_subjects,
                    "weekday":      weekday,
                    "today_map":    {},
                },
            )

        return GroundTruth(
            correct_answer        = json.dumps(today_map),
            answer_type           = "set",
            intent                = "schedule_all_today",
            # N individual get_schedule calls — reward model checks count not exact names
            required_tools        = ["get_subjects"] + ["get_schedule"] * N,
            required_tool_families= ["lms"],
            optimal_steps         = optimal,
            max_steps             = optimal + self.BUFFER,
            exit_paths            = {},
            required_facts        = list(today_map.values()) + list(today_map.keys()),
            oracle_data           = {
                "all_subjects": all_subjects,
                "weekday":      weekday,
                "today_map":    today_map,
            },
        )

    def _due_this_week(self, db, all_subjects, today, N) -> GroundTruth:
        due = {}
        for s in all_subjects:
            for a in db.get_assignments(s):
                d = date.fromisoformat(a["deadline"])
                if 0 <= (d - today).days <= 7:
                    due.setdefault(s, []).append(a["deadline"])

        # oracle simulates N get_assignments calls
        optimal = N + 2

        if not due:
            return GroundTruth(
                correct_answer        = "NO_DATA",
                answer_type           = "NO_DATA",
                intent                = "due_this_week",
                required_tools        = ["get_subjects"] + ["get_assignments"] * N,
                required_tool_families= ["lms"],
                optimal_steps         = optimal,
                max_steps             = optimal + self.BUFFER,
                exit_paths            = {},
                required_facts        = [],
                oracle_data           = {"all_subjects": all_subjects, "due": {}},
            )

        # Flatten required facts: all deadline strings that must appear
        all_deadlines = [d for dl in due.values() for d in dl]
        return GroundTruth(
            correct_answer        = json.dumps(due),
            answer_type           = "set",
            intent                = "due_this_week",
            required_tools        = ["get_subjects"] + ["get_assignments"] * N,
            required_tool_families= ["lms"],
            optimal_steps         = optimal,
            max_steps             = optimal + self.BUFFER,
            exit_paths            = {},
            required_facts        = all_deadlines + list(due.keys()),
            oracle_data           = {"all_subjects": all_subjects, "due": due},
        )

    def _study_plan_single(self, subject, assignments, details,
                           db, all_subjects) -> GroundTruth:
        notes    = db.get_notes(subject)
        schedule = db.get_schedule(subject)
        chain    = TOOL_CHAINS["study_plan_single"]

        # All facts from all 5 discovery steps must appear in response
        required_facts = (
            [a["deadline"] for a in assignments] +
            [a["name"]     for a in assignments] +
            [n["file_name"] for n in notes[:3]] +
            list(schedule.keys())   # class days
        )

        optimal = len(chain) + 1   # 5 calls + RESPOND = 6

        return GroundTruth(
            correct_answer        = "SYNTHESIS",
            answer_type           = "SYNTHESIS",
            intent                = "study_plan_single",
            required_tools        = chain,
            required_tool_families= ["lms"],
            optimal_steps         = optimal,
            max_steps             = optimal + self.BUFFER,
            exit_paths            = {
                "NOT_ENROLLED":   EXIT_CONDITIONS["NOT_ENROLLED"],
                "NO_ASSIGNMENTS": EXIT_CONDITIONS["NO_ASSIGNMENTS"],
            },
            required_facts        = required_facts,
            oracle_data           = {
                "all_subjects": all_subjects,
                "assignments":  assignments,
                "details":      details,
                "notes":        notes,
                "schedule":     schedule,
            },
        )

    def _study_plan_all(self, db, all_subjects, N) -> GroundTruth:
        required_facts = []
        oracle_data    = {"all_subjects": all_subjects, "per_subject": {}}

        for s in all_subjects:
            assigns  = db.get_assignments(s)
            notes    = db.get_notes(s)
            schedule = db.get_schedule(s)
            oracle_data["per_subject"][s] = {
                "assignments": assigns,
                "notes":       notes,
                "schedule":    schedule,
            }
            required_facts += [a["deadline"] for a in assigns]
            required_facts += [n["file_name"] for n in notes[:2]]
            required_facts += list(schedule.keys())

        # 1 + 3N + 1
        optimal = 3 * N + 2

        return GroundTruth(
            correct_answer        = "SYNTHESIS",
            answer_type           = "SYNTHESIS",
            intent                = "study_plan_all",
            required_tools        = (
                ["get_subjects"] +
                ["get_assignments", "get_notes", "get_schedule"] * N
            ),
            required_tool_families= ["lms"],
            optimal_steps         = optimal,
            max_steps             = optimal + self.BUFFER,
            exit_paths            = {},
            required_facts        = required_facts,
            oracle_data           = oracle_data,
        )

    def _resources_lookup(self, subject, assignments, details,
                          db, all_subjects) -> GroundTruth:
        notes  = db.get_notes(subject)
        chain  = TOOL_CHAINS["resources_lookup"]

        required_facts = (
            [a["deadline"] for a in assignments] +
            [n["file_name"] for n in notes]
        )

        return GroundTruth(
            correct_answer        = "SYNTHESIS",
            answer_type           = "SYNTHESIS",
            intent                = "resources_lookup",
            required_tools        = chain,
            required_tool_families= ["lms"],
            optimal_steps         = len(chain) + 1,
            max_steps             = len(chain) + 1 + self.BUFFER,
            exit_paths            = {
                "NOT_ENROLLED":   EXIT_CONDITIONS["NOT_ENROLLED"],
                "NO_ASSIGNMENTS": EXIT_CONDITIONS["NO_ASSIGNMENTS"],
            },
            required_facts        = required_facts,
            oracle_data           = {
                "all_subjects": all_subjects,
                "assignments":  assignments,
                "details":      details,
                "notes":        notes,
            },
        )

    def _material_sync(self, subject, db, all_subjects) -> GroundTruth:
        lms_notes   = db.get_notes(subject)
        local_notes = []                         # Files starts empty
        lms_set     = {n["file_name"] for n in lms_notes}
        local_set   = {n["file_name"] for n in local_notes}
        new_files   = list(lms_set - local_set)

        if not new_files:
            return GroundTruth(
                correct_answer        = "NO_NEW_FILES",
                answer_type           = "NO_NEW_FILES",
                intent                = "material_sync",
                required_tools        = EXIT_CONDITIONS["NO_NEW_FILES"]["required_tools"],
                required_tool_families= ["lms", "files"],
                optimal_steps         = EXIT_CONDITIONS["NO_NEW_FILES"]["optimal_steps"],
                max_steps             = EXIT_CONDITIONS["NO_NEW_FILES"]["optimal_steps"] + self.BUFFER,
                exit_paths            = {
                    "NOT_ENROLLED": EXIT_CONDITIONS["NOT_ENROLLED"],
                    "NO_NEW_FILES": EXIT_CONDITIONS["NO_NEW_FILES"],
                },
                required_facts        = [],
                oracle_data           = {
                    "all_subjects": all_subjects,
                    "lms_notes":    lms_notes,
                    "local_notes":  local_notes,
                },
            )

        writes  = [{"tool": "add_notes", "subject": subject, "file_name": f}
                   for f in new_files]
        # 1 + 1 + 1 + M + 1 (get_subjects, get_notes, get_allnotes, add_notes×M, RESPOND)
        optimal = len(new_files) + 4

        return GroundTruth(
            correct_answer        = f"synced {len(new_files)} files",
            answer_type           = "state_change",
            intent                = "material_sync",
            required_tools        = ["get_subjects","get_notes","get_allnotes","add_notes"],
            required_tool_families= ["lms", "files"],
            optimal_steps         = optimal,
            max_steps             = optimal + self.BUFFER,
            exit_paths            = {
                "NOT_ENROLLED": EXIT_CONDITIONS["NOT_ENROLLED"],
                "NO_NEW_FILES": EXIT_CONDITIONS["NO_NEW_FILES"],
            },
            required_facts        = new_files,
            expected_writes       = writes,
            oracle_data           = {
                "all_subjects": all_subjects,
                "lms_notes":    lms_notes,
                "local_notes":  local_notes,
                "new_files":    new_files,
            },
        )

    def _add_deadline_todo(self, subject, assignments,
                           db, all_subjects) -> GroundTruth:
        existing_todos = db.get_todos()
        existing_keys  = {(t["subject"], t["deadline"]) for t in existing_todos}

        deadline = assignments[0]["deadline"]
        name     = assignments[0]["name"]
        is_dup   = (subject, deadline) in existing_keys

        writes   = [] if is_dup else [{
            "tool":     "add_todo",
            "subject":  subject,
            "deadline": deadline,
        }]

        chain   = TOOL_CHAINS["add_deadline_todo"]
        optimal = len(chain) + 1   # 4 calls + RESPOND = 5

        return GroundTruth(
            correct_answer        = "DUPLICATE" if is_dup else f"todo added: {name}",
            answer_type           = "state_change",
            intent                = "add_deadline_todo",
            required_tools        = chain,
            required_tool_families= ["lms", "planner"],
            optimal_steps         = optimal,
            max_steps             = optimal + self.BUFFER,
            exit_paths            = {
                "NOT_ENROLLED":   EXIT_CONDITIONS["NOT_ENROLLED"],
                "NO_ASSIGNMENTS": EXIT_CONDITIONS["NO_ASSIGNMENTS"],
            },
            required_facts        = [deadline, name],
            expected_writes       = writes,
            oracle_data           = {
                "all_subjects":   all_subjects,
                "assignments":    assignments,
                "existing_todos": existing_todos,
                "is_duplicate":   is_dup,
                "deadline":       deadline,
            },
        )

    def _announcements_lookup(self, subject, db, all_subjects) -> GroundTruth:
        chain = TOOL_CHAINS["announcements_lookup"]
        return GroundTruth(
            correct_answer        = "SYNTHESIS",
            answer_type           = "SYNTHESIS",
            intent                = "announcements_lookup",
            required_tools        = chain,
            required_tool_families= ["lms"],
            optimal_steps         = len(chain) + 1,
            max_steps             = len(chain) + 1 + self.BUFFER,
            exit_paths            = {"NOT_ENROLLED": EXIT_CONDITIONS["NOT_ENROLLED"]},
            required_facts        = [],
            oracle_data           = {"all_subjects": all_subjects},
        )

    def _fallback(self, intent, all_subjects) -> GroundTruth:
        return GroundTruth(
            correct_answer        = "NO_DATA",
            answer_type           = "NO_DATA",
            intent                = intent,
            required_tools        = ["get_subjects"],
            required_tool_families= ["lms"],
            optimal_steps         = 2,
            max_steps             = 2 + self.BUFFER,
            exit_paths            = {},
            required_facts        = [],
            oracle_data           = {"all_subjects": all_subjects},
        )

    def _build_details(self, subject, assignments, db) -> dict:
        """Simulates get_assignment_details — what that tool actually returns."""
        notes = db.get_notes(subject)
        return {
            "requirements": f"Complete all {subject} exercises and submit",
            "materials":    notes[:2],
            "submission":   "submit via LMS portal before deadline",
        }

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