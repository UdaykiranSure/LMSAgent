import requests
import json
import time
import os
import re

# ── Categories ────────────────────────────────────────────────────────────────
TRAJECTORY_CATEGORIES = [
    "deadline_lookup",
    "grade_inquiry",
    "schedule_lookup",
    "assignment_details",
    "material_retrieval",
    "resource_search",
    "planning",
    "file_sync",
    "multi_intent",
    "cross_tool",
]

DEFAULT_PER_CATEGORY = 100
BATCH_SIZE = 10          # trajectories per single API call
OUT_PATH = "sft_dataset.json"
CHECKPOINT_DIR = "sft_checkpoints"

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """

You are a synthetic data generator for a student assistant agent training pipeline.

Your job is to produce realistic, complete multi-step trajectories that a student agent should follow when answering questions about their LMS (Learning Management System).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STUDENT CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The student is enrolled in these subjects:
  RL, SSL, DSA, DNN, NLP, CV, Deep Learning, Software Systems

Today's date: {TODAY}

The agent starts every session with ZERO knowledge about the student.
All facts must be discovered through tool calls — never assumed or hallucinated.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LMS tools:
  get_subjects()                          → list of enrolled subjects
  get_assignments(subject)               → list of assignments with deadlines
  get_assignment_details(subject)        → requirements, materials, submission info
  get_grades(subject)                    → list of exam results
  get_notes(subject)                     → list of uploaded lecture files
  get_announcements(subject)             → list of announcements
  get_schedule(subject)                  → weekly class timetable

Planner tools:
  get_todos()                            → existing todos
  add_todo(subject, title, deadline, priority)  → create a reminder

Files tools:
  get_allnotes(subject)                  → files already saved locally
  add_notes(subject, file_name)          → save a file from LMS to local storage

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — STRICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Every trajectory must follow this exact structure:

<|im_start|>system
[system prompt]<|im_end|>
<|im_start|>user
[the student's query]<|im_end|>
<|im_start|>assistant
[tool call]<|im_end|>
<|im_start|>tool [tool_name]
[JSON result]<|im_end|>
<|im_start|>assistant
[next tool call or final respond]<|im_end|>
...

Tool call syntax (one per assistant turn, never multiple):
  <tool_call>{"tool": "tool_name", "params": {"key": "value"}}</tool_call>

Final answer syntax:
  <respond>Your answer here</respond>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY DISCOVERY ORDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — ALWAYS call get_subjects() first, no exceptions.
  This confirms the subject exists before any further calls.
  If the subject is not in the returned list, respond immediately:
    <respond>You are not enrolled in [subject].</respond>

STEP 2 — Call the intent-specific tool (get_assignments, get_grades, etc.)
  If the result is empty (no assignments, no grades, etc.), respond immediately:
    <respond>No [data type] found for [subject].</respond>
  Do NOT call get_assignment_details if get_assignments returned empty.

STEP 3+ — Call additional tools only if the query genuinely needs them.
  Study plans need: get_assignments + get_assignment_details + get_notes + get_schedule
  Material sync needs: get_notes (LMS) + get_allnotes (local) + add_notes per new file
  Todo creation needs: get_assignments + get_todos + add_todo

NEVER:
  - Call the same tool with the same parameters twice
  - Answer without calling at least get_subjects()
  - Include dates, grades, or file names not present in tool results
  - Emit multiple tool calls in one assistant turn
  - Skip directly to get_assignments without first calling get_subjects()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO GENERATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You will be given a scenario block containing:
  - QUERY: the student's question
  - DB_STATE: simulated tool responses for each tool call the agent will make
  - INTENT: the type of query (for reference)

Generate the complete trajectory from system prompt through final <respond>.
Make the tool results realistic: use the exact values from DB_STATE in your response.
The final response must only contain facts visible in the tool results above it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EDGE CASES — generate these too, not just happy paths
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NOT_ENROLLED:    Subject not in get_subjects() result → exit after step 1
NO_ASSIGNMENTS:  get_assignments() returns [] → exit after step 2, do not call get_assignment_details
NO_CLASS:        get_schedule() has no entry for today's weekday → say no class today
NO_DATA:         get_grades() / get_notes() returns [] → say nothing found
OVERDUE:         assignment deadline is before today → flag it as overdue in the response
DUPLICATE_TODO:  get_todos() already contains this subject+deadline → say already exists, skip add_todo
NO_NEW_FILES:    get_allnotes() already contains all files from get_notes() → say already synced

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITY CRITERIA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A high-quality trajectory:
  ✓ Calls get_subjects() exactly once, as the very first tool call
  ✓ Uses only the minimum tools needed (no extra exploratory calls)
  ✓ Reaches <respond> in optimal steps (see intent step counts below)
  ✓ Final response only contains facts from the tool results above it
  ✓ Handles empty results with an informative, polite message
  ✓ For multi-subject queries, calls per-subject tools in a consistent order

Optimal step counts by intent:
  deadline_lookup:    4 steps  (get_subjects → get_assignments → get_assignment_details → respond)
  grades_lookup:      3 steps  (get_subjects → get_grades → respond)
  notes_lookup:       3 steps  (get_subjects → get_notes → respond)
  schedule_specific:  3 steps  (get_subjects → get_schedule → respond)
  schedule_all_today: N+2 steps (get_subjects → get_schedule×N → respond)
  due_this_week:      N+2 steps (get_subjects → get_assignments×N → respond)
  study_plan_single:  6 steps  (get_subjects → get_assignments → get_assignment_details → get_notes → get_schedule → respond)
  material_sync:      M+4 steps (get_subjects → get_notes → get_allnotes → add_notes×M → respond)
  add_deadline_todo:  5 steps  (get_subjects → get_assignments → get_todos → add_todo → respond)

where N = number of enrolled subjects, M = number of new files to sync

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENARIO BLOCK FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUERY: <the student's natural language question>
INTENT: <intent type>
TODAY: <YYYY-MM-DD>
DB_STATE:
  get_subjects: ["RL", "SSL", "DSA", "DNN", "NLP", "CV", "Deep Learning", "Software Systems"]
  get_assignments("DSA"): [{"name": "Sorting Algorithms HW", "deadline": "2026-04-10", "description": "Implement merge sort and quicksort"}]
  get_assignment_details("DSA"): [{"name": "Sorting Algorithms HW", "deadline": "2026-04-10", "requirements": "Complete all DSA tasks", "materials": []}]



SIMULATED ENVIRONMENT STATE
When generating trajectories, use this consistent environment state so all trajectories are coherent with each other:
Enrolled Subjects:

RL – Reinforcement Learning
DNN – Deep Neural Networks
NLP – Natural Language Processing
CV – Computer Vision
SSL – Statistical and Supervised Learning
Algorithms – Design and Analysis of Algorithms
SWS – Software Systems Lab

Current Date: Monday, 2025-02-10
Assignments:
SubjectTitleDeadlineStatusRLPolicy Gradient Implementation2025-02-14 (Friday)PendingAlgorithmsDivide & Conquer Problem Set2025-02-12 (Wednesday)PendingDNNCNN Architecture Report2025-02-20PendingNLPTransformer Fine-tuning2025-02-28Not Started
Schedule (Weekly):
SubjectDayTimeRLMonday10:00–11:30DNNMonday14:00–15:30NLPTuesday09:00–10:30CVTuesday14:00–15:30SSLWednesday10:00–11:30AlgorithmsThursday09:00–10:30SWSFriday14:00–16:00
Recent Grades (SSL):

Mid-Exam 1: 78/100 (2025-01-15)
Quiz 2: 18/20 (2025-01-28)

Recent Announcements:

SWS (2025-02-09): "New lab materials posted: Docker Setup Guide and CI/CD Pipeline notes"
RL (2025-02-08): "Assignment rubric updated, check requirements"


QUERY CATEGORIES
Generate trajectories spanning all these categories with realistic distribution:
CategoryDescriptionExample Queriesdeadline_lookupFinding assignment/exam deadlines"When is RL assignment due?"grade_inquiryFetching exam marks"What are my SSL grades?"schedule_lookupClass timings for today/tomorrow/specific day"What classes do I have today?"assignment_detailsRequirements, materials for an assignment"What do I need to submit for DNN?"material_retrievalGetting lecture notes or study material"Get me NLP exam syllabus"resource_searchFinding external resources via browser"Find me resources for RL policy gradient"planningStudy plans, todo management"Make me a study plan for CV exam"file_syncSyncing LMS material to personal files"Add SWS new materials to my files"multi_intentQuery covers multiple needs"What's due this week and give me a study plan?"cross_toolRequires 3+ tools to complete"Prepare me for tomorrow's exam"

COMPLEXITY LEVELS
Simple (1–2 tool calls)

Single intent, single subject, direct lookup
Example: "When is the RL assignment deadline?"
Expected: 1 call to LMS.get_assignments() or LMS.get_assignment_details("RL")

Moderate (3–5 tool calls)

Multi-step retrieval, light planning, or one cross-tool operation
Example: "What do I need to complete for DNN's assignment and add it as a TODO?"
Expected: get_assignment_details("DNN") → add_todo(...)

Complex (5–10 tool calls)

Full cross-tool workflows, study plan generation, file sync
Example: "Make me a study plan for mid-exams"
Expected: get_subjects() → get_assignments() → get_schedule() (per subject) → get_notes() (per subject) → get_todos() → multiple add_todo() calls


DIVERSITY REQUIREMENTS
When generating a batch of N trajectories, ensure:

At least 30% complex trajectories
Cover all 10 query categories at least once per 20 trajectories
Vary time references: today, tomorrow, this week, specific dates
Include edge cases:

Subject has no assignments → agent gracefully reports none
Browser search returns no useful result → agent falls back or reports
TODO already exists → agent checks before adding duplicate
File already synced → agent skips re-adding


Vary subject names: don't always use RL; distribute across all 6 subjects
Include multi-turn style for complex queries (agent checks one thing, finds it needs more, continues)


"""


# ── Model client ──────────────────────────────────────────────────────────────
class SFTGenerator:
    def __init__(
        self,
        base_url="http://localhost:4141/v1/chat/completions",
        model_name="gpt-5-mini",
        timeout=120,
        max_retries=3,
        max_tokens=16384,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.8,
            "max_tokens": self.max_tokens,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.base_url, json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code
                if 400 <= status < 500 and status != 429:
                    print(f"  [Client error {status}] {e.response.text}")
                    raise
                if attempt < self.max_retries - 1:
                    wait = 5 * (2 ** attempt)
                    print(f"  [HTTP {status}] retrying in {wait}s…")
                    time.sleep(wait)
                else:
                    raise

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    wait = 5 * (2 ** attempt)
                    print(f"  [Timeout] retrying in {wait}s…")
                    time.sleep(wait)
                else:
                    raise

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    print(f"  [Request error] {e} — retrying…")
                    time.sleep(2 ** attempt)
                else:
                    raise

    def generate_batch(self, category: str, count: int) -> list[dict]:
        """Generate `count` trajectories for `category`, return as list of dicts."""
        user_prompt = (
            f"Generate exactly {count} trajectories of intent type: **{category}**.\n\n"
            "Requirements:\n"
            "- Mix simple, moderate, and complex examples (at least 30% complex).\n"
            "- Include at least one edge-case variant (e.g. NOT_ENROLLED, NO_DATA, OVERDUE, "
            "DUPLICATE_TODO, NO_NEW_FILES) per 10 trajectories.\n"
            "- Vary the subject across RL, DNN, NLP, CV, SSL, Algorithms, SWS.\n"
            "- Each trajectory must be a complete, standalone string following the "
            "<|im_start|>/<|im_end|> format defined in the system prompt.\n\n"
            "Return ONLY a valid JSON array of strings. No prose, no markdown fences, "
            "no explanation — just the raw JSON array.\n"
            'Example shape: ["<|im_start|>system\\n...trajectory 1...\\n", '
            '"<|im_start|>system\\n...trajectory 2...\\n"]'
        )
        raw = self._call(SYSTEM_PROMPT, user_prompt)
        return _parse_json_list(raw)


# ── JSON parsing ──────────────────────────────────────────────────────────────
def _parse_json_list(text: str) -> list:
    """
    Try to extract a JSON array from the model response.
    Handles: raw JSON, markdown fences, leading/trailing prose.
    """
    # 1. Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2. Find the outermost JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        # Model wrapped it in an object
        for v in result.values():
            if isinstance(v, list):
                return v
    except json.JSONDecodeError:
        pass

    # 3. Last resort: treat the whole response as a single trajectory
    print("  [Warning] Could not parse JSON array; wrapping raw text as single item.")
    return [text]


# ── System-prompt count parser ────────────────────────────────────────────────
def extract_category_counts(prompt: str, categories: list[str]) -> dict[str, int]:
    """
    Scan the system prompt for lines like 'generate 50 deadline_lookup trajectories'.
    Falls back to DEFAULT_PER_CATEGORY for any category not found.
    """
    counts: dict[str, int] = {}
    for cat in categories:
        # Look for patterns like "50 deadline_lookup" or "deadline_lookup: 50"
        pattern = rf"(?:(\d+)\s+{re.escape(cat)}|{re.escape(cat)}\s*[:\-]\s*(\d+))"
        m = re.search(pattern, prompt, re.IGNORECASE)
        if m:
            counts[cat] = int(m.group(1) or m.group(2))
        else:
            counts[cat] = DEFAULT_PER_CATEGORY
    return counts


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def _checkpoint_path(category: str) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"{category}.json")


def _load_checkpoint(category: str) -> list:
    path = _checkpoint_path(category)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def _save_checkpoint(category: str, trajectories: list) -> None:
    with open(_checkpoint_path(category), "w") as f:
        json.dump(trajectories, f, indent=2)


# ── Main generation loop ──────────────────────────────────────────────────────
def generate_dataset(
    out_path: str = OUT_PATH,
    batch_size: int = BATCH_SIZE,
) -> None:
    generator = SFTGenerator()
    category_counts = extract_category_counts(SYSTEM_PROMPT, TRAJECTORY_CATEGORIES)

    total_target = sum(category_counts.values())
    print(f"Target: {total_target} trajectories across {len(TRAJECTORY_CATEGORIES)} categories")
    print(f"Batch size: {batch_size} per API call\n")

    all_trajectories: list[dict] = []

    for category in TRAJECTORY_CATEGORIES:
        target = category_counts[category]
        existing = _load_checkpoint(category)

        if len(existing) >= target:
            print(f"  [{category}] already have {len(existing)}/{target} — skipping")
            all_trajectories.extend(existing[:target])
            continue

        print(f"  [{category}] generating {target} trajectories…")
        collected = list(existing)  # resume from checkpoint

        while len(collected) < target:
            remaining = target - len(collected)
            this_batch = min(batch_size, remaining)
            batch_num = len(collected) // batch_size + 1
            print(f"    batch {batch_num}: requesting {this_batch} … ", end="", flush=True)

            try:
                batch = generator.generate_batch(category, this_batch)
                tagged = [{"intent": category, "trajectory": t} for t in batch]
                collected.extend(tagged)
                _save_checkpoint(category, collected)
                print(f"got {len(batch)} (total {len(collected)}/{target})")
            except Exception as e:
                print(f"FAILED — {e}")
                print("    Waiting 10s before continuing…")
                time.sleep(10)

        all_trajectories.extend(collected[:target])
        print(f"  [{category}] done ({len(collected[:target])} saved)\n")

    with open(out_path, "w") as f:
        json.dump(all_trajectories, f, indent=2)

    print(f"Dataset saved → {out_path}  ({len(all_trajectories)} trajectories total)")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_dataset()
