# src/dataset.py
"""
EpisodeDataset — manages query templates and samples episodes
with curriculum-aware weighting.

Each template is tagged with:
  - level:   1 | 2 | 3
  - intent:  the canonical intent string the oracle will parse
  - subject: "named" | "all" | None
"""

import random
from dataclasses import dataclass
from datetime import date
from typing import Optional

from src.mock_db import MockStudentDB
from src.oracle import Oracle, GroundTruth


@dataclass
class QueryTemplate:
    template: str          # e.g. "when is {subject} assignment deadline"
    level: int             # 1 | 2 | 3
    intent: str            # canonical intent key
    needs_subject: bool    # True if {subject} placeholder present


# ─────────────────────────────────────────────────────────────────
# All query templates — exhaustive coverage of the intent space
# ─────────────────────────────────────────────────────────────────

QUERY_TEMPLATES = [

    # ── Level 1: single-subject, single-tool-family ────────────────
    QueryTemplate("when is {subject} assignment deadline",
                  1, "deadline_lookup", True),
    QueryTemplate("what is the deadline for {subject} homework",
                  1, "deadline_lookup", True),
    QueryTemplate("what are my marks in {subject}'s last exam",
                  1, "grades_lookup", True),
    QueryTemplate("show me my {subject} grades",
                  1, "grades_lookup", True),
    QueryTemplate("get notes for {subject}",
                  1, "notes_lookup", True),
    QueryTemplate("what study materials are available for {subject}",
                  1, "notes_lookup", True),
    QueryTemplate("are there any announcements for {subject}",
                  1, "announcements_lookup", True),
    QueryTemplate("what are the requirements for {subject} assignment",
                  1, "resources_lookup", True),
    QueryTemplate("is there a {subject} class today",
                  1, "schedule_specific", True),
    QueryTemplate("when is {subject} class",
                  1, "schedule_specific", True),

    # ── Level 2: temporal, multi-subject, date-sensitive ──────────
    QueryTemplate("are there any assignments due this week",
                  2, "due_this_week", False),
    QueryTemplate("what assignments do I have due this week",
                  2, "due_this_week", False),
    QueryTemplate("what are today's classes",
                  2, "schedule_all_today", False),
    QueryTemplate("what classes do I have today",
                  2, "schedule_all_today", False),
    QueryTemplate("what are tomorrow's classes",
                  2, "schedule_tomorrow", False),
    QueryTemplate("do I have any classes tomorrow",
                  2, "schedule_tomorrow", False),
    QueryTemplate("what is the syllabus for {subject} exam",
                  2, "resources_lookup", True),
    QueryTemplate("get me resources for {subject} exam tomorrow",
                  2, "resources_lookup", True),
    QueryTemplate("what are the upcoming deadlines",
                  2, "due_this_week", False),
    QueryTemplate("show me all my pending assignments",
                  2, "due_this_week", False),

    # ── Level 3: synthesis, write ops, multi-step ─────────────────
    QueryTemplate("give me a study plan for {subject} exam",
                  3, "study_plan_single", True),
    QueryTemplate("create a study schedule for {subject}",
                  3, "study_plan_single", True),
    QueryTemplate("make me a study plan for mid examinations",
                  3, "study_plan_all", False),
    QueryTemplate("prepare a study plan for all my exams",
                  3, "study_plan_all", False),
    QueryTemplate("add the newly posted materials for {subject} to my files",
                  3, "material_sync", True),
    QueryTemplate("sync {subject} lecture notes to my local files",
                  3, "material_sync", True),
    QueryTemplate("add a todo for {subject} assignment deadline",
                  3, "add_deadline_todo", True),
    QueryTemplate("remind me about the {subject} deadline",
                  3, "add_deadline_todo", True),
    QueryTemplate("add all upcoming deadlines to my planner",
                  3, "add_all_todos", False),
    QueryTemplate("what do I need to complete for {subject} assignment",
                  3, "resources_lookup", True),
]


class EpisodeDataset:
    """
    Samples (query, mock_db, ground_truth) triples for training.

    Usage:
        dataset = EpisodeDataset(config, stage=2)
        query, db, gt = dataset.sample()
    """

    def __init__(self, config: dict, stage: int = 2, seed: int = 42):
        self.config   = config
        self.stage    = stage
        self.rng      = random.Random(seed)

        # Filter and weight templates by curriculum config
        active_levels = config["curriculum"][f"stage{stage}_levels"]
        self.templates = [t for t in QUERY_TEMPLATES if t.level in active_levels]

        weights_cfg = config["curriculum"]["level_weights"][f"stage{stage}"]
        self.level_weights = {
            level: weights_cfg[level - 1]
            for level in [1, 2, 3]
        }

        self.oracle = Oracle()

    def sample(self, today: Optional[date] = None) -> tuple[str, MockStudentDB, GroundTruth]:
        """
        Returns one (query_string, mock_db, ground_truth) triple.
        The mock_db is fresh and randomised.
        The ground_truth is derived by the oracle before the episode starts.
        """
        today = today or date.today()
        db    = MockStudentDB(today)

        # Sample a level, then a template within that level
        template = self._sample_template()

        # Fill the {subject} placeholder
        if template.needs_subject:
            subject = self.rng.choice(db.get_subjects())
            query   = template.template.replace("{subject}", subject)
        else:
            query = template.template

        # Oracle derives ground truth from the same DB
        gt = self.oracle.derive(query, db, today)

        return query, db, gt

    def sample_batch(
        self,
        batch_size: int,
        today: Optional[date] = None
    ) -> list[tuple[str, MockStudentDB, GroundTruth]]:
        return [self.sample(today) for _ in range(batch_size)]

    def _sample_template(self) -> QueryTemplate:
        """Weighted sampling: pick level first, then template within level."""
        levels     = [t.level for t in self.templates]
        level_pool = [t.level for t in self.templates if t.level in self.level_weights]
        weights    = [self.level_weights.get(t.level, 0.0) for t in self.templates]

        if sum(weights) == 0:
            return self.rng.choice(self.templates)

        chosen = self.rng.choices(self.templates, weights=weights, k=1)[0]
        return chosen
