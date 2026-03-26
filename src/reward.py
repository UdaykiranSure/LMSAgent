"""
src/reward.py

Standalone rule-based reward model.
Scores a completed trajectory against ground truth from the oracle.
No neural components — pure deterministic rules.
"""

import re
import json
import logging
from datetime import date

logger = logging.getLogger(__name__)


class RuleBasedRewardModel:

    def __init__(self, weights: dict = None):
        self.w = weights or {
            "correctness":    1.0,
            "tool_path":      0.4,
            "subject_scope":  0.3,
            "temporal":       0.3,
            "write_ops":      0.6,
            "step_limit":     0.2,
        }

    def score(self, trajectory: dict, gt) -> dict:
        scores = {
            "correctness":   self._correctness(trajectory, gt),
            "grounding":     self._grounding(trajectory, gt),
            "tool_path":     self._tool_path(trajectory, gt),
            "step_limit":    self._step_limit(trajectory, gt),
            "write_ops":     self._write_ops(trajectory, gt),
        }

        r = scores["correctness"] * self.w["correctness"]
        r += scores["tool_path"]  * self.w["tool_path"]
        r += scores["step_limit"] * self.w["step_limit"]
        r += scores["write_ops"]  * self.w["write_ops"]

        # Hallucination: multiplicative collapse
        if scores["grounding"] < 0:
            r = scores["grounding"]

        total = max(-1.0, min(1.0, r))
        return {"total": total, "breakdown": scores}

    def _correctness(self, traj: dict, gt) -> float:
        resp     = (traj.get("final_response") or "").lower()
        answer   = (gt.correct_answer or "").lower()
        atype    = gt.answer_type

        if atype == "NOT_ENROLLED":
            return 1.0 if any(p in resp for p in
                              ["not enrolled","not found","don't have","no subject"]) else -1.0

        if atype in ("NO_DATA", "NO_ASSIGNMENTS", "NO_CLASS"):
            return 1.0 if any(p in resp for p in
                              ["no assignment","no class","not found","none","no data",
                               "hasn't been","not been posted"]) else -0.5

        if atype == "SYNTHESIS":
            if not gt.required_facts:
                return 0.5
            found = sum(1 for f in gt.required_facts if f.lower() in resp)
            cov   = found / len(gt.required_facts)
            if cov >= 0.9: return 1.0
            if cov >= 0.6: return 0.5
            if cov >= 0.3: return 0.0
            return -0.3

        if atype == "state_change":
            return 0.5   # write ops scored separately

        # exact / set
        if answer and answer in resp:
            return 1.0
        if answer:
            # partial: key entity present
            key = answer.split()[0] if answer else ""
            if key and key in resp:
                return 0.5
        return 0.0

    def _grounding(self, traj: dict, gt) -> float:
        """Check that response only contains facts present in tool results."""
        tool_data = " ".join(
            json.dumps(c.get("result", {})).lower()
            for c in traj.get("tool_calls", [])
        )
        resp = (traj.get("final_response") or "").lower()

        # Extract dates mentioned in response
        resp_dates = re.findall(r'\d{4}-\d{2}-\d{2}', resp)
        for d in resp_dates:
            if d not in tool_data:
                return -1.0   # hallucinated date

        # Extract numbers that look like grades
        resp_grades = re.findall(r'\b\d{2,3}/\d{2,3}\b', resp)
        for g in resp_grades:
            if g not in tool_data:
                return -1.0

        return 0.0

    def _tool_path(self, traj: dict, gt) -> float:
        called = [c["tool"] for c in traj.get("tool_calls", [])]
        score  = 0.0

        # get_subjects must be first
        if called and called[0] == "get_subjects":
            score += 0.2
        elif called:
            score -= 0.3

        # All required tools called
        required = gt.required_tools or []
        # required_tools may have repeated names (e.g. get_assignments × N)
        # just check unique tools are present
        unique_required = set(required)
        unique_called   = set(called)
        missing = unique_required - unique_called
        score -= 0.15 * len(missing)

        # No duplicate calls
        seen = {}
        for c in traj.get("tool_calls", []):
            key = (c["tool"], json.dumps(c.get("params", {}), sort_keys=True))
            if key in seen:
                score -= 0.1
            seen[key] = True

        # Efficiency bonus
        optimal = getattr(gt, "optimal_steps", None)
        actual  = traj.get("steps_used", 999)
        if optimal and actual <= optimal:
            score += 0.2

        return max(-1.0, min(0.5, score))

    def _step_limit(self, traj: dict, gt) -> float:
        actual  = traj.get("steps_used", 0)
        max_s   = getattr(gt, "max_steps", 10)
        if actual == 0:
            return -0.5
        if actual >= max_s:
            return -0.3
        return 0.0

    def _write_ops(self, traj: dict, gt) -> float:
        expected = getattr(gt, "expected_writes", [])
        if not expected:
            return 0.0

        actual_writes = [
            c for c in traj.get("tool_calls", [])
            if c["tool"] in ("add_todo", "add_notes", "update_todo")
        ]
        score = 0.0
        for exp in expected:
            matched = any(
                a["tool"] == exp["tool"] and
                a.get("params", {}).get("subject") == exp.get("subject")
                for a in actual_writes
            )
            score += 0.5 if matched else -0.3

        return max(-1.0, min(1.0, score))
