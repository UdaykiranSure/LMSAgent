from dataclasses import dataclass, field
from datetime import date
from typing import Optional
import json, re
from utils import generate_random_id



@dataclass
class AgentStep:
    step_num: int
    tool_call: Optional[dict]    # None if this is a RESPOND step
    tool_result: Optional[dict]  # None if RESPOND
    response: Optional[str]      # None if tool call
    is_terminal: bool

@dataclass
class EpisodeState:
    query: str
    today: date
    steps: list[AgentStep] = field(default_factory=list)
    tool_calls_made: list[dict] = field(default_factory=list)
    context_window: list[dict]  = field(default_factory=list)
    done: bool = False
    final_response: Optional[str] = None

class StudentAgentEnvironment:
    """
    Zero-knowledge multi-step environment.
    Agent must call get_subjects() before anything else.
    max_steps is set dynamically based on oracle's knowledge of N subjects.
    """

    def __init__(self, mock_db, ground_truth, query: str, today: date):
        self.db           = mock_db
        self.ground_truth = ground_truth
        self.query        = query
        self.today        = today
        self.max_steps    = ground_truth.optimal_steps + 3  # buffer above optimal
        self.state        = EpisodeState(query=query, today=today)

        # Seed the context with system prompt + user query
        self.state.context_window = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": query},
        ]

    def step(self, agent_output: str) -> dict:
        """
        Process one agent output token.
        Returns: {observation, reward_so_far, done, info}
        """
        step_num = len(self.state.steps) + 1

        # ── Hard step limit ────────────────────────────────────────────
        if step_num > self.max_steps:
            self.state.done = True
            return self._build_obs(
                reward=-0.5,
                done=True,
                info={"reason": "step_limit_exceeded"}
            )

        # ── Soft warning at 80% budget ─────────────────────────────────
        if step_num == int(self.max_steps * 0.8):
            warning = f"\n[System: {self.max_steps - step_num} steps remaining. Wrap up soon.]"
            self.state.context_window.append(
                {"role": "system", "content": warning}
            )
        self.state.context_window.append(
            {"role": "assistant", "content": agent_output}
        )
        # ── Parse agent output: tool call or respond ───────────────────
        tool_match    = re.search(r'<tool_call>(.*?)</tool_call>', agent_output, re.DOTALL)
        respond_match = re.search(r'<respond>(.*?)</respond>',    agent_output, re.DOTALL)

        if respond_match:
            return self._handle_respond(respond_match.group(1).strip(), step_num)

        if tool_match:
            try:
                call = json.loads(tool_match.group(1).strip())
                return self._handle_tool_call(call, step_num)
            except json.JSONDecodeError:
                return self._handle_malformed(step_num)

        # Neither tag found — treat as malformed
        return self._handle_malformed(step_num)

    # ── Tool call handler ──────────────────────────────────────────────
    def _handle_tool_call(self, call: dict, step_num: int) -> dict:
        tool   = call.get("tool", "")
        params = call.get("params", {})

        # ── Enforcement: must call get_subjects first ──────────────────
        already_called = [c["tool"] for c in self.state.tool_calls_made]
        if tool != "get_subjects" and "get_subjects" not in already_called:
            # Inject a nudge — don't penalise yet, just redirect
            nudge = "[System: You must call get_subjects() before any subject-specific tool.]"
            self.state.context_window.append({"role": "system", "content": nudge})
            return self._build_obs(reward=-0.1, done=False,
                                   info={"reason": "skipped_get_subjects"})

        # ── Duplicate call check ───────────────────────────────────────
        call_key = (tool, json.dumps(params, sort_keys=True))
        seen_keys = {(c["tool"], json.dumps(c.get("params",{}), sort_keys=True))
                     for c in self.state.tool_calls_made}
        if call_key in seen_keys:
            result = {"error": "duplicate_call", "message": "Already called with same args"}
            # self.state.context_window.append(
            #     {"role": "tool", "name": tool, "content": json.dumps(result), "tool_call_id":generate_random_id()}
            # )
            self.state.context_window.append(
            {
                "role": "user",
                 "content": f"Tool result for {tool}:\n{json.dumps(result)}"
            }
            )
            return self._build_obs(reward=-0.1, done=False,
                                   info={"reason": "duplicate_tool_call"})

        # ── Execute tool ───────────────────────────────────────────────
        result, shape_reward = self._execute_tool(tool, params)

        # Record step
        self.state.tool_calls_made.append({**call, "result": result})
        self.state.steps.append(AgentStep(
            step_num=step_num, tool_call=call,
            tool_result=result, response=None, is_terminal=False
        ))

        # self.state.context_window.append(
        #     {"role": "tool", "name": tool, "content": json.dumps(result), "tool_call_id":generate_random_id()}
        # )
        self.state.context_window.append(
            {
                "role": "user",
                 "content": f"Tool result for {tool}:\n{json.dumps(result)}"
            }
        )
        return self._build_obs(reward=shape_reward, done=False,
                               info={"tool": tool, "result_size": len(str(result))})

    # ── Tool executor — maps tool name to mock DB method ──────────────
    def _execute_tool(self, tool: str, params: dict) -> tuple[dict, float]:
        """Returns (result_dict, shaping_reward)"""
        db = self.db
        try:
            if tool == "get_subjects":
                result = db.get_subjects()
                return {"subjects": result}, 0.05   # small bonus for first discovery

            elif tool == "get_assignments":
                subj = params.get("subject", "")
                # Validate subject exists (agent should have confirmed via get_subjects)
                if subj not in db.get_subjects():
                    return {"error": f"{subj} not found in enrolled subjects"}, -0.2
                result = db.get_assignments(subj)
                reward = 0.1 if result else 0.05   # reward for checking even if empty
                return {"assignments": result}, reward

            elif tool == "get_assignment_details":
                subj = params.get("subject", "")
                assignments = db.get_assignments(subj)
                if not assignments:
                    return {"error": "No assignments to get details for"}, -0.1
                # Return full details for each assignment
                details = [{
                    **a,
                    "requirements": f"Complete all {subj} exercises",
                    "materials":    db.get_notes(subj)[:2],   # attach top 2 notes
                    "submission":   f"submit via LMS before deadline"
                } for a in assignments]
                return {"details": details}, 0.1

            elif tool == "get_grades":
                subj = params.get("subject", "")
                result = db.get_grades(subj)
                return {"grades": result}, 0.05

            elif tool == "get_notes":
                subj = params.get("subject", "")
                result = db.get_notes(subj)
                return {"notes": result}, 0.05

            elif tool == "get_schedule":
                subj = params.get("subject", "")
                if subj:
                    result = db.get_schedule(subj)
                else:
                    result = db.get_schedule()
                return {"schedule": result}, 0.05

            elif tool == "get_announcements":
                subj = params.get("subject", "")
                # Mock: randomly generate announcements
                import random
                result = [{"text": f"No new announcements for {subj}"}] \
                         if random.random() < 0.5 else \
                         [{"text": f"Exam rescheduled for {subj}",
                           "date": self.today.isoformat()}]
                return {"announcements": result}, 0.05

            elif tool == "get_todos":
                return {"todos": db.get_todos()}, 0.05

            elif tool == "add_todo":
                subj     = params.get("subject", "")
                title    = params.get("title", "")
                deadline = params.get("deadline", "")
                priority = params.get("priority", "medium")

                # Validate: deadline must have been seen in a tool result
                seen_deadlines = self._extract_seen_deadlines()
                if deadline not in seen_deadlines:
                    return {"error": "Deadline not found in any tool result"}, -0.3

                result = db.add_todo(subj, title, deadline, priority)
                return {"created": result}, 0.2   # write op bonus

            elif tool == "get_allnotes":
                subj = params.get("subject", "")
                # Files tool — returns what's already saved locally
                result = []  # starts empty in fresh session
                return {"local_notes": result}, 0.05

            elif tool == "add_notes":
                subj      = params.get("subject", "")
                file_name = params.get("file_name", "")

                # Validate: file must have been seen in a get_notes result
                seen_files = self._extract_seen_files(subj)
                if file_name not in seen_files:
                    return {"error": f"{file_name} not seen in any LMS tool result"}, -0.3

                result = db.add_notes(subj, file_name, content="[synced from LMS]")
                return {"saved": result}, 0.2

            else:
                return {"error": f"Unknown tool: {tool}"}, -0.2

        except Exception as e:
            return {"error": str(e)}, -0.1

    # ── RESPOND handler ───────────────────────────────────────────────
    def _handle_respond(self, response: str, step_num: int) -> dict:
        self.state.done           = True
        self.state.final_response = response
        self.state.steps.append(AgentStep(
            step_num=step_num, tool_call=None,
            tool_result=None, response=response, is_terminal=True
        ))
        return self._build_obs(reward=None, done=True,
                               info={"reason": "responded"})

    def _handle_malformed(self, step_num: int) -> dict:
        self.state.context_window.append({
            "role": "system",
            "content": "[System: Malformed output. Use <tool_call>{...}</tool_call> or <respond>...</respond>]"
        })
        return self._build_obs(reward=-0.05, done=False,
                               info={"reason": "malformed_output"})

    # ── Helpers ───────────────────────────────────────────────────────
    def _extract_seen_deadlines(self) -> set:
        """All deadline strings that appeared in any tool result."""
        deadlines = set()
        for c in self.state.tool_calls_made:
            result_str = json.dumps(c.get("result", {}))
            import re
            deadlines.update(re.findall(r'\d{4}-\d{2}-\d{2}', result_str))
        return deadlines

    def _extract_seen_files(self, subject: str) -> set:
        """All file_names that appeared in get_notes results for this subject."""
        files = set()
        for c in self.state.tool_calls_made:
            if c["tool"] == "get_notes" and c.get("params",{}).get("subject") == subject:
                for note in c.get("result", {}).get("notes", []):
                    files.add(note.get("file_name", ""))
        return files

    def _build_obs(self, reward, done, info) -> dict:
        return {
            "context_window":    self.state.context_window.copy(),
            "steps_used":        len(self.state.steps),
            "max_steps":         self.max_steps,
            "steps_remaining":   self.max_steps - len(self.state.steps),
            "done":              done,
            "intermediate_reward": reward,
            "info":              info,
        }

    def build_trajectory(self) -> dict:
        """Called at episode end to produce the object fed to the reward model."""
        from datetime import datetime
        return {
            "query":          self.query,
            "tool_calls":     self.state.tool_calls_made,
            "final_response": self.state.final_response or "",
            "steps_used":     len(self.state.steps),
            "max_steps":      self.max_steps,
            "today":          self.today,
            "intent":         self.ground_truth.intent
                              if hasattr(self.ground_truth, "intent") else "unknown",
            "context_window": self.state.context_window,
        }

SYSTEM_PROMPT = """
You are a student assistant agent. You have ZERO prior knowledge about the student.
You must call tools to discover all information before answering.

You ALWAYS follow this discovery order:
  1. get_subjects()             — confirm subject exists and get exact name
  2. get_assignments(subject)   — check if any assignments exist
  3. get_assignment_details(subject) — get full details if needed
  4. [other tools as needed]
  5. RESPOND with your answer

Rules:
- Never assume a subject name. Always confirm via get_subjects() first.
- Never answer from memory. Every fact must come from a tool call result.
- If a subject is not in get_subjects(), immediately RESPOND that the student is not enrolled.
- If get_assignments() returns empty, immediately RESPOND that no assignments exist.
- Do not call the same tool with the same arguments twice.

Emit tool calls in this exact format:
<tool_call>{"tool": "get_subjects", "params": {}}</tool_call>

When ready to answer, emit:
<respond>Your answer here</respond>
"""