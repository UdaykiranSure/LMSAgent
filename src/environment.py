"""
src/environment.py

StudentAgentEnvironment — multi-step tool execution loop.
The agent calls tools one step at a time; this class executes them
against the MockStudentDB and manages context state.
"""

import re
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

UNIVERSAL_FIRST = "get_subjects"


@dataclass
class AgentStep:
    step_num:    int
    tool_call:   Optional[dict]
    tool_result: Optional[dict]
    response:    Optional[str]
    is_terminal: bool


@dataclass
class EpisodeState:
    query:          str
    today:          date
    steps:          list = field(default_factory=list)
    tool_calls_made:list = field(default_factory=list)
    context_window: list = field(default_factory=list)
    done:           bool = False
    final_response: Optional[str] = None

SYSTEM_PROMPT = """
You are a student assistant agent that interacts with tools step-by-step.

You have ZERO prior knowledge about the student.
ALL information must come from tool calls.

---

## AVAILABLE TOOLS

1. get_subjects()
2. get_assignments(subject)
3. get_assignment_details(subject)
4. get_grades(subject)
5. get_notes(subject)
6. get_announcements(subject)
7. get_schedule(subject)

---

## STRICT ACTION FORMAT

You MUST respond with ONLY ONE of the following:

1. TOOL CALL:

<tool_call>{"tool": "tool_name", "params": {}}</tool_call>

Rules:

* Use DOUBLE quotes only (valid JSON)
* Always include "tool" and "params"
* params must be a JSON object (even if empty: {})
* NO extra text before or after

---

Once you got all the required information, place your answer inside the  <respond> </respond> tags

* Only use this when you have enough information
* Answer MUST be based only on tool outputs

---

## DECISION POLICY (VERY IMPORTANT)

Follow this reasoning strictly:

1. If subject is unknown → call get_subjects()
2. If subject is not in subjects → respond "student not enrolled"
3. If information is required → call the relevant tool
4. If tool returns empty like for example {'grades': []}:
    return 
   * grades → "No grades published"
   * assignments → "No assignments found"
5. Do NOT repeat the same tool call with same arguments
6. Call only ONE tool at a time
7. Once sufficient data is available → respond

---

## EXAMPLES

User: "Show my RL grades"

Step 1:
<tool_call>{"tool": "get_subjects", "params": {}}</tool_call>

Step 2 (after seeing RL exists):
<tool_call>{"tool": "get_grades", "params": {"subject": "RL"}}</tool_call>

Step 3 (if grades empty): <respond>No grades have been published for RL</respond>

---

## IMPORTANT CONSTRAINTS

* DO NOT output anything outside the defined formats
* DO NOT use single quotes
* DO NOT skip steps
* ALWAYS produce valid JSON

---
"""

class StudentAgentEnvironment:

    def __init__(self, db, gt, query: str, today: date):
        self.db    = db
        self.gt    = gt
        self.query = query
        self.today = today
        self.max_steps = getattr(gt, "max_steps", 10)

        self.state = EpisodeState(
            query  = query,
            today  = today,
            context_window = [
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": query},
            ],
        )

    def step(self, agent_output: str) -> dict:
        step_num = len(self.state.steps) + 1

        if step_num > self.max_steps:
            self.state.done = True
            return {"done": True, "intermediate_reward": -0.3,
                    "info": {"reason": "step_limit_exceeded"}}

        # Soft warning
        if step_num == int(self.max_steps * 0.8):
            self.state.context_window.append({
                "role":    "system",
                "content": f"[{self.max_steps - step_num} steps remaining — wrap up soon]"
            })

        tool_m    = re.search(r'<tool_call>(.*?)</tool_call>', agent_output, re.DOTALL)
        respond_m = re.search(r'<respond>(.*?)</respond>',    agent_output, re.DOTALL)

        if respond_m:
            resp = respond_m.group(1).strip()
            self.state.done           = True
            self.state.final_response = resp
            self.state.steps.append(AgentStep(step_num, None, None, resp, True))
            return {"done": True, "intermediate_reward": 0.0, "info": {"reason": "respond"}}

        if tool_m:
            try:
                call = json.loads(tool_m.group(1).strip())
            except json.JSONDecodeError:
                return self._malformed(step_num, agent_output)
            return self._execute(call, step_num)

        return self._malformed(step_num, agent_output)

    def _execute(self, call: dict, step_num: int) -> dict:
        tool   = call.get("tool", "")
        params = call.get("params", {})
        called = [c["tool"] for c in self.state.tool_calls_made]

        # Must call get_subjects first
        # if tool != UNIVERSAL_FIRST and UNIVERSAL_FIRST not in called:
        #     self.state.context_window.append({
        #         "role": "system",
        #         "content": "[Call get_subjects() first to verify enrollment]"
        #     })
        #     return {"done": False, "intermediate_reward": -0.1,
        #             "info": {"reason": "skipped_get_subjects"}}

        # Duplicate detection — soft penalty to discourage loops without killing episodes
        key  = (tool, json.dumps(params, sort_keys=True))
        seen = {(c["tool"], json.dumps(c.get("params", {}), sort_keys=True))
                for c in self.state.tool_calls_made}
        if key in seen:
            result = {"error": "duplicate call — already called this tool with the same params"}
            self.state.context_window.append(
                {"role": "tool", "name": "system", "content": json.dumps(result)}
            )
            return {"done": False, "intermediate_reward": -0.05,
                    "info": {"reason": "duplicate"}}

        # Execute
        result, shaping = self._run_tool(tool, params)
        logger.info(f"Tool called: {call}, tool response: {result}")
        self.state.tool_calls_made.append({**call, "result": result})
        self.state.steps.append(AgentStep(step_num, call, result, None, False))
        self.state.context_window.append(
            {"role": "tool", "name": tool, "content": json.dumps(result)}
        )
        return {"done": False, "intermediate_reward": shaping,
                "info": {"tool": tool}}

    def _run_tool(self, tool: str, params: dict) -> tuple[dict, float]:
        db = self.db
        try:
            if tool == "get_subjects":
                return {"subjects": db.get_subjects()}, 0.05
            elif tool == "get_assignments":
                s = params.get("subject","")
                if s and s not in db.get_subjects():
                    return {"error": f"{s} not found"}, -0.2
                r = db.get_assignments(s)
                return {"assignments": r}, 0.1
            elif tool == "get_assignment_details":
                s = params.get("subject","")
                return {"details": db.get_assignment_details(s)}, 0.1
            elif tool == "get_grades":
                s = params.get("subject","")
                return {"grades": db.get_grades(s)}, 0.05
            elif tool == "get_notes":
                s = params.get("subject","")
                return {"notes": db.get_notes(s)}, 0.05
            elif tool == "get_allnotes":
                s = params.get("subject","")
                return {"local_notes": db.get_allnotes(s)}, 0.05
            elif tool == "get_schedule":
                s = params.get("subject","")
                return {"schedule": db.get_schedule(s)}, 0.05
            elif tool == "get_announcements":
                s = params.get("subject","")
                return {"announcements": db.get_announcements(s)}, 0.05
            elif tool == "get_todos":
                return {"todos": db.get_todos()}, 0.05
            elif tool == "add_todo":
                r = db.add_todo(
                    params.get("subject",""),
                    params.get("title",""),
                    params.get("deadline",""),
                    params.get("priority","medium"),
                )
                return {"created": r}, 0.2
            elif tool == "add_notes":
                r = db.add_notes(
                    params.get("subject",""),
                    params.get("file_name",""),
                )
                return {"saved": r}, 0.2
            else:
                return {"error": f"unknown tool: {tool}"}, -0.1
        except Exception as e:
            return {"error": str(e)}, -0.1

    def _malformed(self, step_num, resp) -> dict:
        self.state.context_window.append({
            "role": "tool",
            "name": "system",
            "content": json.dumps({
                "error": "malformed_output",
                "message": """Your output was not recognised.Use exactly ONE of these formats:
                            Tool call: <tool_call>{'tool':'name','params':{...}} </tool_call> 
                            Final answer: <respond>your answer here</respond>"""
                
            })
        })
        self.state.steps.append(AgentStep(step_num, None, None, resp, False))
        return {"done": False, "intermediate_reward": -0.05,
                "info": {"reason": "malformed"}}

    def build_trajectory(self) -> dict:
        return {
            "query":          self.query,
            "tool_calls":     self.state.tool_calls_made,
            "final_response": self.state.final_response or "",
            "steps_used":     len(self.state.steps),
            "max_steps":      self.max_steps,
            "today":          self.today,
            "intent":         getattr(self.gt, "intent", "unknown"),
        }
