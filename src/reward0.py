def reward( traj, gt: GroundTruth) -> float:
    score        = 0.0
    called_tools = [c["tool"] for c in traj["tool_calls"]]

    # ── 1. get_subjects must be first ─────────────────────────────
    if called_tools and called_tools[0] != "get_subjects":
        score -= 0.4   # started without discovery

    # ── 2. Check ordering — tools must follow the canonical chain ──
    chain       = gt.required_tools
    chain_index = 0
    for called in called_tools:
        # Advance pointer through chain as tools appear in order
        while chain_index < len(chain) and chain[chain_index] != called:
            chain_index += 1
        if chain_index < len(chain):
            score += 0.1   # reward each tool appearing in correct order
            chain_index += 1

    # ── 3. Early exit: was it correct for the data state? ─────────
    answer_type = gt.answer_type
    if answer_type in gt.exit_paths:
        exit_def = gt.exit_paths[answer_type]
        # Reward for exiting at the right step with the right tools
        if set(called_tools) == set(exit_def["required_tools"]):
            score += 0.5   # clean early exit
        # Penalise continuing past a valid exit point
        if len(called_tools) > len(exit_def["required_tools"]):
            score -= 0.2   # kept going after should have stopped

    # ── 4. Dynamic step budget ─────────────────────────────────────
    if traj["steps_used"] <= gt.optimal_steps:
        score += 0.2   # efficiency bonus
    elif traj["steps_used"] > gt.max_steps:
        score -= 0.5   # exceeded budget

    # ── 5. N-call intents (due_this_week, schedule_all_today) ──────
    if gt.intent in ("due_this_week", "schedule_all_today", "study_plan_all"):
        # These require one call per subject — count calls of the repeated tool
        N          = len(gt.oracle_data.get("all_subjects", []))
        multi_tool = ("get_assignments" if gt.intent == "due_this_week"
                      else "get_schedule")
        actual_n   = sum(1 for t in called_tools if t == multi_tool)
        if actual_n == N:
            score += 0.3   # called exactly N times
        elif actual_n < N:
            score -= 0.1 * (N - actual_n)   # missed some subjects

    return max(-1.0, min(1.0, score))