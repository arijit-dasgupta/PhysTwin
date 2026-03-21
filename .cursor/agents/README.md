# Custom subagents (Cursor)

Docs: [Subagents](https://cursor.com/docs/subagents).

## Deploy my minions — **Task only** + **mandatory autoloop**

1. **Planner / Implementer / Verifier** = **`Task(planner)`**, **`Task(implementer)`**, **`Task(verifier)`** only. **Parent chat must not implement** (no edits in orchestrator for minions work).
2. **After `Task(verifier)`:** if **PARTIAL / FAIL / actionable BLOCKED** → **next action MUST be `Task(implementer)`** in the **same flow**, **without** the user typing “continue.” If **PASS** → `[AREA CHAIR]` only (no Implementer).
3. After **`Task(implementer)`** → **immediately** **`Task(verifier)`** again until PASS or cap.

Full rule: **`.cursor/rules/deploy-minions-orchestration.mdc`**

## Files

| File | Role |
|------|------|
| `planner.md` | `Task(planner)` |
| `implementer.md` | `Task(implementer)` — **only** subagent that edits code for minions |
| `verifier.md` | `Task(verifier)`, readonly |

## Manual fallback

`/planner` → `/implementer` → `/verifier` (same order + autoloop discipline).

## Version control

Commit `.cursor/agents/` and `.cursor/rules/`.
