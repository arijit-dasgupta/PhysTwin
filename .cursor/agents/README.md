# Custom subagents (Cursor)

This folder defines **Planner**, **Implementer**, and **Verifier** for multi-phase work. See [Cursor Subagents](https://cursor.com/docs/subagents).

## Files

| File | Role |
|------|------|
| `planner.md` | Questions, iterative plan; writes a **long** plan at `.cursor/plans/` or `docs/plans/` for Cursor Plan mode |
| `implementer.md` | Code, tests, commits, checkpoint/done reports |
| `verifier.md` | Read-only skeptical verification (`readonly: true`) |

## Suggested orchestration (in Agent chat)

1. User pastes a long spec → **`/planner`** (or ask to use the planner subagent).
2. Answer questions until the plan is **READY FOR IMPLEMENTER**.
3. **`/implementer`** — implement phase by phase; read checkpoint reports.
4. **`/verifier`** — validate; if PARTIAL/FAIL, send findings back to **`/implementer`**.
5. Repeat 3–4 until Verifier says **PASS** for the scope.
6. Do a final review as the user (or ask Agent for an Area Chair–style summary).

Invoke explicitly, e.g. `> /verifier confirm the last change meets the plan`, or describe naturally: “Use the verifier subagent on this PR.”

## YAML `description` fields

Frontmatter `description` text is what Cursor uses to decide **when to delegate** to each subagent. These are tuned to be trigger-rich (`Always use…`, `Use proactively…`, artifact paths, when **not** to use the planner).

## Version control

Commit `.cursor/agents/` so the team shares the same agents.
