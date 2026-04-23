# CLAUDE.md

## CRITICAL: Ask Questions, Don't Assume

**When implementing anything, ask the user about ambiguous decisions instead of making silent assumptions.** The user cannot predict what choices you'll face during implementation. A quick question is always cheaper than redoing work.

Examples of things to ask about: data format choices, model selection, parameter values, architectural tradeoffs, naming conventions, edge case handling. Err heavily on the side of asking.

## CRITICAL: Test-Driven Development

**ALWAYS write tests BEFORE implementing any new feature or algorithm. DO NOT ASK — JUST DO IT.**

1. Write test cases that define the expected behavior
2. Run the tests to confirm they fail (red)
3. Implement the feature
4. Run the tests to confirm they pass (green)
5. Refactor if needed

## Spec-Driven Development for Non-Trivial Tasks

**For any task that touches multiple files, involves architectural decisions, or has non-obvious requirements, use the spec-driven workflow. Do NOT ask — recognize when it's needed and initiate it.**

**When to trigger:** Multi-file changes, new features, refactors, anything where "just start coding" would risk wasted effort or misalignment.

**When NOT to trigger:** Single-file bug fixes, typos, config tweaks, simple one-function additions with clear requirements.

### Workflow

1. **Announce** that this task warrants a spec. Briefly explain why.
2. **Explore** the codebase to understand current state (use Glob/Grep/Read).
3. **Draft the spec** to `.claude/specs/{descriptive-name}.md` using the template below.
4. **Drill acceptance criteria.** Present each criterion to the user and ask:
    - *"Is this the right bar? Too strict? Too loose?"*
    - *"What edge cases am I missing?"*
    - *"How will you verify this is done?"*
        Do NOT finalize the spec until the user has explicitly approved the acceptance criteria. This is the most important step — push back if criteria are vague.
5. **Delegate** to a fresh subagent via `Task()` with the spec file path. The subagent prompt should be: *"Read the spec at `.claude/specs/{name}.md` and implement it. Follow all rules in CLAUDE.md."* — nothing else. Clean context.
6. **Verify** the subagent's work: run tests, linter, type checker. If it fails, either fix directly or re-delegate with a narrower spec.
7. **Commit** after successful verification.

### Spec Template

```markdown
# Spec: {title}

## Goal
One sentence: what does this accomplish and why.

## Context
- Relevant files and their roles
- Current behavior (what exists today)
- How this fits into the broader system

## Requirements
Numbered list of concrete changes. Each requirement should be independently verifiable.

## Constraints
- What must NOT change
- Performance/memory bounds
- Compatibility requirements
- Edge cases to handle

## Test Plan
- What tests to write (file paths)
- What to mock
- Key assertions

## Acceptance Criteria
Checkboxed list. Each item must be:
- **Observable** — can be verified by running a command or reading output
- **Specific** — no ambiguity about pass/fail
- **Complete** — if all boxes are checked, the task is done
```

## Self-Verification

**IMPORTANT: Always verify your own work.** After making changes:

1. Run relevant tests
2. Run linter / type checker if configured in the project
3. If tests or type checks fail, fix the root cause — do not suppress errors

**NEVER dismiss ANY failure as "pre-existing."** If you encounter a test failure, type error, lint error, or any other issue — even in files you didn't touch this session — fix it immediately. Do not work around it, note it, or defer it.

**NEVER skip or `--ignore` failing tests.** When a test fails, diagnose and fix the root cause — even if the failure predates your changes. Skipping a failing test is the same as hiding a bug. If the test is genuinely obsolete, delete it and explain why. If it tests behavior that changed, update the test to match the new behavior. The full test suite must pass clean before you declare your work done.

## Self-Improvement: Updating CLAUDE.md

**When you encounter a repeated error, a surprising project convention, or a non-obvious gotcha that could have been avoided with better instructions in this file, suggest adding a rule to CLAUDE.md.** Examples:

- An import pattern that keeps tripping you up
- A framework/library convention that isn't obvious from the code
- A test pattern that requires specific mocking approaches
- A dependency quirk or environment setup issue

Phrase suggestions as: *"I hit \[problem\]. Should I add a rule to CLAUDE.md to prevent this in the future?"*

This keeps the file evolving as a living document that captures hard-won knowledge.

## Committing Code

Use `/commit` to commit changes. This runs pre-commit checks, a code review, and creates the commit automatically.

**NEVER skip the `/commit` skill.** Do not manually `git commit` to bypass the code review step — even for "obvious" or "small" fixes. Every commit must pass the review subagent. No exceptions.

## Code Style

- **Avoid inline comments.** Code should be self-explanatory through clear naming and structure.
- Docstrings for public functions/classes are fine (documentation, not comments).
- Only add inline comments when the logic is truly non-obvious.
- **NEVER write defensive code.** No `or ""`, no `or []`, no `if x is not None` when `x` cannot be `None`, no try/except around code that cannot fail. Trust the types. If a value can legitimately be `None`, fix the upstream code or type signature — don't paper over it at the call site.
- Use type annotations on public APIs (Python: parameters + return types; TypeScript: exported functions/components).

## CRITICAL: Never Stack Long-Running Processes

**One long-running command at a time, verified to exit before starting another.** This rule exists because violating it once froze the user's machine and forced a reboot.

When a command might take more than ~30s (jest, `npm run build`, pytest with many workers, docker builds):

1. Use `run_in_background: true` with an explicit `> /tmp/<name>.log 2>&1` redirect. Do NOT pipe to `tail` — pipe buffering hides whether the process is making progress.
2. Before starting another run of the same tool, **verify the previous one exited** (check the log file for a final summary line, or `ps` for the process). If you cannot confirm exit, do not start another.
3. When a foreground command auto-backgrounds (because it exceeded the foreground timeout), **STOP**. The underlying process is still running. Do not retry with a different invocation — that just stacks processes.
4. Cap parallelism for heavy test runners.
5. Kill idle dev servers before running the full test suite if the machine feels slow.

## Periodic Self-Reflection

**After completing a significant task (shipping a feature, debugging a hard issue, finishing a migration), pause and ask:**

1. Did anything fail that could have been prevented with a rule in this file?
2. Did I repeat any mistake more than once?
3. Is there a non-obvious project convention I discovered that future sessions should know?

If yes to any, update this file immediately. Do not wait for the user to ask.

---

# Project notes

The agent launcher lives in `cli/reva/`. All agent definitions (roles, personas, research interests, harness, global rules) live in `agent_definition/`.
