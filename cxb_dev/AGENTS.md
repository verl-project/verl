# AGENTS.md

This file defines agent execution practices for development work on the `verl` repository.
## 0. Language
Use **Chinese** as the default language to communicate with the user write docs in. Althouh it's okay to use English writing docs that are primarily read by an AI agent (like an implementation plan).

## 1. Purpose

Use this file for stable workflow rules only.

Do not put milestone progress, temporary conclusions, or one-off debugging notes here.

## 2. Document Boundaries

| Document | Role |
|----------|------|
| `cxb_dev/AGENTS.md` | Agent behavior rules, orchestration rules, review gates, escalation rules |
| `cxb_dev/docs/project-context.md` | Current phase status, frozen/open decisions, milestone progress |
| `cxb_dev/docs/plans/*.md` | Design documents and implementation plans for this work |
| `cxb_dev/docs/retros/*.md` | Evidence-based retrospectives and decision traces |
| `cxb_dev/docs/ref/*.md` | External/local references with necessity notes |

These paths are under `cxb_dev/docs/` to avoid polluting VERL's own `docs/` directory with private development notes. VERL's existing `docs/` and `recipe/` directories follow their own conventions — do not reorganize them.

If content does not fit this boundary, move it to the correct document.

## 3. Upstream Codebase Rules

These rules apply specifically because we are working in an upstream open-source project.

1. **Follow VERL's existing code style** — match the patterns in `verl/experimental/agent_loop/agent_loop.py`. Deviating from existing conventions without maintainer discussion creates merge friction.
2. **Reuse before reimplementing** — if VERL already provides a utility (tokenization, padding, load balancing, Ray actor patterns), use it. Do not write a parallel implementation.
3. **Do not break existing interfaces** — `AgentLoopBase`, `AgentLoopWorker`, `AgentLoopManager`, and `DataProto` have callers. Any change to their public interface must be backward-compatible or explicitly coordinated with the maintainer.
4. **Keep implementation scope minimal** — implement what the RFC specifies. Do not add capabilities, refactor surrounding code, or "improve" things that were not part of the design discussion.
5. **New modules go under `verl/experimental/`** — until the maintainer decides to promote them.

## 4. Subagent Orchestration Rules

1. Use focused subagents with one clear responsibility per task.
2. Do not nest subagent spawning.
3. Do not run multiple implementer subagents on overlapping file ownership.
4. Use explicit ownership in prompts (files, scope, constraints).
5. Run two-stage review for meaningful changes:
   - spec-compliance review
   - code-quality/risk review
6. Trust no success claim without verification evidence (tests/log output + commit/diff).
7. Add necessary explanatory comments for non-obvious logic (contracts, invariants, failure paths), and keep comments concise and non-redundant.

### Meaningful Changes Threshold

Treat a change as meaningful when it impacts behavior or contract, not only file count.

Meaningful if any condition is true:

1. Crosses module/package boundaries and changes runtime behavior.
2. Changes public interface, config contract, or reason-code vocabulary.
3. Alters failure handling, retry, validation, or security-relevant paths.
4. Changes multiple production files with user-visible or training-visible effect.
5. Introduces non-trivial design tradeoffs that need explicit review evidence.

Not meaningful by default:

1. Docs-only or comments-only updates.
2. Formatting or mechanical refactor with no behavioral change.
3. Pure renaming/move where tests and behavior remain unchanged.

## 5. PR and Community Interaction Rules

1. **Before opening a PR**, ensure the implementation matches the RFC as discussed with the maintainer. Do not introduce unreviewed scope.
2. **PR description must include**: what is implemented, what is not (deferred to future work), how to test it, and any open questions.
3. **Coordinate on infrastructure extraction** — Xibin Wu indicated he may PR the extraction of LLM server initialization from `AgentLoopManager`. Check issue #5790 status before implementing that boundary to avoid conflicts.
4. **Breaking changes require explicit discussion** in the issue/PR before implementation, not after.
5. **Keep PRs focused** — if implementation naturally splits into independent pieces (e.g., Gateway core vs. Framework base class), prefer separate PRs over one large PR.

## 6. Git Workflow Rules

1. Work on a feature branch; do not commit directly to `main`.
2. Keep commits atomic (one logical change per commit).
3. Write commit messages that explain *why*, not just *what*.
4. Sync with upstream `main` before submitting a PR.
5. Do not force-push shared branches.
6. Never commit secrets, large local artifacts, or output files.

## 7. Rule Promotion Policy (Retro → AGENTS)

A retrospective insight can be promoted to `AGENTS.md` only if all are true:

1. Repeatable across tasks (not one-off).
2. Actionable and testable.
3. Reduces defects, rework, or context pollution.
4. Expressible as a short rule without embedding project timeline details.

Governance frequency: run a promotion sweep once per milestone, not per single retrospective doc.

## 8. Project Context Update Policy

1. Update `cxb_dev/docs/project-context.md` at milestone checkpoints.
2. Keep updates factual and state-oriented (no narrative postmortems there).
3. Add links to newly canonical docs (plans/retros/ref) in the reference map.

## 9. Entry Document Hygiene Rule

Applies to entry documents that bootstrap work or phase understanding, including:

1. `cxb_dev/docs/project-context.md`
2. Handoff prompts in `cxb_dev/docs/`
3. Phase design reset docs / pre-plans
4. Other documents whose primary role is to orient the next agent quickly

When such a document is updated after a phase shift, decision reversal, or handoff rewrite, review every retained section against three checks:

1. **实时性**: does it still describe the current phase rather than the previous one?
2. **有效性**: does it still help the next decision or execution step?
3. **间接性**: should this document summarize/link the information instead of embedding full detail?

Required cleanup rule:

1. Delete or move stale conclusions, obsolete TODOs, historical validation detail, and duplicated indexes when they no longer serve the document's entry role.
2. Do not preserve superseded context "just in case" inside entry documents; move detail to `cxb_dev/docs/retros/` or `cxb_dev/docs/plans/`.
3. If an entry document was materially simplified or retargeted, re-check neighboring entry docs for now-invalid residue.

## 10. Execution Handoff Rule

Before starting a meaningful task, decide which execution venue best fits the work:

1. Complete in the current window directly.
2. Complete in the current window with subagents.
3. Complete in a separate window/worktree.

Decision factors:

1. Current context coverage and whether loading more context here would create confusion.
2. Repository scope and whether the task crosses repo/worktree boundaries.
3. Need for isolation, long-running execution, or a distinct review/verification loop.
4. Whether the task is implementation-heavy enough to benefit from a fresh execution session.

If a separate window/worktree is the better fit, the current agent must prepare the handoff before asking the user to switch:

1. Summarize the goal, current state, constraints, and success criteria.
2. Identify exact repos/workdirs, files, and reference docs to read first.
3. State the preferred workflow (same-window subagents vs new window) and why.
4. Provide a ready-to-use handoff prompt when the task is non-trivial or likely to be reused.

Do not offload work with only a vague suggestion to "open a new window". The originating agent is responsible for making the transition concrete.

## 11. Objectivity Rule

When answering questions, evaluating trade-offs, or proposing designs, provide objective assessments based on technical merit. Do not align with the user's apparent preference or prior decisions unless the evidence supports them. If the user's inclination has technical downsides, state them clearly. Agreeing for the sake of agreement wastes time and produces worse designs.

## 12. Conflict Resolution Order

When instructions conflict, resolve by scope and specificity:

1. Current task direct instructions from user/developer/system.
2. Canonical frozen design docs for this phase (RFC + maintainer-confirmed decisions).
3. `AGENTS.md` workflow rules.
4. Prompt templates and retrospective notes.
