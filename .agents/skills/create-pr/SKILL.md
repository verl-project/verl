---
name: create-pr
description: Rebase onto upstream/main, squash local commits, generate a Conventional Commit message, and create or update the GitHub pull request to verl-project/verl.
---

# Create Pull Request

Use this skill when the user asks to create or update a PR for the current branch.

## Inputs

- Optional `--draft`
- Optional `--base <branch>` (default: `main`)

## Preconditions

1. Verify the current branch is not `main` or `master`.
2. Check for uncommitted changes with `git status --short`.
3. Ensure `gh` is available (`gh auth status`).
4. If uncommitted changes exist, stop and ask the user to commit or stash them first.

## Workflow

### Step 1: Check for an Existing PR

```bash
gh pr view --json number,title,url,state,isDraft
```

If a PR already exists, tell the user before rewriting history or force-pushing.

### Step 2: Fetch and Rebase onto Upstream

veRL PRs target `verl-project/verl`, not a personal fork's main:

```bash
git fetch upstream main
git rebase upstream/main
```

- If rebase conflicts occur, abort and stop:
  ```bash
  git rebase --abort
  ```
- Tell the user which files conflicted and ask them to resolve manually.

### Step 3: Squash into One Commit

```bash
git reset --soft upstream/main
```

Load `commit-conventions` skill before generating the commit message.

**veRL PR title format** (enforced by CI `check-pr-title.yml`):

```
[{modules}] {type}: {description}
```

- `{modules}`: e.g. `fsdp`, `megatron`, `vllm`, `sglang`, `trainer`, `algo`, `reward`,
  `data`, `recipe`, `doc`, `ci` — separate multiple with `,`
- `{type}`: `feat`, `fix`, `refactor`, `chore`, `test`
- If any API breaks (CLI args, config keys, function signatures): prepend `[BREAKING]`
- Example: `[BREAKING][fsdp, megatron] feat: dynamic batching`

Keep the subject imperative and under ~72 characters.

### Step 4: Generate PR Body

Follow `.github/PULL_REQUEST_TEMPLATE.md`. Fill in:

- **What does this PR do?** — concise overview, link related issues/PRs
- **Test** — experiment results or CI test commands; if no CI coverage, explain why
- **API and Usage Example** — code snippet if API changes
- **Design & Code Changes** — high-level design for complex PRs

**Checklist to verify before submitting:**

- [ ] `pre-commit run --all-files` passes
- [ ] Docs updated if behavior changes (`docs/`)
- [ ] Unit or e2e test added (or explanation why not)
- [ ] If `recipe/` submodule changed: `git submodule update --remote`

### Step 5: Push and Create or Update PR

```bash
# First push (or after rebase that rewrites history)
git push origin <branch>          # or --force-with-lease if rebased

# Create PR targeting upstream
gh pr create \
  --repo verl-project/verl \
  --head <your-fork>:<branch> \
  --base main \
  --title "[modules] type: description" \
  --body "$(cat <<'EOF'
### What does this PR do?
...

### Test
...

### Checklist Before Submitting
- [ ] pre-commit passes
- [ ] Docs updated
- [ ] Tests added
EOF
)"
```

To update an existing PR:
```bash
gh pr edit <number> --title "..." --body "..."
```

## Guardrails

- Never create a PR from `main` or `master`
- Never silently force-push over an existing PR branch — confirm first
- Never bypass `commit-conventions` for the squashed commit
- PR title must match `[{modules}] {type}: {description}` — CI will reject otherwise
- Verify `git diff upstream/main` only contains intended changes before pushing

## Output

Report:

- Base branch used (`upstream/main`)
- Final squashed commit message
- PR title (must match veRL format)
- PR URL if creation/update succeeded
- Any steps skipped or requiring user follow-up

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================
Location: .agents/skills/create-pr/SKILL.md

## How to Update
- When PR template changes: update Step 4 body template
- When CI title format changes: update Step 3 format rules
================================================================================
-->
