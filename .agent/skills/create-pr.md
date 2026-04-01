# create-pr

Create a pull request for the current branch following the verl project's PR conventions.

## Instructions

When the user asks to create a PR, follow these steps:

### 1. Gather Context

Understand the current branch's changes compared to main:
- What files were changed and why
- The full commit history on this branch
- Read `.github/PULL_REQUEST_TEMPLATE.md` for the latest template format

### 2. Determine PR Title

Based on the diff and commits, compose a title strictly following the format defined in `.github/PULL_REQUEST_TEMPLATE.md`:
- Pick the correct `{modules}` and `{type}` from the template's allowed values
- If changes break any existing API, prepend `[BREAKING]`

### 3. Compose PR Body

Fill in each section of `.github/PULL_REQUEST_TEMPLATE.md` based on the actual changes. Only check checklist boxes for steps that have actually been completed.

### 4. Pre-submit Checks

Before creating the PR:
- Run `pre-commit run --all-files --show-diff-on-failure --color=always` and fix any issues
- Verify the branch is pushed to the remote with `git push -u origin <branch>`

### 5. Create the PR

```bash
gh pr create --title "<title>" --body "$(cat <<'EOF'
<body>
EOF
)"
```

Target `main` by default unless the user specifies otherwise.

### 6. Report Back

Return the PR URL to the user.

## Important Notes

- Always analyze ALL commits on the branch, not just the latest one
- If unsure which module(s) to tag, ask the user
- If the user provides a PR description or title, incorporate their input rather than overriding it
