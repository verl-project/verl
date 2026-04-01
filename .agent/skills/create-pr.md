---
name: create-pr
description: Create a pull request for the current branch following verl project conventions.
user_invocable: true
---

When the user asks to create a PR, follow these steps:

### 1. Gather Context

Read `.github/PULL_REQUEST_TEMPLATE.md` and understand the current branch's changes compared to main.

### 2. Determine PR Title

Compose a title strictly following the format defined in the PR template:

- Pick the correct `{modules}` and `{type}` from the template's allowed values
- If changes break any existing API, prepend `[BREAKING]`

### 3. Compose PR Body

Fill in each section of the PR template based on the actual changes. Only check checklist boxes for steps that have actually been completed.

### 4. Pre-submit Checks

Run pre-commit and fix any issues before creating the PR.

### 5. Create the PR

Target `main` by default unless the user specifies otherwise. Return the PR URL when done.
