---
name: create-pr
description: Create a pull request for the current branch following verl project conventions.
user_invocable: true
---

When the user asks to create a PR, follow these steps:

### 1. Gather Context

Read `.github/PULL_REQUEST_TEMPLATE.md` and understand the current branch's changes compared to main.

### 2. Compose PR Title and Body

Follow the PR template strictly for both title format and body sections. Only check checklist boxes for steps that have actually been completed.

### 3. Pre-submit Checks

Run pre-commit and fix any issues before creating the PR.

### 5. Create the PR

Target `main` by default unless the user specifies otherwise. Return the PR URL when done.
