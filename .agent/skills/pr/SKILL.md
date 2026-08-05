---
name: pr
description: Create or update a pull request following verl project conventions.
user_invocable: true
---

When the user asks to create or update a PR, follow these steps:

### 1. Gather Context

Read the following and understand the current branch's changes compared to main:

- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md)

If a PR already exists for this branch, also read its current title, body, and review comments.

### 2. Compose PR Title and Body

Follow the PR template strictly for both title format and body sections. Only check checklist boxes for steps that have actually been completed.

When updating, ensure the title and body still accurately reflect **all** changes on the branch, not just the latest commit.

Write the body as GitHub-flavored Markdown, not fixed-width plain text. Keep each paragraph and each list item on a single logical line and separate blocks with blank lines. Do not hard-wrap running prose at a column limit: GitHub renders a single newline inside a paragraph or list item as a space, so a manually wrapped sentence turns into unintended mid-sentence breaks in the rendered PR. Watch two easy-to-miss cases: a wrapped line whose continuation starts with `#<number>` (e.g. an issue reference like `#7060`) reads as a heading in the source and is easy to leave split, and list-item continuations that are indented but should still join their item. Let the editor soft-wrap; only insert a newline to start a new paragraph, list item, table row, or fenced code block. Before submitting, re-read the body (or the round-tripped version fetched back from GitHub) and confirm no paragraph or list item is split across multiple source lines.

### 3. Pre-submit Checks

Run pre-commit and fix any issues before creating or pushing.

### 4. Create or Update the PR

- **Create**: target `main` by default unless the user specifies otherwise.
- **Update**: push new commits and update the title and body if the scope has changed. **Read the current PR title and body first** and incorporate any edits the user may have made directly on GitHub — never overwrite with a version generated from scratch.

Return the PR URL when done.
