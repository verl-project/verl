---
name: issue
description: Create or update a GitHub issue following verl project conventions.
user_invocable: true
---

When the user asks to create or update an issue, follow these steps:

### 1. Gather Context

Read the following to understand available issue types and their required fields:

- [`bug-report.yml`](.github/ISSUE_TEMPLATE/bug-report.yml)
- [`feature-request.yml`](.github/ISSUE_TEMPLATE/feature-request.yml)

If updating an existing issue, read its current title, body, labels, and comments first.

### 2. Determine Issue Type

Based on the user's description, select the appropriate template:

- **Bug report** ([`bug-report.yml`](.github/ISSUE_TEMPLATE/bug-report.yml)) — something is broken or behaves unexpectedly
- **Feature request** ([`feature-request.yml`](.github/ISSUE_TEMPLATE/feature-request.yml)) — a new capability or enhancement
- **Blank issue** — if neither template fits

### 3. Compose the Issue

Fill in the template fields based on information from the user and the codebase. For bug reports, run `python scripts/diagnose.py` to gather system info if possible.

When updating, ensure the title and body still accurately reflect the current state of the issue.

### 4. Check for Duplicates

Search for existing issues before creating:

```
gh issue list --repo verl-project/verl --state open --search "<keywords>"
```

If a duplicate exists, inform the user instead of creating a new one.

### 5. Create or Update the Issue

- **Create**: add `good first issue` and/or `call for contribution` labels if the issue is straightforward and suitable for new contributors.
- **Update**: update title, body, and labels as needed.

Return the issue URL when done.
