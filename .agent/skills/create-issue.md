---
name: create-issue
description: Create a GitHub issue following verl project conventions.
user_invocable: true
---

When the user asks to create an issue, follow these steps:

### 1. Gather Context

Read the following to understand available issue types and their required fields:

- [`bug-report.yml`](.github/ISSUE_TEMPLATE/bug-report.yml)
- [`feature-request.yml`](.github/ISSUE_TEMPLATE/feature-request.yml)
- [`config.yml`](.github/ISSUE_TEMPLATE/config.yml)

### 2. Determine Issue Type

Based on the user's description, select the appropriate template:

- **Bug report** ([`bug-report.yml`](.github/ISSUE_TEMPLATE/bug-report.yml)) — something is broken or behaves unexpectedly
- **Feature request** ([`feature-request.yml`](.github/ISSUE_TEMPLATE/feature-request.yml)) — a new capability or enhancement
- **Blank issue** — if neither template fits

### 3. Compose the Issue

Fill in the template fields based on information from the user and the codebase. For bug reports, run `python scripts/diagnose.py` to gather system info if possible.

### 4. Check for Duplicates

Search for existing issues before creating:

```
gh issue list --repo verl-project/verl --state open --search "<keywords>"
```

If a duplicate exists, inform the user instead of creating a new one.

### 5. Create the Issue

Add `good first issue` and/or `call for contribution` labels if the issue is straightforward and suitable for new contributors.
Create the issue and return the URL when done.
