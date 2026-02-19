---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

## Describe the Issue

<!-- A clear and concise description of what the bug or issue is. -->

## Environment/API Details

- **Environment Class/Name:** [e.g., `atroposlib.envs.MyCustomEnv`, `gsm8k_server.py`]
- **Environment Configuration (`BaseEnvConfig` or subclass):** (Optional)
  ```yaml
  # Paste relevant config values here
  group_size: 4
  max_token_length: 2048
  # ... etc.
  ```
- **API Endpoint/Method Involved:** (If applicable) [e.g., `/register_env`, `/get_status`, `env.collect_trajectory()`]

## Steps to Reproduce

<!-- Detailed steps to reproduce the behavior: -->
1. Initialize environment `...` with config `...`
2. Call method `get_next_item()` and receive `Item` `...`
3. Call method `collect_trajectory()` or `collect_trajectories()` with `Item` `...`
4. Observe issue `...` (e.g., incorrect `ScoredDataGroup`, error during API call)

## Interaction Details (if applicable)

<!-- Provide details about the specific interaction step where the issue occurs. -->
- **Input `Item` to `collect_trajectory`:**
  ```python
  # Paste relevant Item details here
  ```
- **Output `ScoredDataGroup` (or error):**
  ```python
  # Paste relevant ScoredDataGroup or traceback here
  ```
- **Expected `ScoredDataGroup` / Behavior:**

## Setup Details

- **OS:** [e.g. macOS, Windows, Linux]
- **Python Version:** [e.g. 3.10, 3.11]
- **`Atropos` Version:** [e.g. output of `pip show atropos` or commit hash]
- **Relevant Libraries/Versions:** [e.g., `pydantic==2.5.0`, `aiohttp==3.9.0`, `transformers==4.35.0`]

## Additional Context & Logs

<!-- Add any other context about the problem here. Include relevant logs or screenshots. -->

```log
# Paste relevant logs here
```
