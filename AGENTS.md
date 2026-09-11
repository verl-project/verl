# Agent Instructions for verl

> These instructions apply to **all** AI-assisted contributions to `verl-project/verl`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo verl-project/verl --comments
gh pr list --repo verl-project/verl --state open --search "<issue_number> in:body"
gh pr list --repo verl-project/verl --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
  - Why this is not duplicating an existing PR.
  - Test commands run and results.
  - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

uv pip install pre-commit hydra-core
pre-commit install
```

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`). For example:

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

### Resolving agent reviews

Review comments from agent bots (e.g., gemini-code-assist) can be outdated or wrong. Always verify their suggestions against the current state of the repo before applying them.

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.

## Acknowledgements

Adapted from the [vLLM project](https://github.com/vllm-project/vllm)'s [`AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/AGENTS.md).

## Fork Notes (MindLab-Research)

- `main` is the working trunk and syncs from `volcengine/verl` `main` by merge.
- The `verl-mint/*` and `mint/*` refs are separate lines kept for specific
  deployments. Work moves between lines by porting (cherry-pick or a 3-way apply
  of a diff), never by merging the lines into each other.
- Compatibility notes for newer megatron-core: the NVIDIA `dev` line drops
  `get_default_save_sharded_strategy` (construct the save strategy directly), and
  newer cores expose the FSDP wrappers as factory functions, so
  `ALL_MODULE_WRAPPER_CLASSNAMES` must keep only real classes or `unwrap_model`
  raises `TypeError` from `isinstance`.
