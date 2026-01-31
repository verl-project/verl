### What does this PR do?

> Add **concise** overview of what this PR aims to achieve or accomplish. Reference related GitHub issues and PRs that help with the review.

### Checklist Before Starting

- [ ] Search for similar PRs. Paste at least one query link here: ...
- [ ] Format the PR title as `[{modules}] {type}: {description}` (This will be checked by the CI)
  - `{modules}` include `fsdp`, `megatron`, `veomni`, `sglang`, `vllm`, `rollout`, `trainer`, `ci`, `training_utils`, `recipe`, `hardware`, `deployment`, `ray`, `worker`, `single_controller`, `misc`, `perf`, `model`, `algo`, `env`, `tool`, `ckpt`, `doc`, `data`, `cfg`, `reward`
  - If this PR involves multiple modules, separate them with `,` like `[megatron, fsdp, doc]`
  - `{type}` is in `feat`, `fix`, `refactor`, `chore`, `test`
  - If this PR breaks any API (CLI arguments, config, function signature, etc.), add `[BREAKING]` to the beginning of the title.
  - Example: `[BREAKING][fsdp, megatron] feat: dynamic batching`

### Test

> For changes that can not be tested by CI (e.g., algorithm implementation, new model support), validate by experiment(s) and show results like training curve plots, evaluation results, etc.

### API and Usage Example

> Demonstrate how the API changes if any, and provide usage example(s) if possible.

```python
# Add code snippet or script demonstrating how to use this
```

### Design & Code Changes

> Demonstrate the high-level design if this PR is complex, and list the specific changes.

### Checklist Before Submitting

> [!IMPORTANT]
> Please check all the following items before requesting a review, otherwise the reviewer might deprioritize this PR for review.

- [ ] Read the [Contribute Guide](https://github.com/volcengine/verl/blob/main/CONTRIBUTING.md).
- [ ] Apply [pre-commit checks](https://github.com/volcengine/verl/blob/main/CONTRIBUTING.md#code-linting-and-formatting): `pre-commit install && pre-commit run --all-files --show-diff-on-failure --color=always`
- [ ] Add / Update [the documentation](https://github.com/volcengine/verl/tree/main/docs).
- [ ] Add unit or end-to-end test(s) to [the CI workflow](https://github.com/volcengine/verl/tree/main/.github/workflows) to cover all the code. If not feasible, explain why: ...
- [ ] Once your PR is ready for CI, send a message in [the `ci-request` channel](https://verl-project.slack.com/archives/C091TCESWB1) in [the `verl` Slack workspace](https://join.slack.com/t/verl-project/shared_invite/zt-3855yhg8g-CTkqXu~hKojPCmo7k_yXTQ). (If not accessible, please try [the Feishu group (飞书群)](https://applink.larkoffice.com/client/chat/chatter/add_by_link?link_token=772jd4f1-cd91-441e-a820-498c6614126a).)
- [ ] If your PR is related to the `recipe` submodule, please also update the reference to the submodule commit via `git submodule update --remote` or `cd recipe && git pull origin main`.
