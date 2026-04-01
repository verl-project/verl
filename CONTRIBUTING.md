# Contributing to verl

Thank you for considering a contribution to verl! We welcome contributions of any kind - bug fixes, enhancements, documentation improvements, or even just feedback. Whether you're an experienced developer or this is your first open-source project, your help is invaluable.

Your support can take many forms:

- Report issues or unexpected behaviors.
- Suggest or implement new features.
- Improve or expand documentation.
- Review pull requests and assist other contributors.
- Spread the word: share verl in blog posts, social media, or give the repo a ⭐.

## Finding Issues to Contribute

Looking for ways to dive in? Check out these issues:

- [Good first issues](https://github.com/volcengine/verl/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)
- [Call for contribution](https://github.com/volcengine/verl/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22call%20for%20contribution%22)
  Furthermore, you can learn the development plan and roadmap via [RFC](https://github.com/volcengine/verl/issues?q=is%3Aissue%20state%3Aopen%20label%3ARFC) and [Roadmap](https://github.com/volcengine/verl/issues?q=state%3Aopen%20label%3A%22roadmap%22).

## Developing

- **Python-only**: install verl via `pip install -e .[test,vllm]` or `pip install -e .[test,sglang]` and iterate quickly. For full dependency setup, check out the verl [installation doc](https://verl.readthedocs.io/en/latest/start/install.html).

## Code Linting and Formatting

We rely on pre-commit to keep our code consistent. To set it up:

```bash
pip install pre-commit hydra-core
pre-commit install
# for staged changes
pre-commit run
# for all files in the repo
pre-commit run --all-files
# run a specific hook with pre-commit
# pre-commit run --all-files --show-diff-on-failure --color=always <hook-id>
pre-commit run --all-files --show-diff-on-failure --color=always ruff
pre-commit run --all-files --show-diff-on-failure --color=always autogen-trainer-cfg
```

## Testing

Our test suites run on GitHub Actions. Check these workflows for details:

- [GPU unit tests](https://github.com/volcengine/verl/blob/main/.github/workflows/gpu_unit_tests.yml)
- [CPU unit tests](https://github.com/volcengine/verl/blob/main/.github/workflows/cpu_unit_tests.yml)
- [vLLM tests](https://github.com/volcengine/verl/blob/main/.github/workflows/vllm.yml)
- [SGLang tests](https://github.com/volcengine/verl/blob/main/.github/workflows/sgl.yml)

### Adding CI tests

If possible, please add CI test(s) for your new feature:

1. Find the most relevant workflow yml file, which usually corresponds to a `hydra` default config (e.g. `ppo_trainer`, `ppo_megatron_trainer`, `sft_trainer`, etc).
2. Add related path patterns to the `paths` section if not already included.
3. Minimize the workload of the test script(s) (see existing scripts for examples).

## Building the Docs

```
# Ensure verl is on your PYTHONPATH, e.g.:
pip install -e .[test]

# Install documentation dependencies
cd docs
pip install -r requirements-docs.txt

# Generate HTML docs
make clean
make html

# Preview locally
python -m http.server -d _build/html/
```

Open your browser at http://localhost:8000 to explore the docs.

## Pull Requests & Code Reviews

Thanks for submitting a PR! To streamline reviews:

- Follow our Pull Request Template for title format and checklist.
- Adhere to our pre-commit lint rules and ensure all checks pass.
- Update docs for any user-facing changes.
- Add or update tests in the CI workflows, or explain why tests aren't applicable.

## AI-Assisted Contributions

See

- [`AGENTS.md`](AGENTS.md) for rules that all AI coding agents must follow
- [`editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md) for guidelines on editing agent instructions.

## License

See the [LICENSE](https://github.com/volcengine/verl/blob/main/LICENSE) file for full details.

## Thank You

We appreciate your contributions to verl. Your efforts help make the project stronger and more user-friendly. Happy coding!
