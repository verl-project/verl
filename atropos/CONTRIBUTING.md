# Contributing to Atropos

First off, thank you for considering contributing to Atropos! It's people like you that make open source projects such great tools.

We welcome any type of contribution, not just code. You can help with:
*   **Reporting a bug**
*   **Discussing the current state of the code**
*   **Submitting a fix**
*   **Proposing new features**
*   **Becoming a maintainer**

## We Develop with GitHub
We use GitHub to host the code, track issues and feature requests, and accept pull requests.

## We Use GitHub Flow
We follow the [GitHub Flow](https://docs.github.com) development workflow. All code changes happen through Pull Requests.

## Getting Started

### Project Setup

1.  **Fork the repository:** Click the "Fork" button on the top right of the [repository page](https://github.com/NousResearch/atropos). This creates your own copy of the project.
2.  **Clone your fork:**
    ```bash
    git clone https://github.com/your-username/atropos.git
    cd atropos
    ```
3.  **Set up the development environment:** This project uses standard Python `venv` for environment creation and `pip` for dependency management.
    ```bash
    # Ensure you have Python 3.10+ installed
    # Create and activate a virtual environment
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

    # Install dependencies, including development dependencies
    pip install -e ".[dev]"
    ```
4.  **Install pre-commit hooks:** This project uses `pre-commit` for code quality checks. The hooks will run automatically when you commit changes.
    ```bash
    pre-commit install
    ```

### Running Tests

We use `pytest` for running tests. To run the test suite:

```bash
pytest
```

Ensure all tests pass before submitting a pull request.

## How to Contribute

### Reporting Bugs

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/NousResearch/atropos/issues) (replace with the actual link if different).

When opening a bug report, please use the **Bug Report** issue template. This template is designed to gather the information we need to efficiently understand and resolve the issue.

**Great Bug Reports** tend to have:

*   A quick summary and/or background.
*   Steps to reproduce the bug:
    *   Be specific!
    *   Provide the exact commands run or a minimal code snippet if possible.
*   What you expected to happen.
*   What actually happened (including any error messages or logs).
*   Your environment details (OS, Python version, relevant package versions).
*   Notes (possibly including why you think this might be happening, or stuff you tried that didn't work).

Thorough bug reports help us address issues faster!

### Suggesting Enhancements

If you have an idea for a new feature or an improvement to an existing one, please open an issue first to discuss it. This allows us to coordinate efforts and ensure the suggestion aligns with the project's goals.

When suggesting an enhancement, please use the **Feature Request** issue template. This helps structure your request and provides context for maintainers and the community to better understand your suggestion.

### Submitting Changes (Pull Requests)

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1.  **Fork the repo** and create your branch from `main`.
    ```bash
    git checkout -b your-feature-or-fix-branch main
    ```
2.  **Make your changes:** Write your code.
3.  **Add tests:** If you've added code that should be tested, add tests.
4.  **Update documentation:** If you've changed APIs or added features, update relevant documentation (README, docstrings, etc.).
5.  **Ensure tests pass:** Run `pytest`.
6.  **Ensure code lints and formats:** The pre-commit hooks will run automatically on commit. You can also run them manually: `pre-commit run --all-files`.
7.  **Commit your changes:** Use clear and descriptive commit messages that explain the purpose of the changes.
    ```bash
    git add .
    git commit -m "Clearly describe the changes made in this commit"
    ```
8.  **Push your branch:**
    ```bash
    git push origin your-feature-or-fix-branch
    ```
9.  **Open a Pull Request (PR):** Go to the original repository on GitHub and open a PR from your fork's branch to the `main` branch.
    *   Provide a clear title and description for your PR.
    *   Link any relevant issues (e.g., "Closes #123").
    *   Explain the changes you've made and why.
    *   **Follow the PR template**: We have two PR templates:
        - For environment-related changes, use the `environment_pull_request_template.md`
        - For all other changes, use the `non_environment_pull_request_template.md`

        Please fill out the appropriate template thoroughly to help reviewers understand your changes.

## Code Style

This project uses standard Python code style (PEP 8) enforced by `black`, `flake8`, and `isort` via `pre-commit`. Please ensure your code adheres to these standards. The pre-commit hooks should help automate formatting and linting.

You can manually run the checks on all files using:
```bash
pre-commit run --all-files
```
This command will automatically fix formatting issues found by `black` and `isort`. However, you may need to manually address any linting errors reported by `flake8`.

## License for Contributions
Any contributions you make will be under the Apache License 2.0. In short, when you submit code changes, your submissions are understood to be under the same [Apache License 2.0](LICENSE) that covers the project. Feel free to contact the maintainers if that\'s a concern.

## Environment Contribution Guidelines

Since Atropos is focused on reinforcement learning environments, we encourage contributions of new training environments. However, please adhere to the following guidelines:

*   **Directory Structure**: Please create your new environment within the `environments/community/` subdirectory. This helps us organize incoming contributions and allows for a streamlined initial review process before full testing and integration.
*   **Import Style**: We prefer that you treat your environment's directory as the package root for imports. For example, if your environment resides in `environments/community/my_new_env/` and you need to import `SomeClass` (assuming it's in a `some_file_in_my_env.py` file at the root of your `my_new_env` directory or accessible via your Python path setup), you should be able to use a direct import like:
    ```python
    from some_file_in_my_env import SomeClass
    ```
    This convention applies to imports within your environment's own files and any helper modules you create alongside it.
* **Legal compliance**: Do not submit environments that involve illegal activities or content.

* **GitHub compliance**: All contributions must comply with [GitHub's Terms of Service and Community Guidelines](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service).

* **Explicit content**: Explicit environments may be considered, but must be:
  * Clearly labeled as such
  * Comply with all legal requirements

* **Game environments**: Game-based environments are welcome, but:
  * Do not submit reverse-engineered commercial game environments that could lead to copyright or intellectual property issues
  * Ensure you have the appropriate rights to any assets used
  * Open-source games or games with permissive licenses are preferred

* **Ethical considerations**: Consider the ethical implications of your environment. Environments that encourage harmful behaviors without educational context may be rejected.

When in doubt about the appropriateness of an environment, please open an issue to discuss it before investing significant development effort.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

Thank you again for your contribution!
