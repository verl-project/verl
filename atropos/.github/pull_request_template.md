<!--
╭───────────────────────────────────────────────────────────╮
│  ✨  ATROPOS PULL REQUEST TEMPLATE  ✨                    │
│  Select PR type below and fill applicable sections.       │
│  Delete non-applicable sections for your PR type.         │
╰───────────────────────────────────────────────────────────╯
-->

## PR Type
<!-- Please check ONE of the following options -->
- [ ] RL Environment PR - Complete Environment Snapshot & Zero-Training sections
- [ ] Non-Environment PR - Complete Description, Related Issues & Type of Change sections

---

## 📝 General Information
### Description
<!-- Briefly describe the changes or additions introduced by this pull request. -->

<!-- For non-environment PRs -->
### Related Issues
<!-- Link any relevant issues here. Use "Closes #issue_number" to automatically close issues. -->

### Type of Change
<!-- For non-environment PRs - delete options that are not relevant. -->
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update
- [ ] Code refactor (no functional changes)
- [ ] Build/CI/CD related changes
- [ ] Other (please describe):

---

## 🔖 Environment Snapshot
<!-- For RL Environment PRs only -->
| Field | Your Entry |
|-------|------------|
| **Environment Name** | <!-- e.g. "SudokuVerifier-v0" --> |
| **Short Description** | <!-- One-sentence purpose/goal. --> |
| **Category** | <!-- Select: Verifiable-Reasoning / RLAIF / RLHF / Other  --> |
| **Dataset Needed?** | <!-- No / Yes (link & license) --> |
| **External Deps** | <!-- Extra pip packages, system libs, etc. --> |
| **Environmental Variables** | <!-- variable name(s) --> |
| **Compute Footprint Estimate** | <!-- "<1 GB RAM, <1 min CPU verification" or similar --> |

## 🧪 Zero-Training Test Results
<!-- For RL Environment PRs only -->
<details>

**W&B Link:**

**Examples of the Environment scoring a good example and a bad example:**

</details>

---

## ✅ Developer & Reviewer Checklist
<!-- Common checklist for all PR types - adapt as needed for your PR type -->
- [ ] Code follows project style (black, isort, flake8 pass with pre-commit)
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] New and existing unit tests pass locally with my changes
- [ ] Docstrings added for all new public classes / functions
- [ ] If .env vars required, did you add it to the .env.example in repo root?
