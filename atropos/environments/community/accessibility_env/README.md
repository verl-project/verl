# Accessibility Auto-Fixer Environment for Atropos

**Team/Author:** Accessibility Bot / Josh Garza
**Track:** Objective (WCAG rules are specific and rule-based)
**wandb run:** https://wandb.ai/joshgarza-n-a/atropos/runs/tqpiiofa?nw=nwuserjoshgarza

## Environment Design and Motivation

### Problem Addressed

Web accessibility is crucial for ensuring that websites and web applications are usable by everyone, including people with disabilities. Manually auditing and fixing HTML to meet Web Content Accessibility Guidelines (WCAG) is time-consuming and requires specialized knowledge. This Atropos environment fine-tunes an LLM to automatically identify and apply minimal, correct fixes to HTML snippets to improve their WCAG compliance.

### Why This Is Important

Automating accessibility improvements reduces effort and cost, leading to more inclusive web experiences. A fine-tuned model can serve as a developer assistant, batch-processor for large codebases, or educational tool.

### How the Environment Works

1. **Input:**
   - HTML snippets from `data/accessibility_dataset.jsonl`
   - Each snippet is tagged with WCAG issues to fix (e.g. `missing_alt_text`, `missing_label_for`)
2. **LLM Interaction:**
   - Prompt the model (e.g. GPT-3.5-turbo) to output only the corrected HTML
3. **Scoring (Rule-Based):**
   - Define `AccessibilityRule` classes (e.g. `MissingAltTextRule`, `LabelAssociationRule`, `LinkHasTextRule`)
   - Parse the LLM’s output with BeautifulSoup
   - Check each issue in `issues_to_fix` against the corresponding rule
   - Assign a score:
     - **+1.0** All targeted issues fixed correctly
     - **0.0–0.8** Some but not all issues fixed
     - **–0.5** Parseable HTML, but none of the targeted issues fixed
     - **–1.0** Unparseable HTML or regressions on targeted issues
4. **Output:**
   - Rollouts compatible with Atropos (tokenized prompts/responses, masks, scores) for RL training

### MVP: Targeted WCAG Criteria

1. **Images (`<img>`):** missing or empty `alt` attributes (WCAG 1.1.1)
2. **Form labels:** improper `<label for="…">` associations (WCAG 1.3.1, 3.3.2, 4.1.2)
3. **Links (`<a>`):** lacking discernible text or accessible name (`aria-label`/`aria-labelledby`) (WCAG 2.4.4, 4.1.2)

### Potential Impact

A well-trained model could catch and fix common accessibility errors early, streamlining development and improving inclusivity.

---

## Quickstart Documentation

### 1. Prerequisites

- **Python 3.10+**
- **OpenAI API Key** (export as `OPENAI_API_KEY`):
  ```bash
  export OPENAI_API_KEY="sk-YourActualOpenAIKeyHere"
  ```

### 2. Setup

1. **Clone & enter environment directory**
   ```bash
   cd environments/hack0/your_env_folder_name/
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install lxml
   ```
3. **Ensure Atropos core is installed**
   ```bash
   # From the Atropos root:
   pip install -e .[dev]
   ```

### 3. Running the Environment (process mode)

```bash
python -m environments.hack0.your_env_folder_name.accessibility_env process \
  --env.data_path_to_save_groups environments/hack0/your_env_folder_name/rollouts.jsonl \
  --env.dataset_path data/accessibility_dataset.jsonl \
  --env.total_steps 6 \
  --env.group_size 1 \
  --openai.model_name "gpt-3.5-turbo" \
  --openai.api_key "$OPENAI_API_KEY"
```
