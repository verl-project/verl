#!/usr/bin/env python3

import html
import json
import sys
import textwrap
from pathlib import Path

try:
    import fire
except ImportError:
    fire = None

try:
    import markdown
except ImportError:
    print(
        "Error: The 'markdown' library is required. Please install it:", file=sys.stderr
    )
    print("pip install markdown", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
# Assumes template.html is in the same directory as the script
SCRIPT_DIR = Path(__file__).parent.resolve()
TEMPLATE_FILE = SCRIPT_DIR / "template.html"

# --- Helper Functions ---


def get_score_class(score):
    """Determines the CSS class based on the score."""
    try:
        val = float(score)
        if val > 0:
            return "reward-positive"
        elif val < 0:
            return "reward-negative"
        else:
            return "reward-zero"
    except (ValueError, TypeError):
        return ""  # No specific class if score is not numeric


def create_html_for_group(group_data, index):
    """Generates HTML snippet for a single group."""
    messages = group_data.get("messages", [])
    scores = group_data.get("scores", [])

    if len(messages) != len(scores):
        print(
            f"Warning: Group {index} has mismatched lengths for messages "
            f"({len(messages)}) and scores ({len(scores)}). "
            f"Skipping items.",
            file=sys.stderr,
        )
        min_len = min(len(messages), len(scores))
        messages = messages[:min_len]
        scores = scores[:min_len]

    items_html = ""
    for i, (msg, score) in enumerate(zip(messages, scores)):
        rendered_markdown = markdown.markdown(
            msg, extensions=["fenced_code", "tables", "nl2br"]
        )
        score_class = get_score_class(score)
        item_id = f"group-{index}-item-{i}"
        items_html += textwrap.dedent(
            f"""\
            <div class="item {score_class}" id="{item_id}">
                <h4>Content {i}</h4>
                <div class="content-block">
                    {rendered_markdown}
                </div>
                <p><strong>Reward:</strong> {html.escape(str(score))}</p>
            </div>
        """
        )

    if not items_html:
        # Handle case where after length correction, there are no items
        print(
            f"Warning: Group {index} resulted in no items after length correction.",
            file=sys.stderr,
        )
        return ""  # Skip this group entirely in the output

    # Use <details> and <summary> for native collapsibility
    group_html = textwrap.dedent(
        f"""\
        <details>
            <summary>Group {index}</summary>
            <div class="group-content">
                {items_html}
            </div>
        </details>
    """
    )
    return group_html


# --- Main Function ---


def generate_html(input_path: str, output_path: str = None):
    """
    Generates a static HTML file rendering messages from a JSONL file,
    using an external template file (template.html).
    Each line in the JSONL file should be a JSON object with 'messages' (list of strings)
    and 'scores' (list of numbers/strings) keys.
    Args:
        input_path: Path to the input JSONL file.
        output_path: Path to the output HTML file. Defaults to '{input_stem}.html'.
    """
    input_filepath = Path(input_path)
    if not input_filepath.is_file():
        print(f"Error: Input file not found: {input_filepath}", file=sys.stderr)
        sys.exit(1)

    if output_path is None:
        output_filepath = input_filepath.with_suffix(".html")
    else:
        output_filepath = Path(output_path)

    # Ensure output directory exists
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    # --- Read HTML Template ---
    try:
        with open(TEMPLATE_FILE, "r", encoding="utf-8") as f_template:
            html_template_content = f_template.read()
    except FileNotFoundError:
        print(f"Error: Template file not found: {TEMPLATE_FILE}", file=sys.stderr)
        print(
            "Please ensure 'template.html' is in the same directory as the script.",
            file=sys.stderr,
        )
        sys.exit(1)
    except IOError as e:
        print(f"Error reading template file {TEMPLATE_FILE}: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Process JSONL Data ---
    all_groups_html = []
    group_index = 0
    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                try:
                    data = json.loads(line)
                    if (
                        isinstance(data, dict)
                        and "messages" in data
                        and "scores" in data
                    ):
                        group_html = create_html_for_group(data, group_index)
                        if group_html:  # Only add if group wasn't skipped
                            all_groups_html.append(group_html)
                            group_index += 1
                    else:
                        print(
                            f"Warning: Skipping line {line_num}. "
                            f"Invalid format (missing 'messages' or 'scores'): "
                            f"{line[:100]}...",
                            file=sys.stderr,
                        )
                except json.JSONDecodeError:
                    print(
                        f"Warning: Skipping line {line_num}. Invalid JSON: {line[:100]}...",
                        file=sys.stderr,
                    )

    except IOError as e:
        print(f"Error reading input file {input_filepath}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Assemble Final HTML ---
    if not all_groups_html:
        print(
            "Warning: No valid groups found to render in the input file.",
            file=sys.stderr,
        )
        groups_content = "<p>No data to display. Input file might be empty or contain invalid data.</p>"
    else:
        groups_content = "\n".join(all_groups_html)

    # Prepare title
    title = f"Rendered Messages - {input_filepath.name}"

    # Populate the main template read from the file
    try:
        final_html = html_template_content.format(
            title=html.escape(title), groups_html=groups_content
        )
    except KeyError as e:
        print(
            f"Error: Template file '{TEMPLATE_FILE}' is missing a required placeholder: {{{e}}}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Write the output HTML file ---
    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(final_html)
        print(f"Successfully generated HTML file: {output_filepath.absolute()}")
    except IOError as e:
        print(f"Error writing output file {output_filepath}: {e}", file=sys.stderr)
        sys.exit(1)


# --- Command Line Interface Handling ---

if __name__ == "__main__":
    if fire is not None:
        fire.Fire(generate_html)
    else:
        print("`fire` is not installed.")
