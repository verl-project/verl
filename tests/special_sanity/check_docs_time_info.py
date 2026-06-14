# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Check that every .md and .rst file under docs/ contains the substring "Last updated",
with an allow-list for exceptions.
"""

import sys
from pathlib import Path

# === CONFIGURATION ===

# Relative paths (to docs/) or glob patterns to skip checking
ALLOW_LIST = {
    "docs/README.md",  # you can list individual files
    "docs/legacy/*.rst",  # or glob patterns
    "docs/index.rst",
    "docs/start/install.rst",
    "docs/start/quickstart.rst",
    "docs/README_vllm0.7.md",
}

# The folder to scan
DOCS_DIR = Path("docs")

# === SCRIPT ===


def is_allowed(path: Path) -> bool:
    """
    Return True if `path` matches any entry in ALLOW_LIST.
    """
    rel = str(path)
    for pattern in ALLOW_LIST:
        if Path(rel).match(pattern):
            return True
    return False


def main():
    if not DOCS_DIR.exists():
        print(f"Error: Documentation directory '{DOCS_DIR}' does not exist.", file=sys.stderr)
        sys.exit(1)

    missing = []

    # Gather all .md and .rst files under docs/
    for ext in ("*.md", "*.rst"):
        for path in DOCS_DIR.rglob(ext):
            if is_allowed(path):
                continue

            text = path.read_text(encoding="utf-8", errors="ignore")
            if "Last updated" not in text:
                missing.append(path)

    # Report
    if missing:
        print("\nThe following files are missing the 'Last updated' string:\n")
        for p in missing:
            print(f"  - {p}")
        print(f"\nTotal missing: {len(missing)}\n", file=sys.stderr)
        raise AssertionError(
            "Some documentation files lack a 'Last updated' line. Please include info such as "
            "'Last updated: mm/dd/yyyy' to indicate the last update time of the document."
        )
    else:
        print("✅ All checked files contain 'Last updated'.")


if __name__ == "__main__":
    main()
