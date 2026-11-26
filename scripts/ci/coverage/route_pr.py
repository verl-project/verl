# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Combine the precision selector's workflow names with Ascend PR path rules."""

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


def workflow_matches(paths, rules):
    """Apply the ordered GitHub path filters, including negative patterns."""
    matched = set()
    for rule in rules:
        # These filters use literal paths, *, ** and **/ (zero or more directories).
        parts = re.split(r"(\*\*/|\*\*|\*)", rule.removeprefix("!"))
        globs = {"**/": "(?:.*/)?", "**": ".*", "*": "[^/]*"}
        pattern = re.compile("".join(globs.get(part, re.escape(part)) for part in parts), re.DOTALL)
        hits = {path for path in paths if pattern.fullmatch(path)}
        if rule.startswith("!"):
            matched -= hits
        else:
            matched |= hits
    return bool(matched)


@dataclass
class WorkflowSelection:
    by_path: set[str]
    by_coverage: set[str]

    @property
    def selected(self) -> list[str]:
        # Incomplete coverage can add workflows but cannot cancel a path match.
        return sorted(self.by_path | self.by_coverage)


def select_workflows(paths, recommended=()):
    """Select Ascend workflows for repository-relative paths and coverage recommendations."""
    rules = json.loads(Path(__file__).with_name("pr_paths.json").read_text())
    by_path = {name for name, patterns in rules.items() if workflow_matches(paths, patterns)}
    return WorkflowSelection(by_path, set(recommended).intersection(rules))


def main():
    diff = subprocess.check_output(
        ["git", "diff", "--name-only", "--no-renames", "-z", f"{os.environ['BASE_SHA']}...{os.environ['HEAD_SHA']}"]
    )
    paths = os.fsdecode(diff).split("\0")[:-1]
    exported = os.environ.get("COVERAGE_WORKFLOWS")
    recommended = json.loads(exported) if exported else []
    selection = select_workflows(paths, recommended)
    outcome = os.environ["SELECTOR_OUTCOME"]
    selected = selection.selected
    comparison = {
        "base_sha": os.environ["BASE_SHA"],
        "head_sha": os.environ["HEAD_SHA"],
        "baseline_run": os.environ["BASELINE_RUN"] or None,
        "selector_outcome": outcome,
        "precision_test": sorted(set(recommended)) if exported else None,
        "path_filters": sorted(selection.by_path),
        "running": selected,
    }
    print(f"ascend_workflow_selection={json.dumps(comparison)}")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        print(f"workflows={json.dumps(selected)}", file=output)
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as summary:
        print("### Ascend test selection", file=summary)
        print(f"\nCoverage run: {os.environ['BASELINE_RUN'] or 'unavailable'} (selector: {outcome}).", file=summary)
        coverage_choices = ", ".join(sorted(selection.by_coverage)) or ("none" if exported else "unavailable")
        print(f"\nRecommended: {coverage_choices}.", file=summary)
        print(f"\nSelected by path: {', '.join(sorted(selection.by_path)) or 'none'}.", file=summary)
        print(f"\nRunning: {', '.join(selected) or 'none'}.", file=summary)


if __name__ == "__main__":
    main()
