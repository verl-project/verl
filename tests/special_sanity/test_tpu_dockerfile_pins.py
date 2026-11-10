# Copyright (c) Meta Platforms, Inc. and affiliates.
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

"""Keep the TPU GRPO image's TransferQueue pin within verl's requirement.

``examples/tpu/grpo/Dockerfile`` pins its Python packages by hand instead of
installing verl's requirements, so a bump of verl's TransferQueue requirement
does not reach the image on its own. A stale pin is only noticed late: the V1
async trainer modes save TransferQueue state with ``transfer_queue.save_checkpoint``
on every checkpoint, so a run on a release without that API fails at its first
``trainer.save_freq`` step.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10
    tomllib = pytest.importorskip("tomli")

REPO_ROOT = Path(__file__).resolve().parents[2]
TPU_DOCKERFILE = REPO_ROOT / "examples" / "tpu" / "grpo" / "Dockerfile"
PACKAGE = "TransferQueue"


def _verl_core_requirement(package: str) -> Requirement:
    """Return ``package``'s requirement from the ``verl-core`` extra in pyproject.toml."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for spec in pyproject["project"]["optional-dependencies"]["verl-core"]:
        requirement = Requirement(spec)
        if canonicalize_name(requirement.name) == canonicalize_name(package):
            return requirement
    raise AssertionError(f"{package} is not listed in the verl-core extra of pyproject.toml")


def _pinned_versions(dockerfile_text: str, package: str) -> list[str]:
    """Return every ``<package>==<version>`` pin outside comments in a Dockerfile, in order of appearance.

    A ``#`` at the start of a line or after a blank starts a comment that runs to the end of the line, also on a line
    inside a multi-line ``RUN``. A pin in a comment is not installed, so it must not satisfy the check.
    """
    active = re.sub(r"(?m)(?:^|[ \t])#.*", "", dockerfile_text)
    return re.findall(rf"(?<![\w.-]){re.escape(package)}==([^\s\"'\\]+)", active, flags=re.IGNORECASE)


def _unsatisfied(pins: list[str], requirement: Requirement) -> list[str]:
    """Return the pins that are not literal versions or fall outside ``requirement``."""
    bad = []
    for pin in pins:
        try:
            version = Version(pin)
        except InvalidVersion:
            bad.append(pin)
            continue
        if not requirement.specifier.contains(version, prereleases=True):
            bad.append(pin)
    return bad


def test_tpu_image_transfer_queue_pin_satisfies_verl_requirement():
    requirement = _verl_core_requirement(PACKAGE)
    pins = _pinned_versions(TPU_DOCKERFILE.read_text(encoding="utf-8"), PACKAGE)

    assert pins, f"no {PACKAGE}==<version> pin found in {TPU_DOCKERFILE.relative_to(REPO_ROOT)}"
    assert not _unsatisfied(pins, requirement), (
        f"{TPU_DOCKERFILE.relative_to(REPO_ROOT)} installs {PACKAGE} {pins}, but verl requires "
        f"{requirement} (pyproject.toml, verl-core). Update the Dockerfile pin."
    )


def test_stale_or_unresolved_transfer_queue_pin_is_reported():
    requirement = Requirement(f"{PACKAGE}>=0.1.11")
    # TransferQueue 0.1.8 has no checkpoint API; written the way the Dockerfile writes its pins.
    stale = 'pip install "fastapi<0.137" \\\n    tensorboard TransferQueue==0.1.8\n'
    pins = _pinned_versions(stale, PACKAGE)

    assert pins == ["0.1.8"]
    assert _unsatisfied(pins, requirement) == ["0.1.8"]
    # A pin through a build argument cannot be checked here, so it is reported rather than skipped.
    assert _unsatisfied(["${TRANSFER_QUEUE_VERSION}"], requirement) == ["${TRANSFER_QUEUE_VERSION}"]
    assert _unsatisfied(["0.1.11", "0.1.12"], requirement) == []


def test_commented_out_transfer_queue_pin_is_not_counted():
    # TransferQueue installed unpinned, its pin only in comments: a comment line, one inside the RUN, a trailing one.
    commented = (
        "# RUN pip install TransferQueue==0.1.11\n"
        "RUN pip install tensorboard \\\n"
        "    # TransferQueue==0.1.11 \\\n"
        "    TransferQueue  # TransferQueue==0.1.11\n"
    )

    assert _pinned_versions(commented, PACKAGE) == []
    assert _pinned_versions(commented + "RUN pip install TransferQueue==0.1.12\n", PACKAGE) == ["0.1.12"]
