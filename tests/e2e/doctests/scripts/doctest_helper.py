#!/usr/bin/env python3

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

"""Documentation extraction, change selection and resource planning.

Adapted from vllm-ascend's doctest_helper.py. Commands never install packages
or start training. RST uses ordinary comments (.. doctest: name), not a custom
Sphinx directive. Matrix policy is explicit in config.json; versions for source
images come from the same Dockerfiles used by the repository's image builds.
"""

import argparse
import copy
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[4]

CONFIG_PATH = "tests/e2e/doctests/config.json"

INSTALL_DOC = "docs/ascend_tutorial/get_start/install_guidance.rst"

QUICK_DOC = "docs/ascend_tutorial/get_start/quick_start.rst"

CASES = ("fsdp2_vllm", "megatron_vllm", "fsdp2_sglang", "megatron_sglang")

BACKENDS = ("vllm", "sglang")

MARKERS_BY_FILE = {
    INSTALL_DOC: (
        "installation-prerequisites-ubuntu",
        "installation-prerequisites-openeuler",
        "installation-vllm-environment",
        "installation-sglang-environment",
        "installation-checkout",
        "installation-vllm-install",
        "installation-sglang-install",
    ),
    QUICK_DOC: (
        "quickstart-environment",
        "quickstart-model",
        "quickstart-data",
        *(f"quickstart-{case}" for case in CASES),
    ),
}

FILE_BY_MARKER = {marker: path for path, markers in MARKERS_BY_FILE.items() for marker in markers}

SHARED_PATHS = {
    CONFIG_PATH,
    ".github/workflows/doctest_ascend.yml",
    "tests/e2e/doctests/scripts/doctest_helper.py",
    "tests/e2e/doctests/scripts/common.sh",
    "tests/e2e/doctests/scripts/run_doctests.sh",
    "tests/e2e/doctests/scripts/verify_runtime.py",
}

MARKER_RE = re.compile(r"^[ \t]*\.\. doctest: ([A-Za-z0-9][A-Za-z0-9._-]*)[ \t]*$")

DIRECTIVE_RE = re.compile(r"^( *)\.\. (?:code|code-block):: (bash|python)[ \t]*$")

MACRO_RE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


class DoctestError(ValueError):
    """Invalid documentation, configuration or unavailable test resources."""


def git(*args):
    """Run Git with explicit error handling and UTF-8 output."""
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        raise DoctestError(result.stderr.strip() or f"git failed: {args}")
    return result.stdout


def read_repo_text(path, ref=None, allow_missing=False):
    """Read working-tree or historical content; never hide an invalid Git ref."""
    if ref is None:
        target = REPO_ROOT / path
        if allow_missing and not target.is_file():
            return None
        return target.read_text(encoding="utf-8")
    git("rev-parse", "--verify", f"{ref}^{{commit}}")
    names = git("ls-tree", "-r", "--name-only", ref, "--", path).splitlines()
    if path not in names and allow_missing:
        return None
    return git("show", f"{ref}:{path}")


def extract_doctest_block(text, marker, source="input"):
    """Extract one indented RST block, rejecting duplicate or malformed markers."""
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if (m := MARKER_RE.fullmatch(line)) and m[1] == marker]
    if not hits:
        return None
    if len(hits) != 1:
        raise DoctestError(f"Duplicate marker {marker!r} in {source}")
    index = hits[0] + 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    opening = DIRECTIVE_RE.fullmatch(lines[index]) if index < len(lines) else None
    if not opening:
        raise DoctestError(f"Marker {marker!r} in {source} must precede a bash/python code directive")
    base_indent = len(opening[1])
    index += 1

    while index < len(lines) and lines[index].lstrip().startswith(":"):
        index += 1
    if index >= len(lines) or lines[index].strip():
        raise DoctestError(f"Missing blank line after directive for {marker!r}")
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        raise DoctestError(f"Empty code block for {marker!r}")
    indent = len(lines[index]) - len(lines[index].lstrip(" "))
    if indent <= base_indent:
        raise DoctestError(f"Unindented code block for {marker!r}")
    body = []
    for line in lines[index:]:
        if not line.strip():
            body.append("")
            continue
        line_indent = len(line) - len(line.lstrip(" "))
        if line_indent <= base_indent:
            break
        if line_indent < indent:
            raise DoctestError(f"Inconsistent code indentation for {marker!r}")
        body.append(line[indent:])
    return "\n".join(body).rstrip() + "\n"


def require_doctest_block(marker, ref=None):
    """Resolve a registered marker and fail if the documentation dropped it."""
    if marker not in FILE_BY_MARKER:
        raise DoctestError(f"Unknown doctest marker: {marker}")
    path = FILE_BY_MARKER[marker]
    text = read_repo_text(path, ref)
    block = extract_doctest_block(text, marker, path)
    if block is None:
        raise DoctestError(f"Missing marker {marker!r} in {path}")
    return block


def load_config(ref=None):
    """Read the explicit supported hardware/OS/image policy."""
    config = json.loads(read_repo_text(CONFIG_PATH, ref))
    if not isinstance(config.get("profiles"), list) or not config["profiles"]:
        raise DoctestError("config.json requires nonempty profiles")
    ids = set()
    for profile in config["profiles"]:
        required = {
            "id",
            "device",
            "os",
            "runner",
            "devices",
            "backend",
            "quickstart_image",
            "dockerfile",
            "source_install",
        }
        if not required <= profile.keys():
            raise DoctestError(f"Incomplete profile: {profile}")
        if profile["id"] in ids:
            raise DoctestError(f"Duplicate profile: {profile['id']}")
        ids.add(profile["id"])
        if profile["backend"] not in BACKENDS or profile["device"] not in ("a2", "a3"):
            raise DoctestError(f"Unsupported backend/device in {profile['id']}")
        if profile["os"] not in ("ubuntu", "openeuler") or profile["devices"] < 4:
            raise DoctestError(f"Invalid OS/device count in {profile['id']}")
    return config


def expand_macros(content, extra, marker):
    """Expand configured scalar macros; reject missing names instead of running them."""

    def replace(match):
        key = match[1]
        if key not in extra or not isinstance(extra[key], str | int | float):
            raise DoctestError(f"Unknown/non-scalar macro {key!r} in {marker}")
        return str(extra[key])

    return MACRO_RE.sub(replace, content)


def validate_documents(ref=None):
    """Require every marker before any plan is published or command is executed."""
    for marker in FILE_BY_MARKER:
        require_doctest_block(marker, ref)


def select_doctests(base, head):
    """Compare registered blocks and dependencies, including the first migration PR."""
    validate_documents(head)
    paths = set(git("diff", "--name-only", "--no-renames", base, head, "--").splitlines())
    quick, installation = set(), set()
    cache = {}
    for path, markers in MARKERS_BY_FILE.items():
        if path not in paths:
            continue
        cache[path] = (read_repo_text(path, base, allow_missing=True), read_repo_text(path, head))
        old, new = cache[path]
        for marker in markers:
            before = extract_doctest_block(old, marker, path) if old else None
            after = extract_doctest_block(new, marker, path)
            if before == after:
                continue
            if marker in ("quickstart-environment", "quickstart-model", "quickstart-data"):
                quick.update(CASES)
                installation.update(
                    CASES
                )
            elif marker.startswith("quickstart-"):
                case = marker.removeprefix("quickstart-")
                quick.add(case)
                installation.add(case)
            elif marker in (
                "installation-checkout",
                "installation-prerequisites-ubuntu",
                "installation-prerequisites-openeuler",
            ):
                installation.update(CASES)
            else:
                backend = marker.split("-")[1]
                installation.update(case for case in CASES if case.endswith(backend))
    for case in CASES:
        script = f"tests/special_npu/quick_start/run_qwen3_0_6b_{case}_ascend.sh"
        if script in paths:
            quick.add(case)
            installation.add(case)
    for backend in BACKENDS:
        if f"scripts/install_{backend}_mcore_npu.sh" in paths:
            installation.update(case for case in CASES if case.endswith(backend))
    if (
        "tests/e2e/doctests/001-quickstart-test.sh" in paths or "examples/data_preprocess/gsm8k.py" in paths
    ):
        quick.update(CASES)
        installation.update(CASES)
    if paths & {
        "tests/e2e/doctests/002-installation-test.sh",
        "requirements-npu.txt",
        ".gitmodules",
    }:
        installation.update(CASES)
    if paths & SHARED_PATHS or any(p.startswith("docker/ascend/") for p in paths):
        quick.update(CASES)
        installation.update(CASES)
    return {"quickstart": sorted(quick), "installation": sorted(installation)}


def base_image(profile):
    """Read source-install CANN images from the existing Dockerfiles."""
    content = read_repo_text(profile["dockerfile"])
    match = re.search(r"^FROM\s+(\S+)", content, re.MULTILINE)
    if not match or "$" in match[1]:
        raise DoctestError(f"Expected a literal FROM image in {profile['dockerfile']}")
    return match[1]


def source_refs(backend):
    """Extract pinned clone branches from the installer being tested."""
    text = read_repo_text(f"scripts/install_{backend}_mcore_npu.sh")
    refs = []
    for line in text.splitlines():
        match = re.search(r"git clone (.+?) (https://\S+)", line)
        if match:
            branch = re.search(r"(?:--branch|-b)\s+(\S+)", match[1])
            if branch:
                refs.append((match[2], branch[1]))
    return refs


def build_doctest_plan(quickstart, installation, device="all", os_name="all"):
    """Expand valid cases across explicitly supported profiles, not a blind Cartesian product."""
    config = load_config()
    plan = {"quickstart": {"include": []}, "installation": {"include": []}, "skipped": []}
    for kind, cases in (("quickstart", quickstart), ("installation", installation)):
        for case in cases:
            if case not in CASES:
                raise DoctestError(f"Unknown case: {case}")
            count = 0
            for profile in config["profiles"]:
                if not case.endswith(profile["backend"]):
                    continue
                if device not in ("all", profile["device"]) or os_name not in ("all", profile["os"]):
                    continue
                if kind == "installation" and not profile["source_install"]:
                    continue
                entry = {
                    key: profile[key] for key in ("device", "os", "runner", "devices", "backend")
                }
                entry.update(case=case, profile=profile["id"])
                entry["image"] = (
                    expand_macros(profile["quickstart_image"], config.get("extra", {}), profile["id"])
                    if kind == "quickstart"
                    else base_image(profile)
                )
                plan[kind]["include"].append(entry)
                count += 1
            if not count:
                raise DoctestError(f"No supported {kind} profile for {case}, device={device}, os={os_name}")
    for kind in ("quickstart", "installation"):
        plan[f"run_{kind}"] = bool(plan[kind]["include"])
    return plan


def registry_image_exists(image):
    """Check OCI manifests; only an explicit 404 is 'missing', auth/network errors fail."""
    repository, sep, tag = image.rpartition(":")
    if not sep or "/" not in repository:
        raise DoctestError(f"Expected a tagged image: {image}")
    registry, path = repository.split("/", 1)
    url = f"https://{registry}/v2/{path}/manifests/{quote(tag, safe='')}"
    headers = {
        "Accept": ", ".join(
            (
                "application/vnd.oci.image.index.v1+json",
                "application/vnd.oci.image.manifest.v1+json",
                "application/vnd.docker.distribution.manifest.list.v2+json",
                "application/vnd.docker.distribution.manifest.v2+json",
            )
        )
    }
    try:
        with urlopen(Request(url, headers=headers), timeout=30):
            return True
    except HTTPError as error:
        if error.code == 404:
            return False
        if error.code != 401:
            raise DoctestError(f"Registry check failed ({error.code}): {image}") from error
        challenge = error.headers.get("WWW-Authenticate", "")
        if not challenge.lower().startswith("bearer "):
            raise DoctestError(f"Registry requires credentials: {image}") from error
        fields = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
    realm = fields.pop("realm", "")
    if urlparse(realm).scheme != "https":
        raise DoctestError("Registry authentication requires an HTTPS token endpoint")
    fields.setdefault("scope", f"repository:{path}:pull")
    with urlopen(realm + ("&" if "?" in realm else "?") + urlencode(fields), timeout=30) as response:
        token_response = json.load(response)
    token = token_response.get("token", token_response.get("access_token"))
    if not token:
        raise DoctestError(f"No anonymous pull token for {image}")
    headers["Authorization"] = f"Bearer {token}"
    try:
        with urlopen(Request(url, headers=headers), timeout=30):
            return True
    except HTTPError as error:
        if error.code == 404:
            return False
        raise DoctestError(f"Authenticated registry check failed ({error.code}): {image}") from error


def source_ref_exists(url, ref):
    """Distinguish absent refs from unreachable repositories."""
    result = subprocess.run(
        [
            "git",
            "ls-remote",
            "--exit-code",
            "--heads",
            "--tags",
            url,
            f"refs/heads/{ref}",
            f"refs/tags/{ref}",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode not in (0, 2):
        raise DoctestError(f"Cannot check {url}@{ref}: {result.stderr.strip()}")
    return result.returncode == 0


def check_plan_resources(plan, skip_missing=False):
    """Optionally filter explicitly missing resources and report every skipped job."""
    plan = copy.deepcopy(plan)
    images, refs = {}, {}
    for kind in ("quickstart", "installation"):
        kept = []
        for entry in plan[kind]["include"]:
            image = entry["image"]
            if image not in images:
                images[image] = registry_image_exists(image)
            missing = [] if images[image] else [image]
            if kind == "installation":
                for url, ref in source_refs(entry["backend"]):
                    if (url, ref) not in refs:
                        refs[url, ref] = source_ref_exists(url, ref)
                    if not refs[url, ref]:
                        missing.append(f"{url}@{ref}")
            if missing:
                reason = f"{kind}/{entry['profile']}/{entry['case']}: missing {', '.join(missing)}"
                if not skip_missing:
                    raise DoctestError(reason)
                plan["skipped"].append(reason)
            else:
                kept.append(entry)
        plan[kind]["include"] = kept
        plan[f"run_{kind}"] = bool(kept)
    return plan


def parse_args():
    """Validate manual and diff modes as mutually exclusive."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("marker", choices=sorted(FILE_BY_MARKER))
    extract.add_argument("--ref")
    extract.add_argument("--expand-macros", action="store_true")
    commands.add_parser("validate")
    plan = commands.add_parser("plan")
    plan.add_argument("--base")
    plan.add_argument("--head")
    plan.add_argument("--quickstart", choices=("none", "all", *CASES), default="none")
    plan.add_argument("--installation", choices=("none", "all", *CASES), default="none")
    plan.add_argument("--device", choices=("all", "a2", "a3"), default="all")
    plan.add_argument("--os", choices=("all", "ubuntu", "openeuler"), default="all")
    plan.add_argument("--check-resources", action="store_true")
    plan.add_argument("--skip-missing-resources", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        if bool(args.base) != bool(args.head):
            parser.error("--base and --head must be supplied together")
        if args.base and (args.quickstart != "none" or args.installation != "none"):
            parser.error("manual case choices cannot be combined with diff selection")
        if args.skip_missing_resources and not args.check_resources:
            parser.error("--skip-missing-resources requires --check-resources")
    return args


def main():
    """Print extracted code or a JSON plan; never execute the blocks."""
    args = parse_args()
    if args.command == "extract":
        block = require_doctest_block(args.marker, args.ref)
        if args.expand_macros:
            block = expand_macros(block, load_config(args.ref).get("extra", {}), args.marker)
        sys.stdout.write(block)
    elif args.command == "validate":
        validate_documents()
        load_config()
        print(f"Validated {len(FILE_BY_MARKER)} marked code blocks")
    else:
        validate_documents()
        if args.base:
            selection = select_doctests(args.base, args.head)
        else:
            selection = {
                kind: list(CASES)
                if (value := getattr(args, kind)) == "all"
                else ([] if value == "none" else [value])
                for kind in ("quickstart", "installation")
            }
        plan = build_doctest_plan(
            selection["quickstart"], selection["installation"], args.device, args.os
        )
        if args.check_resources:
            plan = check_plan_resources(plan, args.skip_missing_resources)
        print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DoctestError, OSError, URLError, subprocess.TimeoutExpired) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
