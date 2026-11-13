# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Static guards for the cross-file contracts a merge between diverged lines breaks.

Every defect found while merging a long feature branch into `main` had the same shape: a call
site and its target ended up on different sides of the merge, and each file was still valid on
its own. Syntax checks, duplicate-definition scans and `ruff` cannot see that class:

- an importer kept naming a helper the other side deleted  -> ImportError at package import;
- a function's parameters came from one side while its caller kept the other's keywords
  -> TypeError on the first call, after a forward has already run;
- a value was computed and never forwarded -> a fast path silently stops being taken.

These guards read the sources as text and compare the two sides of each contract, so they run
without torch, a GPU or an initialized distributed state. They are deliberately narrow: they
pin the contracts this merge actually touched, not a general-purpose checker.

What they do **not** cover, so no comment here claims more than the code does: a keyword passed
positionally, a dict built in one function and spread in another, a call reached through
``getattr`` or a registry, and any module the merge touched but that is not named below.
"""

import ast
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[2]
MCORE_UTIL = REPO / "verl/models/mcore/util.py"
MODEL_FORWARD = REPO / "verl/models/mcore/model_forward.py"
MODEL_FORWARD_FUSED = REPO / "verl/models/mcore/model_forward_fused.py"


def _module_functions(path: pathlib.Path) -> dict[str, ast.AST]:
    """Module-level function definitions, by name."""
    tree = ast.parse(path.read_text())
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)}


def _module_scope_names(path: pathlib.Path) -> set[str]:
    """Every name a module binds at top level: definitions, assignments and imports."""
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.Import | ast.ImportFrom):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
    return names


def _accepted_keywords(func: ast.AST) -> tuple[set[str], bool]:
    """Keywords a function accepts, and whether it takes ``**kwargs``."""
    keywords = {a.arg for a in func.args.args + func.args.kwonlyargs}
    return keywords, func.args.kwarg is not None


def _calls_to(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


def test_every_name_imported_from_mcore_util_is_defined_there():
    """Importers must not name helpers this module does not define.

    This is the ImportError shape: `verl/models/mcore/__init__.py` reaches the forward module at
    import time, so one missing name breaks every Megatron engine run.
    """
    defined = _module_scope_names(MCORE_UTIL)
    missing: list[str] = []
    # Include new source files while excluding virtualenvs and downloaded models.
    # Walking those ignored trees can take minutes on a shared training filesystem.
    sources = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.py"], cwd=REPO
        )
        .decode()
        .split("\0")
    )
    for relative_path in sorted(set(sources) - {""}):
        path = REPO / relative_path
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            # Absolute importers, and siblings that say `from .util import ...` -- the forward
            # modules use the relative form, which is why an absolute-only check missed them.
            absolute = node.module == "verl.models.mcore.util"
            sibling = node.level == 1 and node.module == "util" and path.parent == MCORE_UTIL.parent
            if not (absolute or sibling):
                continue
            for alias in node.names:
                if alias.name != "*" and alias.name not in defined:
                    missing.append(f"{path.relative_to(REPO)}:{node.lineno} imports {alias.name}")
    assert missing == [], "imported from verl.models.mcore.util but not defined there:\n" + "\n".join(missing)


def test_the_forward_files_call_preprocessing_with_declared_keywords_only():
    """A packing call site must not pass a keyword its target does not declare.

    The merge left `use_fp8_padding` on one side and `quant_align_size` on the other; a call with
    the wrong one is a TypeError on the first forward, not at import time.
    """
    targets = {}
    for name in ("preprocess_thd_engine", "preprocess_bshd_engine", "preprocess_vlm_thd_engine"):
        funcs = _module_functions(MCORE_UTIL)
        assert name in funcs, f"{name} disappeared from verl/models/mcore/util.py"
        targets[name] = _accepted_keywords(funcs[name])

    offenders: list[str] = []
    packing_names = ("use_fp8_padding", "quant_align_size", "min_local_rows", "cp_layout", "local_cp_size")
    all_declared = set().union(*(declared for declared, _ in targets.values()))
    for path in (MODEL_FORWARD, MODEL_FORWARD_FUSED):
        tree = ast.parse(path.read_text())
        for name, (declared, takes_kwargs) in targets.items():
            for call in _calls_to(tree, name):
                for keyword in call.keywords:
                    if keyword.arg is None or takes_kwargs:
                        continue
                    if keyword.arg not in declared:
                        offenders.append(f"{path.name}:{call.lineno} {name}({keyword.arg}=...)")
        # A call can also be fed through a `**kwargs` dict built elsewhere in the same function --
        # the mxfp8 padding regression lived exactly there, in `thd_kwargs = dict(...)`.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
                continue
            keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
            if not any(k in packing_names for k in keys):
                continue
            for key in keys:
                if key not in all_declared:
                    offenders.append(f"{path.name}:{node.lineno} kwargs dict carries {key}=")
    assert offenders == [], "call site passes an undeclared keyword:\n" + "\n".join(offenders)


def test_the_distillation_call_passes_only_what_the_inner_loss_declares():
    """The same module can still hold two halves of one contract, as it did after the merge."""
    path = REPO / "verl/trainer/distillation/losses.py"
    funcs = _module_functions(path)
    assert "compute_topk_loss" in funcs, "compute_topk_loss disappeared"
    declared, takes_kwargs = _accepted_keywords(funcs["compute_topk_loss"])
    if takes_kwargs:
        return

    offenders = [
        f"{path.name}:{call.lineno} compute_topk_loss({keyword.arg}=...)"
        for call in _calls_to(ast.parse(path.read_text()), "compute_topk_loss")
        for keyword in call.keywords
        if keyword.arg is not None and keyword.arg not in declared
    ]
    assert offenders == [], "distillation loss call passes an undeclared keyword:\n" + "\n".join(offenders)


def test_the_chunk_kwargs_the_forward_builds_are_actually_forwarded():
    """A dict of chunk kwargs that is built but never spread means the chunked path is dead.

    Scope: the dict and its spread must sit in the same function, which is how the forward builds
    and forwards them. A dict built in one function and spread in another is not caught here.

    With `DSV41_SEQ_CHUNKS > 1` the tape never reaches the model: each pass packs a full-sequence
    CP shard, the compressed KV is never reused, and the loss is still zeroed per chunk window --
    silently, because the unchunked path is correct.
    """
    path = REPO / "verl/workers/engine/megatron/transformer_impl.py"
    offenders: list[str] = []
    for func in ast.walk(ast.parse(path.read_text())):
        if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        assigned = {
            target.id
            for node in ast.walk(func)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and "kwargs" in target.id
        }
        for name in assigned:
            spread = any(
                keyword.arg is None and getattr(keyword.value, "id", None) == name
                for node in ast.walk(func)
                if isinstance(node, ast.Call)
                for keyword in node.keywords
            )
            if not spread:
                offenders.append(f"{path.name}:{func.lineno} {func.name} builds {name} and never spreads it")
    assert offenders == [], "chunk kwargs built but not forwarded:\n" + "\n".join(offenders)
