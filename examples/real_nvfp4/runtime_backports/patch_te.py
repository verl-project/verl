"""Build-time, hash-guarded backport of the measured row-scale batching path."""

import ast
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from transformer_engine.pytorch.cpp_extensions import gemm

candidate = Path(__file__).resolve().with_name("candidate.py")
assert (
    hashlib.sha256(candidate.read_bytes()).hexdigest()
    == "2c75eb8933b4e812357488387c9d317a3f322befdd3b7687718aa75a4c0ef471"
)
assert version("transformer-engine") == "2.18.0"
target = Path(gemm.__file__)
source = target.read_text()
before = hashlib.sha256(source.encode()).hexdigest()
assert before == "b24b7575533cdbaddec4c798d7b533b6b388ac6a49cb3cd19f77766f90b560d2", (
    before
)
tree = ast.parse(candidate.read_text())
function = next(
    n
    for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "batched_row_scaled_gemm"
)


class Rename(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == "ORIGINAL":
            node.id = "_nvfp4_rowscale_original_grouped_gemm"
        elif node.id == "COUNTERS":
            node.id = "_nvfp4_rowscale_backport_counters"
        return node


function = Rename().visit(function)
addition = (
    "\n\n# Local TE 2.18 backport: batched per-token global-scale epilogue.\n"
    "NVFP4_ROWSCALE_GROUPED_BACKPORT = '20260915-v1'\n"
    "_nvfp4_rowscale_original_grouped_gemm = general_grouped_gemm\n"
    "_nvfp4_rowscale_backport_counters = {'batched': 0, 'fallback': 0}\n\n"
    + ast.unparse(function)
    + "\n\ngeneral_grouped_gemm = batched_row_scaled_gemm\n"
)
patched = source + addition
ast.parse(patched)
# This runs in an ephemeral build container; the base image is never modified.
target.write_text(patched)
print(
    "TE_ROWSCALE_PATCH",
    json.dumps(
        {
            "path": str(target),
            "before": before,
            "after": hashlib.sha256(patched.encode()).hexdigest(),
        }
    ),
    flush=True,
)
