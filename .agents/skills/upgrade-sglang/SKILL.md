---
name: upgrade-sglang
description: Upgrade SGLang in veRL by auditing affected call sites, cross-referencing upstream source, and updating for compatibility.
---

# Upgrade SGLang

Upgrade the pinned SGLang version in veRL and update all affected call sites.

## Usage

```
Target SGLang version: $VERSION
```

`$VERSION`: target SGLang version tag, e.g. `v0.5.3`, `0.6.0`. If omitted, check
current pinned version in `requirements_sglang.txt` and validate compatibility.

## Prerequisites — Clone Upstream Source

```bash
SGLANG_DIR="$(git rev-parse --show-toplevel)/sglang-src"
# Validate VERSION to prevent command injection
if [[ ! "$VERSION" =~ ^[a-zA-Z0-9._/-]+$ ]]; then
  echo "Error: Invalid version format"; exit 1
fi
if [ ! -d "$SGLANG_DIR" ]; then
  git clone --depth 1 --branch "${VERSION}" https://github.com/sgl-project/sglang.git "$SGLANG_DIR"
else
  cd "$SGLANG_DIR" && git fetch origin && git checkout "${VERSION}" && cd -
fi
```

If cloning fails, stop and report to the user.

---

## Affected Files in veRL

### Primary — Most Likely to Break

| File | SGLang APIs Used |
|------|-----------------|
| `verl/workers/rollout/sglang_rollout/async_sglang_server.py` | `sglang.srt.entrypoints.engine`, `http_server` (`launch_server`, `add_api_routes`, etc.), `io_struct` (GenerateReqInput, BatchStrOut, BatchEmbeddingOut, etc.), `tokenizer_manager.ServerStatus`, `utils.common.add_prometheus_middleware`, `layers.moe.routed_experts_capturer.extract_routed_experts_from_meta_info` |
| `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | `sglang.srt.entrypoints.engine`, `server_args.ServerArgs`, `srt.utils` (kill_process_tree, etc.), `weight_sync.utils._preprocess_tensor_for_update_weights`, `weight_sync.utils.update_weights`, `io_struct.LoadLoRAAdapterFromTensorsReqInput` |
| `verl/workers/rollout/sglang_rollout/http_server_engine.py` | `entrypoints.EngineBase.EngineBase`, `http_server.launch_server`, `io_struct.UpdateWeightsFromTensorReqInput`, `server_args.ServerArgs`, `utils.kill_process_tree` |

### Secondary — Utilities

| File | SGLang APIs Used |
|------|-----------------|
| `verl/utils/sglang/sglang_fp8_utils.py` | Commented-out: `distributed.device_communicators.pynccl.PyNcclCommunicator`, `distributed.utils.statelessprocessgroup` — verify if re-enabled |
| `verl/experimental/reward_loop/router/inner_sglang_router.py` | Internal router using SGLang HTTP API |

### Version / Docker

| File | Usage |
|------|-------|
| `requirements_sglang.txt` | Pinned version: `sglang[all]==X.Y.Z` |
| `docker/Dockerfile.stable.sglang` | Docker image SGLang version |
| `docker/verl*/Dockerfile.app.sglang*` | Versioned Docker images |

---

## Audit Workflow

### Step 1: Identify Current Pinned Version

```bash
cat requirements_sglang.txt | grep sglang
```

### Step 2: Enumerate All SGLang Imports

```bash
grep -rn "from sglang\|import sglang" verl/ --include="*.py" | grep -v "__pycache__"
```

### Step 3: Cross-Reference Against Upstream Source

For each imported symbol, verify it still exists at `$VERSION` in `sglang-src/`:

**Key symbols to always check:**

| Symbol | Upstream Source Path |
|--------|----------------------|
| `ServerArgs` | `sglang-src/python/sglang/srt/server_args.py` |
| `EngineBase` | `sglang-src/python/sglang/srt/entrypoints/EngineBase.py` |
| `launch_server` | `sglang-src/python/sglang/srt/entrypoints/http_server.py` |
| `UpdateWeightsFromTensorReqInput` | `sglang-src/python/sglang/srt/managers/io_struct.py` |
| `LoadLoRAAdapterFromTensorsReqInput` | `sglang-src/python/sglang/srt/managers/io_struct.py` |
| `GenerateReqInput` | `sglang-src/python/sglang/srt/managers/io_struct.py` |
| `ServerStatus` | `sglang-src/python/sglang/srt/managers/tokenizer_manager.py` |
| `update_weights` | `sglang-src/python/sglang/srt/weight_sync/utils.py` |
| `_preprocess_tensor_for_update_weights` | `sglang-src/python/sglang/srt/weight_sync/utils.py` |
| `kill_process_tree` | `sglang-src/python/sglang/srt/utils/__init__.py` or `utils.py` |
| `add_prometheus_middleware` | `sglang-src/python/sglang/srt/utils/common.py` |
| `extract_routed_experts_from_meta_info` | `sglang-src/python/sglang/srt/layers/moe/routed_experts_capturer.py` |

For each symbol, check:
- Does it still exist at this path?
- Has the class/function signature changed?
- Has the module been moved or renamed?

### Step 4: io_struct Is Especially Fragile

`sglang.srt.managers.io_struct` changes frequently between releases. Fully audit all
symbols imported from it:

```bash
grep -n "from sglang.srt.managers.io_struct" verl/ -r --include="*.py"
```

For each symbol, verify its class definition, field names, and `__init__` signature in
`sglang-src/python/sglang/srt/managers/io_struct.py`.

### Step 5: weight_sync API

The weight synchronization API (`weight_sync/utils.py`) is veRL's critical path for
updating rollout weights after each training step. Any signature change here will break
the HybridEngine weight sync:

```bash
# Check current usage
grep -n "update_weights\|_preprocess_tensor" verl/workers/rollout/sglang_rollout/sglang_rollout.py
```

Verify against upstream that function signatures and expected tensor formats are
unchanged.

### Step 6: EngineBase Interface

`EngineBase` is subclassed in `http_server_engine.py`. If the base class interface
changes (new abstract methods, changed signatures), the subclass must be updated:

```bash
# Check current subclass
grep -n "class\|def " verl/workers/rollout/sglang_rollout/http_server_engine.py | head -20
```

### Step 7: Update Version Pin

```bash
# Update requirements_sglang.txt
sed -i "s/sglang\[all\]==[0-9.]*/sglang[all]==$VERSION/" requirements_sglang.txt

# Check Docker files that may also pin sglang
grep -rl "sglang" docker/ | grep Dockerfile
```

### Step 8: Run Validation

```bash
# Check imports resolve (CPU, no GPU needed)
python -c "
import sys
sys.path.insert(0, '.')
# Test that sglang can be imported
import sglang
print('sglang version:', sglang.__version__)
"

# Run CPU unit tests
pytest tests/test_*_on_cpu.py -x

# If GPU available, run sglang-specific tests
pytest tests/ -k "sglang" -x
```

---

## Common Breaking Patterns Across SGLang Versions

| Pattern | What to Check |
|---------|---------------|
| `io_struct` field renames | `GenerateReqInput`, `UpdateWeightsFromTensorReqInput` field names change between versions |
| `weight_sync` module reorganization | `update_weights` moved between submodules |
| `EngineBase` new abstract methods | Subclass `http_server_engine.py` may need new methods |
| `ServerArgs` new required fields | New arguments added to `ServerArgs.__init__` |
| `kill_process_tree` location | Moved between `srt.utils` and `srt.utils.common` across versions |
| `ServerStatus` enum values | New status values or renamed values in tokenizer_manager |
| MoE routed experts API | `extract_routed_experts_from_meta_info` signature changes with MoE updates |
| Prometheus middleware location | `add_prometheus_middleware` sometimes moved or renamed |

---

## Output

Report:

1. Current pinned version vs target version
2. Table of changed/broken imports with fix applied
3. `io_struct` fields that changed
4. `weight_sync` API changes if any
5. Files modified
6. Validation commands run and their results

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================
Location: .agents/skills/upgrade-sglang/SKILL.md

## How to Update
- When new sglang imports added to verl: update Affected Files table
- When new common breaking patterns emerge: update the patterns table
- When sglang repo structure changes: update upstream source paths in Step 3
================================================================================
-->
