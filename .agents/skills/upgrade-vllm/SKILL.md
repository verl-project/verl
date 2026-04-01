---
name: upgrade-vllm
description: Upgrade vLLM in veRL by auditing affected call sites, cross-referencing upstream source, and updating for compatibility.
---

# Upgrade vLLM

Upgrade the pinned vLLM version in veRL and update all affected call sites.

## Usage

```
Target vLLM version: $VERSION
```

`$VERSION`: target vLLM version tag, e.g. `v0.9.0`, `0.8.5`. If omitted, check current
pinned version in `requirements.txt` and validate compatibility.

## Prerequisites — Clone Upstream Source

Cross-reference API signatures against the target version:

```bash
VLLM_DIR="$(git rev-parse --show-toplevel)/vllm-src"
# Validate VERSION to prevent command injection
if [[ ! "$VERSION" =~ ^[a-zA-Z0-9._/-]+$ ]]; then
  echo "Error: Invalid version format"; exit 1
fi
if [ ! -d "$VLLM_DIR" ]; then
  git clone --depth 1 --branch "${VERSION}" https://github.com/vllm-project/vllm.git "$VLLM_DIR"
else
  cd "$VLLM_DIR" && git fetch origin && git checkout "${VERSION}" && cd -
fi
```

If cloning fails, stop and report to the user.

---

## Affected Files in veRL

### Primary — Most Likely to Break

| File | vLLM APIs Used |
|------|----------------|
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | `SamplingParams`, `AsyncEngineArgs`, `build_app`, `init_app_state`, `run_headless`, `TokensPrompt`, `LoRARequest`, `RequestOutput`, `UsageContext`, `AsyncLLM`, `FlexibleArgumentParser`, `FinishReason` |
| `verl/workers/rollout/vllm_rollout/vllm_omni_async_server.py` | Similar to above, multimodal variant |
| `verl/workers/rollout/vllm_rollout/utils.py` | vLLM utility imports |
| `verl/utils/vllm/utils.py` | `LoRAModel`, `LoRARequest`, `get_adapter_absolute_path`, `LRUCacheWorkerLoRAManager`, `PEFTHelper` |
| `verl/utils/vllm/patch.py` | vLLM internal patches |
| `verl/utils/vllm/vllm_fp8_utils.py` | vLLM FP8 internals |
| `verl/workers/sharding_manager/` | vLLM weight resharding hooks |

### Secondary — Config / Version Tracking

| File | Usage |
|------|-------|
| `requirements.txt` | Pinned version: `# vllm==X.Y.Z` |
| `requirements-cuda.txt` | May also pin vllm |
| `docker/Dockerfile.stable.vllm` | Docker image vllm version |

---

## Audit Workflow

### Step 1: Identify Current Pinned Version

```bash
grep -i vllm requirements.txt requirements-cuda.txt 2>/dev/null
```

### Step 2: Enumerate All vLLM Imports

```bash
grep -rn "from vllm\|import vllm" verl/ --include="*.py" | grep -v "__pycache__"
```

For each import, note the module path and symbol name.

### Step 3: Cross-Reference Against Upstream Source

For each imported symbol, verify in `vllm-src/`:

**Key symbols to always check:**

| Symbol | Upstream Source Path |
|--------|----------------------|
| `SamplingParams` | `vllm-src/vllm/sampling_params.py` |
| `AsyncEngineArgs` | `vllm-src/vllm/engine/arg_utils.py` |
| `build_app` | `vllm-src/vllm/entrypoints/openai/api_server.py` |
| `init_app_state` | `vllm-src/vllm/entrypoints/openai/api_server.py` |
| `run_headless` | `vllm-src/vllm/entrypoints/cli/serve.py` |
| `TokensPrompt` | `vllm-src/vllm/inputs/__init__.py` |
| `LoRARequest` | `vllm-src/vllm/lora/request.py` |
| `RequestOutput` | `vllm-src/vllm/outputs.py` |
| `AsyncLLM` | `vllm-src/vllm/v1/engine/async_llm.py` |
| `FinishReason` | `vllm-src/vllm/v1/engine/__init__.py` |
| `FlexibleArgumentParser` | `vllm-src/vllm/utils/argparse_utils.py` or `vllm/utils.py` |
| `LoRAModel` | `vllm-src/vllm/lora/lora_model.py` or `vllm/lora/models.py` |
| `PEFTHelper` | `vllm-src/vllm/lora/peft_helper.py` |
| `LRUCacheWorkerLoRAManager` | `vllm-src/vllm/lora/worker_manager.py` |
| `UsageContext` | `vllm-src/vllm/usage/usage_lib.py` |

For each symbol, check:
- Does it still exist at this path?
- Has the class/function signature changed?
- Has the module been moved or renamed?

### Step 4: Handle try/except Version Fallbacks

veRL already has several try/except blocks for cross-version compatibility, e.g.:

```python
try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
except ImportError:
    from vllm.utils import FlexibleArgumentParser
```

For each such block:
1. Check which branch is correct at `$VERSION`
2. If only one branch is now valid, simplify to a direct import
3. If neither branch is valid, update both

### Step 5: Check Internal Patch Compatibility

`verl/utils/vllm/patch.py` monkey-patches vLLM internals. These are the most fragile:

```bash
cat verl/utils/vllm/patch.py
```

For each patched attribute/method:
- Verify the patched object still exists in `vllm-src/`
- Verify its signature hasn't changed
- If the upstream bug is fixed, consider removing the patch

### Step 6: Update Version Pin

```bash
# Update requirements.txt
sed -i "s/# vllm==.*/# vllm==$VERSION/" requirements.txt

# Update Docker images if applicable
grep -l "vllm" docker/Dockerfile.stable.vllm
```

### Step 7: Run Validation

```bash
# Check imports resolve (CPU, no GPU needed)
python -c "from verl.utils.vllm.utils import *; print('OK')"

# Run CPU unit tests
pytest tests/test_*_on_cpu.py -x

# If GPU available, run vllm-specific tests
pytest tests/ -k "vllm" -x
```

---

## Common Breaking Patterns Across vLLM Versions

| Pattern | What to Check |
|---------|---------------|
| Module reorganization | `from vllm.X import Y` → `from vllm.A.B import Y` |
| `FlexibleArgumentParser` moved | Between `vllm.utils` and `vllm.utils.argparse_utils` |
| `LoRAModel` renamed | Between `lora.lora_model` and `lora.models` |
| `FinishReason` added/moved | In `vllm.v1.engine` |
| `init_app_state` signature change | New required params in OpenAI server |
| `AsyncLLM` API change | `v1` engine interface evolves frequently |
| `SamplingParams` new fields | Often adds new sampling options |

---

## Output

Report:

1. Current pinned version vs target version
2. Table of changed/broken imports with fix applied
3. Patches in `patch.py` that need updating or removal
4. Files modified
5. Validation commands run and their results

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================
Location: .agents/skills/upgrade-vllm/SKILL.md

## How to Update
- When new vllm imports added to verl: update Affected Files table
- When new common breaking patterns emerge: update the patterns table
================================================================================
-->
