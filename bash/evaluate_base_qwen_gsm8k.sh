#!/usr/bin/env bash
set -euo pipefail

# -------- Config --------
REPO_DIR="/local/home/tommaben/workspace"
REPO_URL="https://github.com/QwenLM/Qwen.git"
MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
OUT_FILE="gsm8k_qwen25_3b_instruct_zeroshot.jsonl"

# -------- Prep --------
mkdir -p workspace logs

if [[ -d "${REPO_DIR}/.git" ]]; then
  echo "[INFO] Repo already exists at ${REPO_DIR}, updating..."
  git -C "${REPO_DIR}" fetch --all --prune
  git -C "${REPO_DIR}" reset --hard origin/main || git -C "${REPO_DIR}" reset --hard origin/master
else
  echo "[INFO] Cloning Qwen repo to ${REPO_DIR}..."
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

# -------- Write wrapper (exact content as provided) --------
mkdir -p "${REPO_DIR}/eval"
cat > "${REPO_DIR}/eval/run_qwen25_evaluate_chat_gsm8k.py" <<'PY'
#!/usr/bin/env python3
import runpy
import sys

import torch
from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
from transformers import AutoTokenizer as _AutoTokenizer


def _patch_tokenizer_from_pretrained():
    orig = _AutoTokenizer.from_pretrained

    def wrapped(*args, **kwargs):
        # Qwen repo script passes these, HF tokenizers usually do not accept them
        kwargs.pop("bf16", None)
        kwargs.pop("use_flash_attn", None)
        return orig(*args, **kwargs)

    _AutoTokenizer.from_pretrained = wrapped


def _patch_model_from_pretrained():
    orig = _AutoModelForCausalLM.from_pretrained

    def wrapped(*args, **kwargs):
        model = orig(*args, **kwargs)

        # If the model lacks .chat (common for Qwen2.5 HF models), add a compatible one.
        if not hasattr(model, "chat"):

            def chat(tokenizer, query, history=None, system=None, **gen_kwargs):
                if system is None:
                    system = "You are a helpful assistant."
                messages = [{"role": "system", "content": system},
                            {"role": "user", "content": query}]

                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer([text], return_tensors="pt").to(model.device)

                max_new_tokens = gen_kwargs.pop("max_new_tokens", None)
                if max_new_tokens is None:
                    # Keep this conservative, GSM8K CoT can be long.
                    max_new_tokens = 512

                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        **gen_kwargs,
                    )

                gen_ids = out[0][inputs["input_ids"].shape[1]:]
                resp = tokenizer.decode(gen_ids, skip_special_tokens=True)

                # Maintain the return shape expected by evaluate_chat_gsm8k.py
                new_history = history if history is not None else []
                new_history = list(new_history) + [(query, resp)]
                return resp, new_history

            model.chat = chat

        return model

    _AutoModelForCausalLM.from_pretrained = wrapped


def main():`
    _patch_tokenizer_from_pretrained()
    _patch_model_from_pretrained()

    # Execute the original script unchanged.
    runpy.run_path("eval/evaluate_chat_gsm8k.py", run_name="__main__")


if __name__ == "__main__":
    main()
PY
chmod +x "${REPO_DIR}/eval/run_qwen25_evaluate_chat_gsm8k.py"

# -------- Caches --------
export HF_HOME="${HF_HOME:-$PWD/.hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

# -------- Install deps --------
python -m pip install -U pip
python -m pip install -U "transformers>=4.37.0" accelerate datasets tqdm numpy jsonlines torch
python -m pip install -U hf_transfer

# Avoid torch/torchvision ABI mismatch issues for text-only eval
python -m pip uninstall -y torchvision || true

# Optional speed envs (safe even if ignored)
export HF_HUB_ENABLE_HF_TRANSFER=1

# -------- Run eval --------
cd "${REPO_DIR}"
echo "[INFO] Running GSM8K eval on ${MODEL_ID}"
python eval/run_qwen25_evaluate_chat_gsm8k.py \
  -c "${MODEL_ID}" \
  -o "${OUT_FILE}"

echo "[DONE] Wrote results to: ${REPO_DIR}/${OUT_FILE}"
