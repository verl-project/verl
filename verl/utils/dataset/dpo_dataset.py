# Copyright 2026 Amazon.com Inc and/or its affiliates
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
Single-turn DPO dataset processing.
"""

import logging
import os
from dataclasses import dataclass, field
from itertools import takewhile
from typing import Any, Optional

import glob
import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.base_config import BaseConfig
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ─── Constants ────────────────────────────────────────────────────────────

DPO_SUPPORTED_TRUNCATION = ("left", "right", "error")
DPO_SUPPORTED_FILE_TYPES = ("*.parquet", "*.jsonl")


# ─── DPO Data Configuration ──────────────────────────────────────────────


@dataclass
class DPODataConfig(BaseConfig):
    """Configuration for the DPO data pipeline.

    Covers the full ``data:`` section of ``dpo_trainer.yaml``.  Inherits from
    BaseConfig to provide a dict-like interface and frozen-field semantics,
    following the same pattern as FSDPEngineConfig / CheckpointConfig.
    """

    # ── Batching / engine knobs (used by trainer, not by DPODataset) ─────
    train_batch_size: int = 256
    micro_batch_size_per_gpu: int = 4
    max_token_len_per_gpu: int = 8192
    use_dynamic_bsz: bool = True
    dataloader_num_workers: int = 8

    # ── File paths and sampling ──────────────────────────────────────────
    train_files: Any = None
    val_files: Any = None
    train_max_samples: int = -1
    val_max_samples: int = -1

    # ── Dataset-level settings (used by DPODataset) ──────────────────────
    prompt_key: str = "prompt"
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"
    max_length: int = 1024
    truncation: str = "error"
    shuffle: bool = True
    seed: int = 42
    remove_invalid_samples: bool = False
    use_shm: bool = False
    chat_template_key: Optional[str] = None
    apply_chat_template_kwargs: dict = field(default_factory=dict)

    # ── Collator / padding ───────────────────────────────────────────────
    pad_mode: str = "no_padding"

    # ── Custom dataset class override ────────────────────────────────────
    custom_cls: Any = field(default_factory=lambda: {"path": None, "name": None})

    def __post_init__(self):
        assert self.truncation in DPO_SUPPORTED_TRUNCATION, (
            f"Expect truncation to be one of {DPO_SUPPORTED_TRUNCATION}. Got '{self.truncation}'"
        )


# ─── Data loading and validation ─────────────────────────────────────────


def load_data_file(data_file: str, use_shm: bool = False) -> pd.DataFrame:
    """
    Copy to local node memory and read a single data file into a pandas DataFrame.

    Uses a try-parse approach: attempts to read as parquet first, then falls
    back to JSON Lines. This makes the function robust to misnamed files
    (e.g., parquet content in a ``.dat`` file).

    Args:
        data_file: Path to a data file (parquet or JSON Lines). Can be a local
            path or a remote path (HDFS) that will be downloaded first.
        use_shm: Whether to copy the file to shared memory (/dev/shm) for
            faster I/O. Useful when the same file is read multiple times.

    Returns:
        pd.DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the local file does not exist after copy.
        ValueError: If the file cannot be parsed as either parquet or JSON Lines.
            The parquet error is chained via ``from`` and the JSON Lines error
            is included in the message text.
    """
    local_file = copy_to_local(data_file, verbose=True, use_shm=use_shm)
    if not os.path.isfile(local_file):
        raise FileNotFoundError(f"Data file not found: '{local_file}'")
    try:
        dataframe = pd.read_parquet(local_file)
    except Exception as parquet_err:
        try:
            dataframe = pd.read_json(local_file, lines=True)
        except Exception as jsonl_err:
            raise ValueError(
                f"Could not parse {local_file} as parquet or JSON Lines "
                f"(supported formats: .parquet, .jsonl).\n"
                f"  JSON Lines error: {jsonl_err}"
            ) from parquet_err
    logger.info(f"Dataset loaded: {len(dataframe)} rows from {local_file}")
    return dataframe


def resolve_data_files(data_files: str | list[str] | ListConfig) -> list[str]:
    """
    Resolve data_files into a flat list of file paths.

    Handles three input types:
      - A single file path (string): returns [path]
      - A directory path (string): scans for all .parquet and .jsonl files inside
      - A list of file/directory paths: expands directories and returns all files

    Args:
        data_files: A file path, directory path, or list of either.

    Returns:
        List of resolved file paths (each ending in .parquet or .jsonl).

    Raises:
        ValueError: If no supported files are found.
    """

    if isinstance(data_files, str):
        data_files = [data_files]
    elif isinstance(data_files, ListConfig):
        data_files = list(data_files)

    resolved = []
    for path in data_files:
        if os.path.isdir(path):
            for ext in DPO_SUPPORTED_FILE_TYPES:
                resolved.extend(sorted(glob.glob(os.path.join(path, ext))))
        else:
            resolved.append(path)

    if not resolved:
        raise ValueError(
            f"No .parquet or .jsonl files found in: {data_files}. "
            f"Provide file paths or a directory containing .parquet/.jsonl files."
        )

    return resolved


def load_and_concatenate_data_files(
    data_files: str | list[str] | ListConfig,
    required_columns: dict[str, str],
    use_shm: bool = False,
) -> pd.DataFrame:
    """
    Load one or more data files, validate columns, and concatenate into a single DataFrame.

    Each file is validated to ensure it contains the required columns. All files
    must have the same required columns (they may have additional columns that
    will be ignored). After validation, the DataFrames are concatenated row-wise.

    Args:
        data_files: A file path, directory path, or list of either. Directories
            are scanned for .parquet and .jsonl files.
        required_columns: Dict mapping config key names to expected column names.
            Example: {"prompt_key": "prompt", "chosen_key": "chosen",
                      "rejected_key": "rejected"}
        use_shm: Whether to use shared memory for file loading.

    Returns:
        A single concatenated pd.DataFrame with all rows from all files.

    Raises:
        ValueError: If no files are found, or any file is missing required columns.
    """
    file_paths = resolve_data_files(data_files)
    logger.info(f"Loading {len(file_paths)} data file(s): {file_paths}")

    dataframes = []
    for file_path in file_paths:
        df = load_data_file(file_path, use_shm=use_shm)
        validate_columns(df, required_columns)
        dataframes.append(df)

    if len(dataframes) == 1:
        combined = dataframes[0]
    else:
        combined = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined {len(dataframes)} files → {len(combined)} total rows")

    return combined


def validate_columns(dataframe: pd.DataFrame, required: dict[str, str]) -> None:
    """
    Verify that all required columns exist in the DataFrame.

    Args:
        dataframe: The loaded DataFrame to validate.
        required: A dict mapping config key names to expected column names.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    available = set(dataframe.columns)
    missing = {k: v for k, v in required.items() if v not in available}
    if missing:
        desc = ", ".join(f"'{col}' (set via data.{key})" for key, col in missing.items())
        raise ValueError(
            f"Dataset is missing required columns: {desc}. "
            f"Available columns: {sorted(available)}. "
            f"To fix: rename your columns or set data.prompt_key / data.chosen_key / data.rejected_key."
        )


def shuffle_and_limit(
    dataframe: pd.DataFrame,
    max_samples: int = -1,
    shuffle: bool = False,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Optionally shuffle the DataFrame and/or limit to a maximum number of rows.

    Args:
        dataframe: The DataFrame to process.
        max_samples: Maximum number of samples to keep. -1 = keep all.
        shuffle: Whether to randomly shuffle the row order.
        seed: Random seed for reproducible shuffling.

    Returns:
        pd.DataFrame with rows optionally shuffled and/or limited.
    """
    total = len(dataframe)

    # Determine how many rows we actually want
    n = max_samples if 0 < max_samples < total else total

    if shuffle:
        # .sample(n=n) handles both the random selection and the limit
        # ignore_index=True is usually best for training data to avoid index collisions
        dataframe = dataframe.sample(n=n, random_state=seed, ignore_index=True)
    elif n < total:
        # Just limit without shuffling
        dataframe = dataframe.iloc[:n]

    if n < total:
        logger.info(f"Selected {n} samples out of {total}")

    return dataframe


# ─── DPO-specific preprocessing functions ────────────────────────────────


def extract_columns(
    dataframe: pd.DataFrame,
    prompt_key: str,
    chosen_key: str,
    rejected_key: str,
) -> tuple[list[str], list[str], list[str]]:
    """
    Extract the prompt, chosen, and rejected columns from a DataFrame as Python lists.

    Args:
        dataframe: The source DataFrame.
        prompt_key: Column name containing the user prompt text.
        chosen_key: Column name containing the preferred (chosen) response.
        rejected_key: Column name containing the dispreferred (rejected) response.

    Returns:
        Tuple of (prompts, chosen_responses, rejected_responses), each a list of strings.
    """
    return (
        dataframe[prompt_key].tolist(),
        dataframe[chosen_key].tolist(),
        dataframe[rejected_key].tolist(),
    )


def inspect_and_filter_invalid_samples(
    prompts: list[str],
    chosen_responses: list[str],
    rejected_responses: list[str],
    remove_invalid: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """Inspect DPO preference pairs for invalid samples and optionally remove them.

    Always scans all samples and logs a summary of any issues found:
      - prompt is None/empty/whitespace
      - chosen response is None/empty/whitespace
      - rejected response is None/empty/whitespace
      - chosen == rejected (no preference signal)

    If remove_invalid=True, the invalid samples are dropped and only valid
    samples are returned. If remove_invalid=False, the warning is logged
    but all samples (including invalid ones) are returned unchanged.

    Args:
        prompts: List of prompt strings.
        chosen_responses: List of chosen response strings.
        rejected_responses: List of rejected response strings.
        remove_invalid: If True, remove invalid samples. If False, only warn.

    Returns:
        Tuple of (prompts, chosen_responses, rejected_responses) — filtered
        if remove_invalid=True, otherwise unchanged.
    """
    total_count = len(prompts)
    valid_indices = []
    null_prompt_count = 0
    null_chosen_count = 0
    null_rejected_count = 0
    identical_count = 0

    for i in range(total_count):
        p = prompts[i]
        c = chosen_responses[i]
        r = rejected_responses[i]

        p_valid = p is not None and isinstance(p, str) and p.strip()
        c_valid = c is not None and isinstance(c, str) and c.strip()
        r_valid = r is not None and isinstance(r, str) and r.strip()

        if not p_valid:
            null_prompt_count += 1
            continue
        if not c_valid:
            null_chosen_count += 1
            continue
        if not r_valid:
            null_rejected_count += 1
            continue
        if c.strip() == r.strip():
            identical_count += 1
            continue

        valid_indices.append(i)

    invalid_count = total_count - len(valid_indices)

    # Always log a summary
    if invalid_count > 0:
        detail_parts = []
        if null_prompt_count > 0:
            detail_parts.append(f"{null_prompt_count} null/empty prompts")
        if null_chosen_count > 0:
            detail_parts.append(f"{null_chosen_count} null/empty chosen responses")
        if null_rejected_count > 0:
            detail_parts.append(f"{null_rejected_count} null/empty rejected responses")
        if identical_count > 0:
            detail_parts.append(f"{identical_count} identical chosen/rejected pairs")
        detail_str = ", ".join(detail_parts)

        if remove_invalid:
            logger.info(
                f"DPO dataset validation: found {invalid_count}/{total_count} invalid samples "
                f"({detail_str}). "
                f"These samples have been removed (data.remove_invalid_samples=true). "
                f"Training will proceed with {len(valid_indices)} valid samples."
            )
        else:
            logger.warning(
                f"DPO dataset validation: found {invalid_count}/{total_count} invalid samples "
                f"({detail_str}). "
                f"Samples with empty or null fields may produce meaningless DPO loss values and "
                f"could negatively affect model quality."
            )
    else:
        logger.info(f"DPO dataset validation: all {total_count} samples are valid")

    # Apply filtering only if requested
    if remove_invalid and invalid_count > 0:
        prompts = [prompts[i] for i in valid_indices]
        chosen_responses = [chosen_responses[i] for i in valid_indices]
        rejected_responses = [rejected_responses[i] for i in valid_indices]

    return prompts, chosen_responses, rejected_responses


def apply_chat_template(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    response: str,
    chat_template_kwargs: dict | None = None,
) -> tuple[str, str]:
    """
    Format a prompt+response pair using the tokenizer's chat template.

    Builds a full conversation ([user prompt] + [assistant response]), applies
    the chat template to get the formatted string, then splits it back into
    the prompt portion and the response portion using longest common prefix
    matching. This ensures the prompt/response boundary is correctly identified
    even when the chat template adds special tokens between turns.

    Args:
        tokenizer: The tokenizer whose chat template will be applied.
        prompt: The raw user prompt text.
        response: The raw assistant response text.
        chat_template_kwargs: Any additional chat template args

    Returns:
        Tuple of (formatted_prompt_str, formatted_response_str) where
        formatted_prompt_str includes all chat template tokens up to the
        assistant generation point, and formatted_response_str is the
        remainder including assistant content and closing tokens.
    """
    if chat_template_kwargs is None:
        chat_template_kwargs = {}
    prompt_chat = [{"role": "user", "content": prompt}]
    prompt_str = tokenizer.apply_chat_template(
        prompt_chat, add_generation_prompt=True, tokenize=False, **chat_template_kwargs
    )
    response_chat = [{"role": "assistant", "content": response}]
    response_str = tokenizer.apply_chat_template(
        prompt_chat + response_chat, tokenize=False, **chat_template_kwargs
    )

    prompt_str = "".join(x for x, _ in takewhile(lambda x: x[0] == x[1], zip(prompt_str, response_str)))
    response_str = response_str[len(prompt_str):]
    return prompt_str, response_str

def tokenize_prompt_response(
    tokenizer: PreTrainedTokenizer,
    prompt_str: str,
    response_str: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize the formatted prompt and response strings into token ID tensors.

    The prompt and response are tokenized separately so we know exactly where
    the prompt ends and the response begins. This boundary is needed to create
    the loss_mask (0 for prompt tokens, 1 for response tokens).

    An EOS token is appended to the response_ids to mark the end of generation.

    Args:
        tokenizer: The tokenizer to use for encoding.
        prompt_str: The chat-template-formatted prompt string.
        response_str: The response string.
    Returns:
        Tuple of (prompt_ids, response_ids), both 1D torch.Tensor of token IDs.
        response_ids includes the appended EOS token.
    """
    prompt_ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    response_ids = tokenizer(response_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    eos_tensor = torch.tensor([tokenizer.eos_token_id], dtype=response_ids.dtype)
    response_ids = torch.cat([response_ids, eos_tensor])

    return prompt_ids, response_ids


# ─── Dataset class ───────────────────────────────────────────────────────


class DPODataset(Dataset):
    """
    Single-turn DPO dataset that produces (chosen, rejected) pairs.

    Loads a .parquet or .jsonl file and tokenizes each sample into paired
    sequences for DPO training. Each sample contains a prompt with two
    responses: one preferred (chosen) and one dispreferred (rejected).

    The dataset applies a fixed preprocessing pipeline in __init__:
        load_and_concatenate_data_files → shuffle_and_limit → extract_columns
        → inspect_and_filter_invalid_samples

    And per-sample processing in __getitem__:
        apply_chat_template → tokenize_prompt_response → build loss_mask → truncate → pack

    Args:
        data_files: Path(s) to .parquet or .jsonl file(s), or a directory.
        tokenizer: A PreTrainedTokenizer instance or a string path to load one.
        config: A DPODataConfig dataclass with data processing options:
            max_length (int): Maximum sequence length per side. Default 1024.
            truncation (str): "left", "right", or "error". Default "error".
            prompt_key (str): Column name for prompt. Default "prompt".
            chosen_key (str): Column name for chosen response. Default "chosen".
            rejected_key (str): Column name for rejected response. Default "rejected".
            remove_invalid_samples (bool): Whether to remove invalid samples. Default False.
            shuffle (bool): Whether to shuffle before limiting to max_samples. Default True.
            seed (int): Random seed for shuffle reproducibility. Default None.
        max_samples: Maximum number of samples to use. -1 for all.

    Returns per __getitem__:
        dict with keys:
            input_ids:       (chosen_len + rejected_len,) — packed sequence
            loss_mask:       (chosen_len + rejected_len,) — 0=prompt, 1=response for each half
            boundary_offset: int — index where rejected starts in the packed sequence
    """

    def __init__(
        self,
        data_files: str | list[str] | ListConfig,
        tokenizer,
        config: DPODataConfig,
        max_samples: int = -1,
    ):
        config = config or DPODataConfig()
        self.max_length = config.max_length
        self.truncation = config.truncation

        self.prompt_key = config.prompt_key
        self.chosen_key = config.chosen_key
        self.rejected_key = config.rejected_key
        self.chat_template_key = config.chat_template_key
        self.apply_chat_template_flag = self.chat_template_key is not None
        self.chat_template_kwargs = config.apply_chat_template_kwargs
        self.remove_invalid_samples = config.remove_invalid_samples
        use_shm = config.use_shm

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        # Preprocessing pipeline: load → validate columns → limit → extract → inspect/filter
        required_columns = {
            "prompt_key": self.prompt_key,
            "chosen_key": self.chosen_key,
            "rejected_key": self.rejected_key,
        }
        dataframe = load_and_concatenate_data_files(data_files, required_columns, use_shm=use_shm)
        dataframe = shuffle_and_limit(
            dataframe,
            max_samples=max_samples,
            shuffle=config.shuffle,
            seed=config.seed,
        )
        self.prompts, self.chosen_responses, self.rejected_responses = extract_columns(
            dataframe, self.prompt_key, self.chosen_key, self.rejected_key
        )

        # Inspect for invalid samples (always logs), optionally remove them
        self.prompts, self.chosen_responses, self.rejected_responses = inspect_and_filter_invalid_samples(
            self.prompts,
            self.chosen_responses,
            self.rejected_responses,
            remove_invalid=self.remove_invalid_samples,
        )

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.prompts)

    def _process_sequence(self, prompt: str, response: str, prefix: str) -> dict[str, torch.Tensor]:
        """
        Run the full per-sample processing pipeline for one prompt+response pair.

        Pipeline: apply_chat_template → tokenize_prompt_response → build loss_mask → truncate

        Args:
            prompt: The raw user prompt text.
            response: The raw assistant response text (chosen or rejected).
            prefix: Key prefix for the returned dict — either "chosen" or "rejected".

        Returns:
            Dict with keys "{prefix}_input_ids" and "{prefix}_loss_mask",
            each a 1D tensor of shape ≤ max_length.
        """
        # Step 1: Format with chat template (only if chat_template_key is set)
        # Guard against None values (possible when remove_invalid_samples=False)
        prompt_str = prompt if isinstance(prompt, str) else ""
        response_str = response if isinstance(response, str) else ""
        if self.apply_chat_template_flag:
            prompt_str, response_str = apply_chat_template(
                self.tokenizer, prompt_str, response_str, self.chat_template_kwargs
            )

        # Step 2: Tokenize
        prompt_ids, response_ids = tokenize_prompt_response(self.tokenizer, prompt_str, response_str)

        # Step 3: Concatenate and create loss_mask
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        loss_mask = torch.zeros_like(input_ids)
        prompt_len = prompt_ids.shape[0]
        loss_mask[prompt_len:] = 1

        # Step 4: Truncate to max_length (no padding — engine uses NestedTensors)
        original_len = input_ids.shape[0]
        if original_len > self.max_length:
            if self.truncation == "error":
                raise ValueError(
                    f"Sequence length {original_len} exceeds max_length {self.max_length} "
                    f"(truncation='error'). Increase data.max_length or set data.truncation to 'left'/'right'."
                )
            elif self.truncation == "right":
                input_ids = input_ids[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]

        return {
            f"{prefix}_input_ids": input_ids,
            f"{prefix}_loss_mask": loss_mask,
        }

    def __getitem__(self, item):
        """
        Get a single DPO training sample as a packed sequence.

        Each sample packs chosen + rejected into ONE sequence:
          input_ids = [chosen_tokens..., rejected_tokens...]
          loss_mask = [chosen_loss_mask..., rejected_loss_mask...]

        Plus a boundary_offset integer marking where chosen ends and rejected begins.

        This ensures the chosen/rejected pair stays together through dynamic
        micro-batching. The micro_batch_transform_fn splits them back into
        separate sequences before the model forward pass.

        Args:
            item: Integer index into the dataset.

        Returns:
            Dict with:
              input_ids: (chosen_len + rejected_len,) — packed sequence
              loss_mask: (chosen_len + rejected_len,) — packed mask
              boundary_offset: int — index where rejected starts
        """
        prompt = self.prompts[item]
        chosen = self.chosen_responses[item]
        rejected = self.rejected_responses[item]

        # Process chosen and rejected separately
        chosen_result = self._process_sequence(prompt, chosen, prefix="chosen")
        rejected_result = self._process_sequence(prompt, rejected, prefix="rejected")

        chosen_ids = chosen_result["chosen_input_ids"]
        chosen_mask = chosen_result["chosen_loss_mask"]
        rejected_ids = rejected_result["rejected_input_ids"]
        rejected_mask = rejected_result["rejected_loss_mask"]

        # Pack into one packed sequence: [chosen_tokens, rejected_tokens]
        boundary_offset = chosen_ids.shape[0]
        packed_input_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
        packed_loss_mask = torch.cat([chosen_mask, rejected_mask], dim=0)

        return {
            "input_ids": packed_input_ids,
            "loss_mask": packed_loss_mask,
            "boundary_offset": int(boundary_offset),
        }
