import fcntl
import json
import os
import re
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentError
from verl.utils.reward_score.deepsearch.api_call import async_reward_server_call

_MERGED_HOURS: set[tuple[str, str, str]] = set()


def _build_log_paths(output_dir: str, log_type: str, now: datetime) -> tuple[str, str]:
    """Return per-worker shard path and canonical hourly merged path."""
    date_str = now.strftime("%m%d%H")
    worker_tag = f"{socket.gethostname()}_{os.getpid()}"
    shard_filename = f"{log_type}_queries_{date_str}_{worker_tag}.jsonl"
    merged_filename = f"{log_type}_queries_{date_str}.jsonl"
    return os.path.join(output_dir, shard_filename), os.path.join(output_dir, merged_filename)


def _merge_hourly_shards(output_dir: str, log_type: str, hour_str: str) -> None:
    """Merge all worker shard files for a given hour into one file."""
    merge_key = (output_dir, log_type, hour_str)
    if merge_key in _MERGED_HOURS:
        return

    output_path = Path(output_dir)
    shard_paths = sorted(output_path.glob(f"{log_type}_queries_{hour_str}_*.jsonl"))
    if not shard_paths:
        return

    merged_path = output_path / f"{log_type}_queries_{hour_str}.jsonl"
    lock_path = output_path / f".{log_type}_queries_{hour_str}.merge.lock"
    lock_fd = None
    try:
        # Best-effort single merger across workers.
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return
    except Exception as e:
        print(f"[judge_api_client][_merge_hourly_shards] Failed to create merge lock: {e}")
        return

    tmp_path = output_path / f".{log_type}_queries_{hour_str}.{os.getpid()}.tmp"
    try:
        with open(tmp_path, "wb") as fout:
            for shard in shard_paths:
                try:
                    data = shard.read_bytes()
                except Exception as e:
                    print(f"[judge_api_client][_merge_hourly_shards] Failed reading shard {shard}: {e}")
                    continue
                if not data:
                    continue
                fout.write(data)
                if not data.endswith(b"\n"):
                    fout.write(b"\n")
        os.replace(tmp_path, merged_path)
        _MERGED_HOURS.add(merge_key)
        print(f"[judge_api_client][_merge_hourly_shards] Merged {len(shard_paths)} shards into {merged_path}")
        # Delete original shard files after successful merge
        for shard in shard_paths:
            try:
                shard.unlink()
            except Exception as e:
                print(f"[judge_api_client][_merge_hourly_shards] Failed to delete shard {shard}: {e}")
    except Exception as e:
        print(f"[judge_api_client][_merge_hourly_shards] Failed to merge hourly shards: {e}")
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except Exception:
                pass
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass


def _maybe_merge_previous_hour(output_dir: str, log_type: str, now: datetime) -> None:
    prev_hour = (now - timedelta(hours=1)).strftime("%m%d%H")
    _merge_hourly_shards(output_dir, log_type, prev_hour)

def _aggregate_and_cleanup_worker_logs(output_dir: Path, log_type: str, force_current_hour: bool = False) -> None:
    """Merge all worker shard files. Called at training end to flush current hour."""
    now = datetime.now()
    _maybe_merge_previous_hour(str(output_dir), log_type, now)
    if force_current_hour:
        cur_hour = now.strftime("%m%d%H")
        _merge_hourly_shards(str(output_dir), log_type, cur_hour)



def _clean_utf8_data(obj: Any) -> Any:
    """Recursively clean invalid UTF-8 characters from data structures.

    This function ensures all strings in nested dicts/lists are valid UTF-8
    by encoding with errors='ignore' to remove invalid bytes, then decoding back.
    This approach avoids introducing replacement characters (U+FFFD) that could
    cause issues when reading the JSONL file later.

    Also removes unusual line terminators like:
    - Line Separator (LS, U+2028)
    - Paragraph Separator (PS, U+2029)

    Args:
        obj: The data structure to clean (dict, list, str, or other)

    Returns:
        Cleaned data structure with valid UTF-8 strings and normalized line terminators
    """
    if isinstance(obj, str):
        # Encode to bytes with errors='ignore' to remove invalid UTF-8 sequences,
        # then decode back to string. This ensures only valid UTF-8 remains.
        try:
            # Use errors='ignore' to silently remove invalid bytes
            cleaned = obj.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
            # Remove unusual Unicode line terminators (LS and PS)
            # U+2028 = Line Separator, U+2029 = Paragraph Separator
            cleaned = cleaned.replace('\u2028', '\n').replace('\u2029', '\n')
            return cleaned
        except Exception:
            # If encoding fails completely, return empty string
            return ""
    elif isinstance(obj, dict):
        return {key: _clean_utf8_data(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_clean_utf8_data(item) for item in obj]
    else:
        # For other types (int, float, bool, None), return as-is
        return obj


def _messages_from_rollout_non_tensor(messages_raw: Any) -> list[dict[str, Any]] | None:
    """Expects ``{'messages': [...]}`` (one sample from ``non_tensor_batch['messages']``)."""
    if not isinstance(messages_raw, dict):
        return None
    inner = messages_raw.get("messages")
    if not isinstance(inner, list) or not inner:
        return None
    cleaned = _clean_utf8_data(inner)
    if not isinstance(cleaned, list):
        return None
    return cleaned


def _write_case_to_log(
    reward_dict: dict,
    log_type: str,
    error_msg: str = "",
    reward_config: dict = None,
    output_dir: str = None,
    *,
    messages: list[dict[str, Any]] | None = None,
):
    """Write case to log file in JSONL format.

    Args:
        reward_dict: Dictionary containing reward information
        log_type: Type of log ("right" or "error")
        error_msg: Error message if any
        reward_config: Reward configuration dictionary
        output_dir: Base directory for log file. If None, uses reward_log_base_dir from reward_config.
        messages: Optional list of messages to include in the log entry.
    """
    # Get log config from reward_config with default values
    reward_config = reward_config or {}
    enable_logging_str = str(reward_config.get("reward_log_enabled", "True"))

    # Check if logging is enabled (default: True)
    enable_logging = enable_logging_str.lower() in ("true", "1", "yes", "on")
    if not enable_logging:
        return

    os.makedirs(output_dir, exist_ok=True)
    now = datetime.now()
    shard_path, _ = _build_log_paths(output_dir, log_type, now)
    print(f"[judge_api_client][_write_case_to_log] Writing log to shard file: {shard_path}")

    messages = messages or []

    # Decode literal \uXXXX escape sequences in message content to readable Chinese characters
    def decode_unicode_escapes(text):
        if isinstance(text, str):
            return re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), text)
        return text

    decoded_messages = []
    for msg in messages:
        decoded_msg = dict(msg)
        if "content" in decoded_msg:
            decoded_msg["content"] = decode_unicode_escapes(decoded_msg["content"])
        decoded_messages.append(decoded_msg)
    messages = decoded_messages

    # Build log entry as JSON object
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "total_score": reward_dict.get('total_score', 0),
        "format_score": reward_dict.get('format_score', 0),
        "rule_score": reward_dict.get('rule_score', 0),
        "answer_score": reward_dict.get('answer_score', 0),
        "hallucination": reward_dict.get('hallucination', 0),
        "messages": messages,
    }

    # Add error message if present
    if error_msg:
        log_entry["error"] = error_msg

    # Add reward_dict response_dict if available
    if "response_dict" in reward_dict:
        log_entry["response_dict"] = reward_dict["response_dict"]

    # Convert to JSON string
    try:
        log_json = json.dumps(log_entry, ensure_ascii=False)
    except Exception as e:
        print(f"[judge_api_client][_write_case_to_log] Failed to serialize log entry to JSON, skipping: {e}")
        return

    # Write to shard file with lock
    try:
        with open(shard_path, 'a', encoding='utf-8', errors='ignore') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(log_json + "\n")
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        print(f"[judge_api_client][_write_case_to_log] Failed to write log, skipping: {e}")
        return

    _maybe_merge_previous_hour(output_dir, log_type, now)


def _write_failed_case_to_log(
    reward_dict: dict,
    error_msg: str = "",
    reward_config: dict = None,
    output_dir: str = None,
    *,
    messages: list[dict[str, Any]] | None = None,
):
    _write_case_to_log(
        reward_dict, "error", error_msg, reward_config, output_dir, messages=messages
    )


def _write_right_case_to_log(
    reward_dict: dict,
    error_msg: str = "",
    reward_config: dict = None,
    output_dir: str = None,
    *,
    messages: list[dict[str, Any]] | None = None,
):
    _write_case_to_log(
        reward_dict, "right", error_msg, reward_config, output_dir, messages=messages
    )


async def compute_score_server(
    data_source, solution_str, ground_truth, extra_info, reward_router_address, reward_model_tokenizer, tokenizer,   # pyright: ignore[reportUnusedParameter]
):
    """Override base class method to include format validation."""
    # 基于actor rollout的回答和真实答案构造judge model的prompts

    reward_config = extra_info["reward_config"]
    url = reward_config.url
    sampling_mode = reward_config.get("sampling_mode", "single")
    num_samples = reward_config.get("num_samples", 1)
    output_format = reward_config.get("output_format", "dict")
    reward_config_name = reward_config.get("reward_config_name", "deepsearch")
    end_func_max_attempts = reward_config.get("end_func_max_attempts", 10)
    end_func_retry_delay_seconds = reward_config.get("end_func_retry_delay_seconds", 2.0)
    reward_extra_info = {
        **reward_config.get("extra_info", {}),
        **{
            k: v for k, v in extra_info.items()
            if k not in ("reward_config", "prompts", "default_local_dir", "max_assistant_turns",
                         "tool_rewards", "tool_reward_reasons", "env")
        }
    }
    retry_interval = reward_config.get("retry_interval_base", 1)
    max_retry = reward_config.get("max_retry", 5)

    prompt = extra_info["prompts"]
    response = solution_str
    error = AgentError.Success
    error_reason = ""

    # Use structured messages from rollout (contains proper tool_calls field)
    raw_messages = extra_info.get("messages")
    reward_messages = _messages_from_rollout_non_tensor(raw_messages)
    if reward_messages is None:
        raise ValueError(f"[judge_api_client] messages_from_rollout is None, raw_messages={raw_messages}")

    payload = {
        "message_list": reward_messages,
        "extra_info": reward_extra_info,
        "config": {
            "sampling_mode": sampling_mode,
            "num_samples": num_samples,
            "output_format": output_format,
            "reward_config_name": reward_config_name,
            "end_func_max_attempts": end_func_max_attempts,
            "end_func_retry_delay_seconds": end_func_retry_delay_seconds
        }
    }
    headers = {
        "Content-Type": "application/json"
    }

    response_dict = await async_reward_server_call(
        url=url, payload=payload, headers=headers, retry_interval=retry_interval, max_retry=max_retry
    )

    # 设置默认值
    score = 0
    answer_score = 0
    format_score = 0
    rule_score = 0
    hallucination = 0
    cot_tokens = 0
    cot_penalty = 0.0

    if response_dict is None:
        error = error.update(AgentError.RewardCallFail)
        error_reason = "response is empty"
    elif isinstance(response_dict, str):
        # 如果 response_dict 是字符串，尝试解析为字典
        try:
            response_dict = json.loads(response_dict)
        except json.JSONDecodeError:
            error = error.update(AgentError.RewardCallFail)
            error_reason = "fail to parse response_dict as JSON"
            response_dict = None

    # 统一处理字典格式的response_dict（包括原本是字典和从字符串解析出来的字典）
    if response_dict is not None and isinstance(response_dict, dict):
        score_dict = response_dict.get('reward', {})
        sub_scores = score_dict.get("sub_scores", {})
        score = score_dict.get("total", 0)
        format_score = score_dict.get("high_priority_format_score", 0)
        rule_score = score_dict.get("format_score",
                     score_dict.get("medium_priority_format_score", 0))
        answer_score = sub_scores.get("relevance",
                       score_dict.get("relev", 0))
        hallucination = sub_scores.get("hallucination",
                        score_dict.get("hallucination", 0))
        cot_tokens = score_dict.get("cot_tokens", 0)
        cot_penalty = score_dict.get("cot_penalty", 0.0)

    # Logging the case to local files
    tool_rewards = extra_info.get("tool_rewards", []) if extra_info else []
    tool_reward_reasons = extra_info.get("tool_reward_reasons", []) if extra_info else []
    log_reward_dict = {
        "total_score": score,
        "format_score": format_score,
        "rule_score": rule_score,
        "answer_score": answer_score,
        "hallucination": hallucination,
        "tool_rewards": tool_rewards,
        "tool_reward_reasons": tool_reward_reasons,
    }
    if response_dict:
        log_reward_dict["response_dict"] = response_dict

    # Merge env config into reward_config for logging
    if not isinstance(reward_config, dict):
        reward_config = dict(reward_config) if reward_config else {}
    if extra_info and "env" in extra_info:
        env_config = extra_info["env"]
        if "reward_log_base_dir" in env_config:
            reward_config["reward_log_base_dir"] = env_config["reward_log_base_dir"]
        if "reward_log_enabled" in env_config:
            reward_config["reward_log_enabled"] = env_config["reward_log_enabled"]

    output_dir = reward_config.get("reward_log_base_dir")

    # Skip logging for RewardCallFail cases (network errors, empty responses, etc.)
    # These are infrastructure failures, not model quality issues, so they add noise to logs.
    if error != AgentError.RewardCallFail:
        if score > 0:
            _write_right_case_to_log(log_reward_dict, reward_config=reward_config, output_dir=output_dir, messages=reward_messages)
        else:
            _write_failed_case_to_log(log_reward_dict, error_reason, reward_config=reward_config, output_dir=output_dir, messages=reward_messages)

    dict_score = {
        "score": score,
        "error": error,
        "error_reason": error_reason,
        "rewardext_format_score": format_score,
        "rewardext_answer_score": answer_score,
        "rewardext_rule_score": rule_score,
        "hallucination": hallucination,
        "rewardext_cot_tokens": cot_tokens,
        "rewardext_cot_penalty": cot_penalty,
    }
    return dict_score


def parse_solution_str_to_full_messages_with_system(solution_str: str) -> list:
    """将solution_str转换为消息列表格式

    Args:
        solution_str: 已去掉special token的字符串，包含<think>, <tool_call>, <tool_response>, <answer>等标签

    Returns:
        消息列表，格式为[{"role": "assistant", "content": "xxx"}, {"role": "user", "content": "xxx"}]
        包含system、user、assistant和tool四种角色，所有内容都保留在对应的content中
    """
    import re
    messages = []

    # 使用正则表达式分割system、user、assistant和tool角色，要求后面跟换行符
    # 这样可以避免在标签内容中误切分
    split_pattern = r"(system|user|assistant|tool)\n"
    parts = re.split(split_pattern, solution_str)

    current_role = "system"  # 默认第一个是system
    current_content = ""

    for i, part in enumerate(parts):
        if not part.strip():
            continue

        # 检查是否是角色标识
        if part.strip() in ["system", "user", "assistant", "tool"]:
            # 如果之前有内容，先保存
            if current_content.strip():
                messages.append({"role": current_role, "content": current_content.strip()})

            # 设置新角色
            current_role = part.strip()
            current_content = ""
        else:
            # 内容部分，添加到当前内容中
            current_content += part

    # 处理最后的内容
    if current_content.strip():
        messages.append({"role": current_role, "content": current_content.strip()})

    return messages
