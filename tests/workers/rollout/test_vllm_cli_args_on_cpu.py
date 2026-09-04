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

import json
from types import SimpleNamespace

import pytest

# vllm is not part of the `cpu` extra (it conflicts with the cpu torch world), so
# cpu_unit_tests skips this module; vllm.yml runs it in the vllm venv.
pytest.importorskip("vllm")

from verl.workers.config import RolloutConfig
from verl.workers.rollout.vllm_rollout.utils import (
    _resolve_vllm_weight_sync_local_rank,
    build_cli_args_from_config,
    vLLMColocateWorkerExtension,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    _resolve_rollout_model_path,
    _validate_rollout_model_path,
)


def test_rollout_model_path_defaults_to_actor_checkpoint():
    config = SimpleNamespace(model_path=None)
    model_config = SimpleNamespace(local_path="actor-checkpoint")

    assert _resolve_rollout_model_path(config, model_config) == "actor-checkpoint"


def test_rollout_model_path_can_override_actor_checkpoint():
    config = SimpleNamespace(model_path="rollout-checkpoint")
    model_config = SimpleNamespace(local_path="actor-checkpoint")

    assert _resolve_rollout_model_path(config, model_config) == "rollout-checkpoint"


def test_rollout_model_path_rejects_other_backends():
    with pytest.raises(ValueError, match="only supported by the vLLM backend"):
        RolloutConfig(name="sglang", model_path="rollout-checkpoint")


@pytest.mark.parametrize(
    ("lora_rank", "merge", "load_format", "backend", "message"),
    [
        (0, False, "safetensors", "naive", "adapter-only LoRA"),
        (8, True, "safetensors", "naive", "adapter-only LoRA"),
        (8, False, "dummy", "naive", "non-dummy"),
        (8, False, "safetensors", "nccl", "backend=naive"),
    ],
)
def test_rollout_model_path_rejects_unsafe_weight_sync(lora_rank, merge, load_format, backend, message):
    config = SimpleNamespace(
        model_path="rollout-checkpoint",
        load_format=load_format,
        checkpoint_engine=SimpleNamespace(backend=backend),
    )
    model_config = SimpleNamespace(lora={"rank": lora_rank, "merge": merge}, lora_rank=0)

    with pytest.raises(ValueError, match=message):
        _validate_rollout_model_path(config, model_config)


def test_rollout_model_path_allows_adapter_only_sync():
    config = SimpleNamespace(
        model_path="rollout-checkpoint",
        load_format="safetensors",
        checkpoint_engine=SimpleNamespace(backend="naive"),
    )
    model_config = SimpleNamespace(lora={"rank": 8, "merge": False}, lora_rank=0)

    _validate_rollout_model_path(config, model_config)


class TestBuildCliArgsFromConfig:
    """Tests for CLI argument serialization from config dictionaries."""

    def test_string_value(self):
        """String values become '--key value'."""
        config = {"model": "gpt2"}
        result = build_cli_args_from_config(config)
        assert result == ["--model", "gpt2"]

    def test_integer_value(self):
        """Integer values are converted to strings."""
        config = {"tensor-parallel-size": 4}
        result = build_cli_args_from_config(config)
        assert result == ["--tensor-parallel-size", "4"]

    def test_float_value(self):
        """Float values are converted to strings."""
        config = {"temperature": 0.7}
        result = build_cli_args_from_config(config)
        assert result == ["--temperature", "0.7"]

    def test_bool_true(self):
        """Bool True adds flag without value."""
        config = {"enable-prefix-caching": True}
        result = build_cli_args_from_config(config)
        assert result == ["--enable-prefix-caching"]

    def test_bool_false(self):
        """Optional[bool] args emit '--no-key' for an explicit False."""
        config = {"enable-prefix-caching": False}
        result = build_cli_args_from_config(config)
        assert result == ["--no-enable-prefix-caching"]

    def test_bool_false_plain_bool_omitted(self):
        """Bool False on a plain-bool arg is skipped (parser default is False)."""
        config = {"enforce_eager": False}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_bool_false_union_str_arg_omitted(self):
        """Bool False on a `bool | str | None` arg is skipped (string flag, no --no- form)."""
        config = {"hf_token": False}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_bool_false_underscore_key(self):
        """Underscore keys emit the negative flag in the key's own spelling."""
        config = {"enable_prefix_caching": False}
        result = build_cli_args_from_config(config)
        assert result == ["--no-enable_prefix_caching"]

    def test_bool_false_non_engine_arg_omitted(self):
        """Bool False on args unknown to AsyncEngineArgs is omitted."""
        config = {"disable-log-requests": False}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_none_value(self):
        """None values are skipped."""
        config = {"lora-path": None}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_list_values(self):
        """List values are expanded into multiple arguments."""
        config = {"cudagraph-capture-sizes": [1, 2, 4, 8]}
        result = build_cli_args_from_config(config)
        assert result == ["--cudagraph-capture-sizes", "1", "2", "4", "8"]

    def test_empty_list(self):
        """Empty lists are skipped (vLLM nargs='+' requires at least one value)."""
        config = {"cudagraph-capture-sizes": []}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_list_with_strings(self):
        """List of strings is properly expanded."""
        config = {"allowed-origins": ["http://localhost", "http://example.com"]}
        result = build_cli_args_from_config(config)
        assert result == ["--allowed-origins", "http://localhost", "http://example.com"]

    def test_dict_value(self):
        """Dict values are JSON serialized."""
        config = {"extra-config": {"key": "value", "nested": True}}
        result = build_cli_args_from_config(config)
        assert result[0] == "--extra-config"
        # JSON output may have different key ordering, so parse and compare
        assert json.loads(result[1]) == {"key": "value", "nested": True}

    def test_mixed_config(self):
        """Test a realistic mixed configuration."""
        config = {
            "tensor-parallel-size": 4,
            "enable-prefix-caching": True,
            "disable-log-requests": False,
            "lora-path": None,
            "cudagraph-capture-sizes": [1, 2, 4, 8],
            "max-model-len": 2048,
        }
        result = build_cli_args_from_config(config)

        # Check expected args are present
        assert "--tensor-parallel-size" in result
        assert "4" in result
        assert "--enable-prefix-caching" in result
        assert "--cudagraph-capture-sizes" in result
        assert "1" in result
        assert "8" in result
        assert "--max-model-len" in result
        assert "2048" in result

        # Check skipped values are not present
        assert "--disable-log-requests" not in result
        assert "--lora-path" not in result

    def test_preserves_order(self):
        """Arguments should preserve dictionary order (Python 3.7+)."""
        config = {"first": "a", "second": "b", "third": "c"}
        result = build_cli_args_from_config(config)
        assert result == ["--first", "a", "--second", "b", "--third", "c"]

    def test_empty_config(self):
        """Empty config returns empty list."""
        config = {}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_single_element_list(self):
        """Single element list works correctly."""
        config = {"sizes": [42]}
        result = build_cli_args_from_config(config)
        assert result == ["--sizes", "42"]


class TestCliArgsVllmParserRoundTrip:
    """Serialized args must round-trip through vLLM's serve CLI parser."""

    @staticmethod
    def _build_parser():
        import vllm.entrypoints.cli.serve as serve_mod
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        parser = FlexibleArgumentParser(description="test")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        for cmd in serve_mod.cmd_init():
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
        return parser

    def test_explicit_false_survives_parsing(self):
        """Explicit False survives parse_args and AsyncEngineArgs.from_cli_args."""
        from vllm.engine.arg_utils import AsyncEngineArgs

        parser = self._build_parser()
        config = {
            "skip_tokenizer_init": False,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "enable_sleep_mode": True,
            "enforce_eager": False,
            "disable_log_stats": False,
        }
        argv = ["serve", "dummy-model"] + build_cli_args_from_config(config)
        namespace = parser.parse_args(args=argv)
        engine_args = AsyncEngineArgs.from_cli_args(namespace)
        assert engine_args.enable_prefix_caching is False
        assert engine_args.enable_chunked_prefill is True
        assert engine_args.enable_sleep_mode is True
        assert engine_args.skip_tokenizer_init is False
        assert engine_args.enforce_eager is False
        assert engine_args.disable_log_stats is False


class TestVllmColocateZmqHandle:
    def test_dp_local_rank_offsets_tensor_parallel_rank(self):
        """DP workers on the same node must not reuse the same TP-local socket."""
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_size=4,
            data_parallel_size_local=2,
            data_parallel_rank_local=1,
        )

        assert _resolve_vllm_weight_sync_local_rank(1, parallel_config) == 3
        assert _resolve_vllm_weight_sync_local_rank(3, parallel_config) == 3

    def test_single_dp_keeps_local_rank(self):
        """The old single-DP handle layout remains unchanged."""
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_size=1,
            data_parallel_size_local=1,
            data_parallel_rank_local=0,
        )

        assert _resolve_vllm_weight_sync_local_rank(1, parallel_config) == 1

    def test_uses_global_dp_rank_when_local_rank_is_unset(self):
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_size=4,
            data_parallel_size_local=2,
            data_parallel_rank_local=None,
            data_parallel_rank=3,
        )

        assert _resolve_vllm_weight_sync_local_rank(0, parallel_config) == 2

    def test_zmq_handle_uses_resolved_dp_rank(self, monkeypatch):
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_size=4,
            data_parallel_size_local=2,
            data_parallel_rank_local=1,
        )
        worker = SimpleNamespace(
            local_rank=1,
            model_runner=SimpleNamespace(
                vllm_config=SimpleNamespace(parallel_config=parallel_config),
            ),
        )
        monkeypatch.setenv("VERL_REPLICA_RANK", "2")
        monkeypatch.setenv("VERL_RAY_JOB_ID", "job-123")

        handle = vLLMColocateWorkerExtension._get_zmq_handle(worker)

        assert handle == "ipc:///tmp/rl-colocate-zmq-job-123-replica-2-rank-3.sock"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
