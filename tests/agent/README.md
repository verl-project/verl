# Agent tests

This directory contains the CPU-only unit tests for the new `verl.agent`
packages introduced for the agent framework / gateway path.

## Naming and CI routing

All executable test modules in this directory use the `*_on_cpu.py` suffix so
they are picked up by VERL's existing `cpu_unit_tests.yml` workflow instead of
the default GPU unit-test workflow.

## Coverage inventory

### Framework

- `framework/test_assembler_on_cpu.py`
  - `test_trajectory_assembler_matches_training_batch_contract`
    - Verifies the assembled `TensorDict` matches the expected training batch contract, including prompt/response padding, masks, `position_ids`, rollout logprobs, routed experts, `rm_scores`, and non-tensor metadata packing.
  - `test_trajectory_assembler_rejects_empty_trajectories`
    - Verifies `assemble()` rejects an empty trajectory list.
  - `test_trajectory_assembler_rejects_response_mask_length_mismatch`
    - Verifies `response_mask` length must match `response_ids`.
  - `test_trajectory_assembler_rejects_response_logprobs_length_mismatch`
    - Verifies `response_logprobs` length must match `response_ids` when logprobs are present.
  - `test_trajectory_assembler_requires_reward_score`
    - Verifies each trajectory must have a non-`None` `reward_score` before assembly.
  - `test_trajectory_assembler_supports_numpy_routed_experts`
    - Verifies `routed_experts` accepts `numpy.ndarray` input and preserves the expected tensor shape/dtype in the output.
- `framework/test_openai_compatible_framework_on_cpu.py`
  - `test_openai_compatible_framework_runs_against_fake_session_runtime`
    - Verifies the framework can run end-to-end against a fake in-memory session runtime and propagate sample-level non-tensor fields into the assembled batch.
  - `test_openai_compatible_framework_waits_for_completion_when_configured`
    - Verifies optional `wait_for_completion()` is invoked with the configured timeout before finalization.
  - `test_openai_compatible_framework_broadcasts_sample_fields_to_multiple_trajectories`
    - Verifies a single sample's non-tensor fields are broadcast to all trajectories materialized from that sample.
  - `test_openai_compatible_framework_aborts_session_on_agent_error`
    - Verifies agent runner failures trigger `abort_session()` and do not finalize the session.
  - `test_openai_compatible_framework_omits_rollout_log_probs_when_missing`
    - Verifies missing rollout logprobs stay absent from the assembled batch instead of being synthesized.

### Gateway

- `gateway/test_gateway_actor_on_cpu.py`
  - `test_normalize_request_context_preserves_structured_fields`
    - Verifies request normalization preserves structured multimodal content, `tool_calls`, and `tool_call_id` fields needed for prefix comparison.
  - `test_gateway_actor_complete_wait_and_finalize`
    - Verifies `/complete`, `wait_for_completion()`, and `finalize_session()` work together on the happy path and attach `reward_info` to materialized trajectories.
  - `test_gateway_actor_prefix_mismatch_splits_trajectories`
    - Verifies a message-history prefix mismatch starts a new trajectory instead of continuing the active one.
  - `test_gateway_actor_tool_context_change_splits_trajectory`
    - Verifies a tool-schema change is treated as a request-context split boundary.
  - `test_gateway_actor_does_not_forward_tools_in_sampling_params`
    - Verifies `tools` are stripped before backend generation params are forwarded.
  - `test_gateway_actor_strips_request_envelope_but_keeps_sampling_params`
    - Verifies request-envelope fields such as `messages`, `model`, and `tools` are removed at the backend boundary while real sampling params still pass through.
  - `test_gateway_actor_continuation_preserves_prompt_and_generation_masks`
    - Verifies continuation tokenization appends mask `0` for replayed/incremental context and mask `1` for newly generated tokens.
  - `test_gateway_actor_tool_argument_format_drift_splits_after_valid_continuation`
    - Verifies after one valid continuation, tool-call argument string drift is treated as a context split at the actor behavior level.
  - `test_gateway_actor_serializes_same_session_concurrent_requests`
    - Verifies concurrent requests targeting the same session are serialized rather than entering the backend concurrently.
  - `test_gateway_actor_rejects_chat_after_complete`
    - Verifies chat requests are rejected once the session has been marked completed.
  - `test_gateway_actor_finalizes_without_complete`
    - Verifies `finalize_session()` can materialize and remove the active trajectory even if `/complete` was never called.
  - `test_gateway_actor_rejects_malformed_requests_with_bad_request`
    - Verifies representative malformed request shapes are rejected with HTTP 400.
  - `test_gateway_actor_backend_failure_does_not_commit_partial_state`
    - Verifies backend generation failure returns HTTP 500 without committing partial trajectory/session state.
  - `test_gateway_actor_backend_failure_after_tool_mismatch_does_not_split`
    - Verifies a failed request after a tool-context mismatch does not prematurely materialize/split the previous trajectory.
  - `test_gateway_actor_tool_call_decode_returns_openai_format`
    - Verifies tool-parser output is decoded back into OpenAI-compatible `tool_calls` responses and can be continued with a tool-result turn.
- `gateway/test_gateway_manager_on_cpu.py`
  - `test_gateway_manager_routes_sessions_stickily`
    - Verifies session creation/finalization stay routed to the owning gateway.
  - `test_gateway_manager_uses_least_active_sessions_routing`
    - Verifies new sessions are assigned to the gateway with the fewest active sessions.
  - `test_gateway_manager_wait_for_completion_delegates_to_session_owner`
    - Verifies `wait_for_completion()` is delegated to the gateway that owns the session.
- `gateway/test_session_runtime_on_cpu.py`
  - `test_gateway_serving_runtime_owns_gateway_lifecycle_and_session_runtime`
    - Verifies the runtime can own gateway actor lifecycle plus session creation, wait, completion, and finalization behavior.
  - `test_gateway_serving_runtime_injects_runtime_owned_gateway_backend`
    - Verifies runtime-owned gateways use the runtime itself as backend and correctly integrate with the mocked load balancer and rollout server.
  - `test_gateway_serving_runtime_releases_server_when_generate_fails`
    - Verifies backend-server slots are still released when `generate()` raises, preventing load-balancer bookkeeping leaks.
  - `test_gateway_serving_runtime_gateway_count_zero_falls_back_to_generate_only_mode`
    - Verifies `gateway_count=0` still supports direct `generate()` requests without creating owned gateway actors or a session runtime.

## Mocking boundaries

- No test in this directory depends on a real `LLMServer`, model weights, or a
  production serving runtime.
- `tests/agent/support.py` provides the fakes and lightweight Ray actors used by
  the gateway/runtime tests.
- The only retained dependency on the old experimental tree is
  `verl.experimental.agent_loop.tool_parser`, which is intentionally reused by
  `GatewayActor` until the community-wide extraction lands.
