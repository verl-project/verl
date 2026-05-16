# Agent tests

This directory contains CPU-only tests for the `verl.agent` framework and
gateway packages. The suite focuses on behavior that reviewers need to trust:
TransferQueue output schema, multimodal postprocessing, OpenAI-compatible
gateway sessions, session routing, and runtime ownership.

## Naming and CI routing

Executable test modules use the `*_on_cpu.py` suffix so VERL's CPU unit-test
workflow can discover them without pulling GPU-only rollout infrastructure.
Run the current suite with:

```bash
pytest tests/agent/ -q
```

## Coverage inventory

### Framework

- `framework/test_generate_sequences_on_cpu.py`
  - `test_generate_sequences_writes_tq_schema_for_each_session`
    - Verifies `generate_sequences()` runs multiple sessions per prompt and
      writes the TransferQueue key, tag, tensor, nested-tensor, and non-tensor
      schema consumed by sync training.
  - `test_generate_sequences_keeps_successful_sessions_when_one_session_fails`
    - Verifies a failed session is aborted and reported without dropping
      successful sessions for the same prompt.
  - `test_generate_sequences_marks_prompt_failure_when_all_sessions_fail`
    - Verifies all-failed prompts write a failure status and no trajectory
      batch.
  - `test_generate_sequences_omits_rm_scores_when_reward_fn_is_none`
    - Verifies reward-free generation omits `rm_scores` instead of inventing
      scores.
  - `test_generate_sequences_keeps_other_prompts_when_prompt_task_raises`
    - Verifies an unexpected prompt-level exception is counted as one failed
      uid while other prompt tasks still contribute results.
- `framework/test_multi_modal_postprocess_on_cpu.py`
  - `test_compute_multi_modal_inputs_returns_empty_dict_without_processor`
    - Verifies text-only execution produces no multimodal processor inputs.
  - `test_compute_multi_modal_inputs_returns_image_tensors_and_images_seqlens`
    - Verifies processor image outputs drop duplicate text tensors and add
      `images_seqlens`.
  - `test_compute_position_ids_returns_text_shape_without_processor`
    - Verifies text-only position ids keep the standard 2-D shape.
  - `test_compute_position_ids_returns_multimodal_shape_with_processor`
    - Verifies processor-aware position ids include text and vision channels
      and derive `mm_token_type_ids` from image/video token ids.

### Gateway

- `gateway/test_gateway_actor_on_cpu.py`
  - `test_gateway_actor_abort_session_does_not_wait_for_backend_generate`
    - Verifies `abort_session()` can complete while a backend generation is
      still in flight.
  - `test_normalize_request_context_preserves_multimodal_blocks_for_later_extraction`
    - Verifies request normalization preserves multimodal content, `tools`,
      `tool_calls`, and `tool_call_id` fields needed by later gateway stages.
  - `test_gateway_actor_forwards_image_data_on_initial_multimodal_request`
    - Verifies the initial multimodal request extracts image data, forwards it
      to the backend, and materializes it into trajectory metadata.
  - `test_gateway_actor_complete_wait_and_finalize`
    - Verifies `/complete`, `wait_for_completion()`, and `finalize_session()`
      cooperate on the happy path and attach reward info.
  - `test_gateway_actor_continuation_reuses_accumulated_media_context`
    - Verifies continuation turns reuse existing session media without
      re-extracting the original image.
  - `test_gateway_actor_multimodal_reference_change_splits_trajectory`
    - Verifies changing multimodal request context starts a new trajectory.
  - `test_gateway_actor_continuation_with_tool_returned_image_appends_media`
    - Verifies a tool-returned image is appended to accumulated media and
      encoded into the incremental prompt.
  - `test_gateway_actor_prefix_mismatch_splits_trajectories`
    - Verifies message-history prefix mismatch materializes the active
      trajectory and starts the next one.
  - `test_gateway_actor_tool_context_change_splits_trajectory`
    - Verifies tool-schema changes split trajectories.
  - `test_gateway_actor_does_not_forward_tools_in_sampling_params`
    - Verifies `tools` do not leak into backend sampling params.
  - `test_gateway_actor_strips_request_envelope_but_keeps_sampling_params`
    - Verifies backend sampling params come from gateway base params plus
      whitelisted request overrides, not request-envelope fields.
  - `test_gateway_actor_ignores_non_whitelisted_request_sampling_params`
    - Verifies non-whitelisted request sampling fields are ignored.
  - `test_gateway_actor_continuation_preserves_prompt_and_generation_masks`
    - Verifies continuation context uses mask `0` and new model output uses
      mask `1`.
  - `test_gateway_actor_tool_argument_json_equivalence_does_not_split_after_valid_continuation`
    - Verifies JSON-equivalent tool-call argument strings do not split a valid
      continuation.
  - `test_message_prefix_falls_back_to_raw_tool_argument_value_comparison_when_arguments_are_invalid_json`
    - Verifies invalid tool-call argument strings compare by raw value.
  - `test_gateway_actor_serializes_same_session_concurrent_requests`
    - Verifies concurrent requests for one session are serialized before they
      reach the backend.
  - `test_gateway_actor_rejects_chat_after_complete`
    - Verifies chat requests after completion return HTTP 409.
  - `test_gateway_actor_finalizes_without_complete`
    - Verifies finalization can materialize an active trajectory even when
      `/complete` was never called.
  - `test_gateway_actor_rejects_malformed_requests_with_bad_request`
    - Verifies representative malformed OpenAI request shapes return HTTP 400.
  - `test_gateway_actor_backend_failure_does_not_commit_partial_state`
    - Verifies backend failure returns HTTP 500 without committing a partial
      trajectory.
  - `test_gateway_actor_backend_failure_after_tool_mismatch_does_not_split`
    - Verifies a failed split attempt leaves the previous active trajectory
      intact.
  - `test_gateway_actor_tool_call_decode_returns_openai_format`
    - Verifies tool-parser output is decoded into OpenAI-compatible
      `tool_calls` and can be continued with a tool-result turn.
- `gateway/test_gateway_manager_on_cpu.py`
  - `test_gateway_manager_routes_sessions_stickily`
    - Verifies created sessions remain routed to their owning gateway through
      chat and finalization.
  - `test_gateway_manager_uses_least_active_sessions_routing`
    - Verifies new sessions are assigned to the gateway with the fewest active
      sessions and counters are decremented on finalization.
  - `test_gateway_manager_wait_for_completion_delegates_to_session_owner`
    - Verifies completion waits are delegated to the gateway that owns the
      session.
- `gateway/test_session_runtime_on_cpu.py`
  - `test_gateway_serving_runtime_owns_gateway_lifecycle_and_session_runtime`
    - Verifies `GatewayServingRuntime` can own gateway actors, expose the
      session runtime, delegate backend generation through itself, and shut down.
  - `test_gateway_serving_runtime_delegates_generate_to_llm_client`
    - Verifies generate-only mode delegates directly to the supplied LLM client
      when no gateway actors are configured.

## Mocking boundaries

- Real code under test: `verl.agent.framework.*` and `verl.agent.gateway.*`.
  Gateway actor tests also use real Ray actors, FastAPI routing, and HTTPX
  requests against local in-process servers.
- External systems intentionally excluded: real `LLMServer`, model weights,
  GPU rollout engines, recipe submodules, external smoke tests, and trainer
  integration jobs.
- `tests/agent/support.py` provides shared fakes:
  - Tokenization and processors: `FakeTokenizer`, `FakeProcessor`.
  - Multimodal extraction: `fake_vision_info_extractor`,
    `SingleUseVisionInfoExtractor`.
  - Backend behavior: `InspectingBackend`, `InspectingSequencedBackend`,
    `QueuedBackend`, `SlowBackend`, `RecordingLLMClient`,
    `RejectToolsSamplingParamsBackend`, `RejectRequestEnvelopeBackend`,
    `FailingBackend`, `SequencedBackend`, `RejectConcurrentSessionBackend`.
  - Manager/runtime actors: `TrackingGatewayActor`.
- Currently unused support fakes are `NoLogprobBackend`,
  `RecordingLoadBalancer`, `RecordingRolloutServer`, and
  `FailingRolloutServer`; treat them as cleanup candidates unless a follow-up
  test needs them.

## Intentional gaps

- Backend-fatal layering is intentionally not covered here. Risk analysis
  classifies it as P0 follow-up work because current framework/gateway code
  does not yet distinguish backend-fatal failures from recoverable session
  failures.
- `abort_session` backend propagation is intentionally not covered here. The
  current tests only verify gateway-side session cleanup/non-blocking behavior;
  request-level backend abort is P1 follow-up work.
- Framework timeout and health behavior is intentionally not covered here.
  Current code has only optional completion waiting and no health/heartbeat
  contract, so tests for that would describe future code rather than this PR.
