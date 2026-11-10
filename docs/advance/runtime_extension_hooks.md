# Proposal: runtime extension hooks for external controllers

Last updated: 11/02/2026

## Motivation

VeRL currently lacks a supported way for an external component to observe runtime events, participate at safe execution points, and request controlled topology changes without subclassing the trainer or accessing private manager state. This proposal adds narrow, versioned, disabled-by-default runtime extension points while VeRL retains authority over trainer progress, Ray resources, rollout membership, and checkpoint receivers.

The hook is a general runtime extension mechanism. Recovery is the first intended consumer because it needs all three capabilities: runtime events, safe-point coordination, and transactional membership changes. Other consumers may use a subset of the same contract for diagnostics, adaptive resource management, validation, or orchestration.

The intended API is:

```python
trainer = create_trainer(config)
registration = trainer.register_runtime_extension(recovery_extension)
try:
    trainer.fit()
finally:
    registration.close()
```

A later factory integration may discover providers through a versioned Python entry-point group. A trainer with no registered extensions must preserve the current execution path.

## Runtime context

VeRL exposes a versioned, capability-based context rather than the trainer object:

```python
@dataclass(frozen=True)
class RuntimeExtensionContext:
    run_id: str
    api_version: int
    capabilities: frozenset[RuntimeCapability]
```

The first implementation grants `events` and `safe_points`. Later membership-controller capabilities can add narrow handles for rollout and checkpoint changes. Those handles will expose supported operations and opaque references; consumers will not receive Ray actor handles, placement groups, vLLM engines, or mutable trainer containers.

## Required hooks

The trainer provides ordered callbacks for `before_fit`, `step_boundary`, `waiting_for_samples`, `after_fit`, and `terminal_failure`. `waiting_for_samples` is emitted from the built-in replay buffers on every unsuccessful poll. This covers synchronous sampling inside `ReplayBuffer.sample()` and asynchronous pre-waiting inside `ReplayBufferAsync.wait_for_sampleable()`.

Each wait event includes trainer mode, global step, partition, required and available group counts, pending and running counts, elapsed wait time, poll interval, and poll count. Replay buffers do not impose a training timeout. A consumer detects a stall by comparing `elapsed_seconds` with its policy threshold and correlating the queue state with rollout health events. This keeps timeout policy outside the generic hook while allowing recovery to run before another step boundary is reached. Custom samplers can provide the same observation contract when they implement their own blocking loop.

A safe-point callback returns `CONTINUE` or `RESTART_REQUIRED`. A consumer may finish synchronous coordination before returning. It cannot mutate optimizer state concurrently with a training step or weight synchronization.

## Events and membership

Rollout adapters publish stable, versioned events such as `replica_failed` and `engine_unhealthy`. Events include run ID, logical recovery unit, generation, evidence level, stable event ID, and physical identities when known. Business errors, tool failures, and a single timeout are not confirmed death.

VeRL owns transactional membership controllers:

```python
rollout.register_membership_controller(controller)
checkpoint.register_receiver_controller(controller)
```

Controllers must support generation fencing, candidate registration, receiver updates, compare-and-swap activation, late-result rejection, and idempotent operation lookup. An RPC timeout never proves that a mutation was rejected.

The membership-controller interfaces are the next implementation stage. This PR implements the generic registry, event bus, trainer lifecycle integration, and capability validation used by those controllers.

## Registration lifecycle

`register_runtime_extension()` validates API versions and requested capabilities, subscribes the consumer to declared events, installs its safe-point callbacks, and returns one closeable registration. Duplicate registration of the same consumer fails before installing callbacks. Consumers request only the capabilities they need; registering an observer does not grant topology mutation authority.

A recovery package consumes this generic API through a small VeRL adapter. The adapter connects its recovery coordinator and infrastructure provider, subscribes to failure and membership events, installs generation-aware request retry, and invokes membership controllers only at permitted safe points. Recovery policy, Journal, audit, and infrastructure integration remain outside VeRL.

VeRL owns the extension registry, capability authorization, framework topology, actor lifecycle operations, membership publication, policy-version evidence, and collective safety. The generic hook does not import or depend on a recovery implementation.

## Non-goals

- Encoding recovery policy in the generic extension registry.
- Hiding PPU/NCCL or TensorDict compatibility changes in extension installation.
- Letting an external controller allocate arbitrary cluster GPUs.
- Retrying a mutation under a new operation ID after timeout.
- Treating before/after-step hooks as sufficient when sample collection is blocked.

## Incremental delivery

1. Add the generic hook registry, event bus, runtime context, capability validation, trainer lifecycle integration, and tests without enabling a consumer. This PR includes this stage.
2. Add rollout and checkpoint membership-controller interfaces and generation-fencing tests.
3. Implement recovery as a consumer and convert the native recovery recipe from a trainer subclass to `register_runtime_extension()`.
4. Optionally add entry-point discovery after the explicit API is stable.

## Acceptance criteria

- Ordinary `PPOTrainerSeparateAsync` can register a runtime extension without subclassing.
- Recovery can consume the extension API without adding recovery-specific imports to VeRL.
- No registered extensions produces no callbacks, event subscriptions, or membership wrappers.
- Duplicate registration, version mismatch, and missing capabilities fail at startup.
- Recovery runs while the trainer is waiting for samples.
- Unknown mutation outcomes retain ownership and are reconciled by operation ID.
- Recovery policy outside the VeRL adapter imports no Ray, vLLM, or trainer implementation.
