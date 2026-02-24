# Router Replay

Router Replay is an advanced routing replay functionality within the Verl framework designed for Mixture of Experts (MoE) models. It enables deterministic training by recording and replaying routing decisions, ensuring consistent model behavior across training runs.


## Key Features

### Multiple Operating Modes
- **`disabled`**: Router replay functionality is completely disabled
- **`R2`**: Standard router replay mode for recording and replaying routing decisions
- **`R3`**: Rollout-specific router replay mode optimized for reinforcement learning workflows

### Core Capabilities
- **Seamless Integration**: Works with reinforcement learning pipelines including PPO
- **Distributed Training Support**: Compatible with multi-GPU and multi-node training environments
- **Flexible Configuration**: Easy to configure via YAML files or command-line parameters

## Configuration

### RouterReplayConfig Parameters

```yaml
router_replay:
  mode: "disabled"  # Available options: disabled, R2, R3
  record_file: null  # Path for recording routing decisions
  replay_file: null   # Path for replaying recorded decisions
```

## Quick Start Guide

### Enabling R2 Mode

#### Configuration File Method
Add the following to your training configuration:

```yaml
actor:
  router_replay:
    mode: "R2"
```

#### Command Line Method
Enable R2 mode via command-line parameters:

```bash
actor_rollout_ref.actor.router_replay.mode="R2"
```

### Enabling R3 Mode

#### Configuration File Method
Configure both actor and rollout settings:

```yaml
# Actor configuration
router_replay:
  mode: "R3"

# Rollout configuration  
enable_rollout_routing_replay: True
```

#### Command Line Method
Enable R3 mode via command-line parameters:

```bash
actor_rollout_ref.actor.router_replay.mode="R3"
actor_rollout_ref.rollout.enable_rollout_routing_replay=True
```

R3 mode requires the rollout backend to support returning router selection results. Currently, this functionality is being tested based on the vllm implementation at https://github.com/vllm-project/vllm/pull/28284 as well as bug fix at https://github.com/vllm-project/vllm/pull/33013 and SGLang implementation at https://github.com/sgl-project/sglang/commit/bed301a5acaa9577c9aa706468bdf242f6a43051.
