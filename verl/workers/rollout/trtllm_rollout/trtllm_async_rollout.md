# Running VeRL with TensorRT-LLM Rollout

We provide initial support for [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) as an asynchronous rollout engine in VERL's reinforcement learning pipeline. It covers key features such as distributed inference with Ray-based orchestration, dynamic weight updates via IPC (Inter-Process Communication), and efficient GPU memory management for GRPO training.

TRT-LLM rollout uses hybrid engine colocate mode, where training and inference workers are colocated on the same GPUs. Memory is managed via `resume()`/`release()` APIs to enable GPU sharing between training and inference workloads.

While the current design factors in multi-node use cases, more extensive multi-node testing and functionality will be delivered in the near future. Current focus is on FSDP and Megatron backend support for Qwen model variants.

---

## 1. Quick Start


```bash
# GRPO with FSDP training engine and TP1
>> bash examples/grpo_trainer/run_qwen2-7b_math_trtllm.sh 1
```

Note that using the TRT-LLM rollout requires setting the following environment variables before launching the Ray cluster, as included in the above script.

```bash
# Clean all SLURM/MPI/PMIx env to avoid pmix mismatch error.
for v in $(env | awk -F= '/^(PMI|PMIX|MPI|OMPI|SLURM)_/{print $1}'); do
    unset "$v"
done
```

## 2. Architecture Design

### 2.1 High-Level Component Diagram

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontSize':'18px', 'edgeLabelBackground':'#eeeeee'}}}%%
flowchart TB
    space1[" "]
    style space1 fill:none,stroke:none
    
    subgraph VERL["<b>VERL Training Pipeline</b>"]
        subgraph Workers["<b>Training Workers</b>"]
            Actor["<b>Actor Worker</b>"]
            Critic["<b>Critic Worker</b>"]
            RefModel["<b>Ref Model Worker</b>"]
        end
        
        Actor -->|<b>Weight Updates<br/>IPC</b>| Rollout["<b>TensorRT-LLM Rollout</b>"]
        
        subgraph RayCluster["<b>Rollout Workers<br/>(Ray Cluster)</b>"]
            space2[" "]
            style space2 fill:none,stroke:none
            
    subgraph AsyncRollout["<b>ServerAdapter<br/>(per DP rank)</b>"]
        DPLeader["<b>• DP Leader coordination</b>"]
        IPCMgmt["<b>• IPC handle management</b>"]
        HTTPAdapter["<b>• HTTP adapter for server communication</b>"]
    end
            
            AsyncRollout -->|<b>HTTP/REST API</b>| HTTPServer
            
            subgraph HTTPServer["<b>TRTLLMHttpServer<br/>(Ray Actor per Replica)</b>"]
                OpenAI["<b>• OpenAI Server wrapper</b>"]
                EngMgmt["<b>• AsyncLLM engine management</b>"]
                MemMgmt["<b>• Memory management (resume/release)</b>"]
            end
            
            HTTPServer --> AsyncLLM
            
            subgraph AsyncLLM["<b>TensorRT-LLM<br/>AsyncLLM Engine</b>"]
                GPUWorkers["<b>• GPU workers (Tensor Parallel)</b>"]
                KVCache["<b>• KV Cache management</b>"]
                CUDAGraph["<b>• CUDA Graph optimization</b>"]
            end
        end
    end
    
    space1 ~~~ VERL
    
    style VERL fill:#e1f5ff
    style RayCluster fill:#fff4e6
    style AsyncRollout fill:#f3e5f5
    style HTTPServer fill:#e8f5e9
    style AsyncLLM fill:#fce4ec
```

### 2.2 Agent Loop Architecture

TRT-LLM rollout follows the same Agent Loop architecture described in the [VERL documentation](https://verl.readthedocs.io/en/latest/advance/agent_loop.html).

With TensorRT-LLM rollout, the AsyncLLM engine runs in the same process as the TRTLLMHttpServer (Ray actor). The engine spawns Ray workers as ModelRunner through Ray's native orchestration with placement groups.

AsyncLLM engine communicates with Ray workers through TensorRT-LLM's internal communication layer. When the server receives a request, it directly calls the AsyncLLM engine to generate response_ids. The Ray workers are separate processes from FSDP/Megatron-LM workers but are co-located on the same GPUs in hybrid engine mode.

The diagram below illustrates TRT-LLM's implementation in hybrid engine mode (Ray Workers and FSDP workers share GPUs):

```mermaid
flowchart TB
    generate[generate]
    
    generate --> Server
    
    Server[TRTLLMHttpServer<br/>AsyncLLM Engine]
    
    Server --> Workers
    
    subgraph Workers["TRT-LLM group (TP4)"]
        direction LR
        subgraph W0[ ]
            RW0[Ray Worker-0]
            F0[FSDP-0]
        end
        subgraph W1[ ]
            RW1[Ray Worker-1]
            F1[FSDP-1]
        end
        subgraph W2[ ]
            RW2[Ray Worker-2]
            F2[FSDP-2]
        end
        subgraph W3[ ]
            RW3[Ray Worker-3]
            F3[FSDP-3]
        end
    end
    
    style Server fill:#ffb6c1
    style RW0 fill:#ffffe0
    style RW1 fill:#ffffe0
    style RW2 fill:#ffffe0
    style RW3 fill:#ffffe0
    style F0 fill:#ffb6c1
    style F1 fill:#ffb6c1
    style F2 fill:#ffb6c1
    style F3 fill:#ffb6c1
    style W0 fill:#d3d3d3
    style W1 fill:#d3d3d3
    style W2 fill:#d3d3d3
    style W3 fill:#d3d3d3
    style Workers fill:#f5f5f5
```


### 2.3 Ray Placement Group Architecture

1. **Placement APIs & GPU Assignment**: TRT-LLM rollout leverages TRT-LLM's Ray-based APIs (`placement_groups`, `placement_bundle_indices`, `per_worker_gpu_share`) to control GPU placement. Each replica (corresponding to one `TRTLLMHttpServer`) is assigned GPU bundles from placement groups based on its replica rank and TP size.

2. **Server Placement**: `TRTLLMHttpServer` is pinned to the same node as its first bundle using `NodeAffinitySchedulingStrategy`, ensuring efficient communication between the HTTP server and its Ray workers.

3. **GPU Sharing**: In hybrid engine mode, training and inference workers share GPUs. Memory is managed via `resume()`/`release()` APIs. The resource pool uses `max_colocate_count=3` internally to support colocation of ActorRollout, RewardModel, and Critic workers.

4. **Multi-Node Design**: The placement group slicing algorithm supports spanning multiple placement groups for multi-node deployments. **Note**: Formal multi-node testing and functionality will be delivered in subsequent MRs.

The following diagram shows an example of TP=4 and DP=2. Replica 0 takes bundles 0-3 and Replica 1 takes bundles 4-7 from the same placement group, with each replica managing TP workers across its assigned bundles:

```mermaid
flowchart TB
    subgraph RayCluster["Ray Cluster Resource Pool"]
        subgraph PG0["Placement Group 0 (Node 0)"]
            B0_0["Bundle 0: GPU 0"]
            B0_1["Bundle 1: GPU 1"]
            B0_2["Bundle 2: GPU 2"]
            B0_3["Bundle 3: GPU 3"]
            B0_4["Bundle 4: GPU 4"]
            B0_5["Bundle 5: GPU 5"]
            B0_6["Bundle 6: GPU 6"]
            B0_7["Bundle 7: GPU 7"]
        end
        
        subgraph PG1["Placement Group 1 (Node 1)"]
            B1_0["Bundle 0: GPU 0"]
            B1_1["Bundle 1: GPU 1"]
            B1_2["Bundle 2: GPU 2"]
            B1_3["Bundle 3: GPU 3"]
            B1_4["Bundle 4: GPU 4"]
            B1_5["Bundle 5: GPU 5"]
            B1_6["Bundle 6: GPU 6"]
            B1_7["Bundle 7: GPU 7"]
        end
        
        PG0 --> Assignment
        PG1 --> Assignment
        
        Assignment["Assigned to TRTLLMReplica"]
        
        Assignment --> Replica0
        Assignment --> Replica1
        
        Replica0["Replica 0<br/>(bundles 0-3 from PG0)<br/>TP=4, DP=2"]
        Replica1["Replica 1<br/>(bundles 4-7 from PG0)<br/>TP=4, DP=2"]
    end
    
    style PG0 fill:#e3f2fd
    style PG1 fill:#e3f2fd
    style Replica0 fill:#c8e6c9
    style Replica1 fill:#c8e6c9
```

---

## 3. Core Components

### 3.1 `TRTLLMHttpServer`

**Purpose**: Ray actor that wraps TensorRT-LLM's AsyncLLM engine and exposes an OpenAI-compatible HTTP API.

**Key Responsibilities**:
- Initialize and manage AsyncLLM engine with placement group constraints
- Wrap AsyncLLM with OpenAIServer to expose HTTP endpoints
- Handle HTTP server lifecycle (launch, shutdown)
- Process generation requests with sampling parameters
- Coordinate memory management (wake_up/sleep) for GPU sharing with training workers


### 3.2 `TRTLLMReplica`

**Purpose**: Manages the mapping between replicas and Ray placement groups, orchestrating server deployment.

**Key Responsibilities**:
- Calculate placement group and bundle index assignments per replica
- Pin TRTLLMHttpServer to specific nodes using NodeAffinitySchedulingStrategy
- Launch and coordinate HTTP servers across distributed nodes
- Validate placement group configurations


### 3.3 `ServerAdapter`

**Purpose**: Rollout worker that handles weight updates, memory management, and generation via HTTP adapter.

Each DP rank has one leader (the first TP rank within that DP group), and that leader coordinates weight updates to the corresponding TRTLLMHttpServer replica.

**Key Responsibilities**:
- Act as DP leader for weight synchronization across exclude_dp mesh
- Convert PyTorch tensors to IPC handles for zero-copy weight updates
- Stream weight updates in chunks to avoid memory exhaustion
- Coordinate resume/release operations for memory management
- Initialize HTTP adapter for server communication


### 3.4 `AsyncTRTLLMHttpAdapter`

**Purpose**: HTTP client for communicating with TRTLLMHttpServer.

**Key Features**:
- Async request handling with retry logic
- Connection pooling for high throughput
- Exponential backoff on failures
- Timeout management

---

## 4. Data Flow Diagrams

### 4.1 Generation Request Flow

```mermaid
sequenceDiagram
    participant Client as Client/Actor
    participant Rollout as ServerAdapter
    participant Adapter as AsyncHttpAdapter
    participant Server as TRTLLMHttpServer
    participant AsyncLLM as AsyncLLM Engine
    
    Client->>Rollout: generate(prompts)
    
    rect rgb(240, 248, 255)
        Note over Rollout: Init adapter if needed
    end
    
    Rollout->>Adapter: POST /v1/completions<br/>{prompt_ids, sampling_params}
    
    rect rgb(255, 250, 240)
        Note over Adapter: Retry loop with backoff
    end
    
    Adapter->>Server: HTTP POST
    
    rect rgb(245, 255, 245)
        Note over Server: Parse request<br/>Validate params
    end
    
    Server->>AsyncLLM: generate_async()
    
    rect rgb(255, 245, 245)
        Note over AsyncLLM: Schedule to execution queue
        Note over AsyncLLM: Run inference (TP workers)<br/>- Forward pass<br/>- Sample tokens<br/>- Update KV cache
    end
    
    AsyncLLM-->>Server: Output (token_ids, log_probs)
    
    Server-->>Adapter: JSON response
    Adapter-->>Rollout: TokenOutput
    Rollout-->>Client: Results
```
