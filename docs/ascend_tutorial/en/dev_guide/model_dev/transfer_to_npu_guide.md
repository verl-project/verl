# Model Migration to NPU Guide

Last updated: 05/14/2026

This article provides developers with complete practical experience for migrating from GPU to NPU or independently adapting models on NPU. It covers the full workflow, including preliminary preparation, component integration, precision alignment, performance optimization, and long-run evaluation.

## 1. Preliminary Preparation

Set up a basic environment that supports NPU execution. This ensures models load correctly and datasets are read successfully, laying the foundation for subsequent migration, debugging, and business execution.


### 1.1 Software and Hardware Environment and Dependency Configuration

Refer to the official documentation [Ascend Installation Guide](../../get_start/install_guidance.rst); if the versions of the inference engines vllm and vllm_ascend and the training engines Megatron, MindSpeed, and transformers required by the model differ from the tutorial, **use the actual adapted versions of the model**.

### 1.2 Model Weights

BF16 is the **default mixed precision training data type** for training backends such as FSDP and Megatron in the VeRL framework. The Ascend NPU environment uniformly uses **BF16** as the baseline precision format, and weights must be aligned and dequantized to BF16. Currently, A2 and A3 products **do not support FP8 precision training** and only support BF16 precision. Later versions of the Ascend 950 series products will provide FP8 low precision training capability.

### 1.3 Data Preparation

Refer to [Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html) to preprocess the dataset into the parquet format: (1) to ensure it contains the necessary fields for computing reinforcement learning rewards; (2) to achieve a faster reading speed.

## 2. Integration and joint debugging of all components

The VeRL framework adopts a decoupled architecture design that separates the inference engine, training engine, and weight synchronization bridge (Checkpoint Engine). This design enables deep separation of computation and data, providing a flexible extension foundation for migrating and adapting models to Ascend NPU. When performing model migration and adaptation on the NPU, we recommend that you first complete the separate adaptation and verification of each component. These components include the inference engine, training engine, and Megatron-Bridge. After each component runs stably, proceed with the integration and debugging of the end-to-end VeRL pipeline. For specific feature support of different VeRL inference and training backends on Ascend NPU, refer to the [Ascend Backend Feature Guide](../../feature_support/ascend_backend_features.md).

### 2.1 Inference Engine Adaptation

The VeRL inference engine adopts a layered architecture design. Through abstract interfaces and the factory pattern, it provides flexible support for multiple mainstream inference backends such as vllm and sglang. During the migration and adaptation from GPU to NPU, the recommended process for inference engine adaptation is as follows:

Before running the full VeRL pipeline on the NPU, refer to the official model deployment tutorials for [vllm-ascend](https://github.com/vllm-project/vllm-ascend/tree/main/docs/source/tutorials/models) and [sglang](https://github.com/sgl-project/sglang/tree/main/docs/docs/basic_usage). Prioritize getting the **single-instance inference pipeline** working. Fully verify **basic inference functions** such as model loading and initialization, Tokenizer loading, single-turn / batch generation, stop word termination, and long-context inference. After the underlying inference engine is stable and usable, integrate it into the VeRL training process.

### 2.2 Training Engine Selection and Adaptation

The VeRL mainline code abstracts the training engine into the `Engine` class. This decouples the scheduling logic from the underlying training implementation through a standardized interface layer. This architectural design supports the flexible, plug-and-play integration of multiple training backends such as FSDP and Megatron. You do not need to modify the core algorithms and scheduling logic of VeRL, which significantly reduces the migration and adaptation costs.

Currently, the system automatically detects the NPU device through the `is_npu_available` interface and applies the corresponding NPU device adaptation patch. You only need to configure model_engine=fsdp/megatron to switch the training backend to FSDP or Megatron. The system automatically loads the NPU adaptation logic for the corresponding backend, so you do not need to modify the code. Ascend adapts and optimizes Megatron in VeRL. For specific feature configurations, refer to the [verl-MindSpeed feature documentation](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docs/en/user-guide/verl.md).

### 2.3 Megatron-Bridge Adaptation

In the VeRL framework, Megatron-Bridge primarily performs bidirectional conversion between the HuggingFace weights required by the inference engine and the mcore weights required by Megatron-Core. You can enable this feature using the following configuration:

```
actor_rollout_ref.actor.megatron.use_mbridge=True
actor_rollout_ref.actor.megatron.vanilla_mbridge=False
```

Megatron-Bridge has natively adapted a large number of mainstream model architectures in the community. For the supported list, refer to [supported model](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/models/README.md). When performing model migration and adaptation in the Ascend NPU environment, you can complete the basic configuration based on existing community capabilities. However, some special model structures and scenarios still require additional customized adaptation.

This section uses the DSA (DeepSeek Sparse Attention) sparse attention structure as an example to introduce the method for customized adaptation. Ascend MindSpeed supports the DSA capability based on the absorption matrix. This feature requires splitting the original `linear_kv_up_proj` operator in Megatron into two independent operators: `linear_k_up_proj` and `linear_v_up_proj`. The weights required for the split need to be converted from the HuggingFace format `self_attn.kv_b_proj.weight`, but the aforementioned native PR does not adapt to this operator splitting logic.

Therefore, you must manually modify and adapt the weight conversion logic to ensure that the absorption matrix is properly loaded and takes effect. You can enable the [sparse\_flash\_attention](https://gitcode.com/cann/ops-transformer/tree/master/attention/sparse_flash_attention) and [lightning\_indexer](https://gitcode.com/cann/ops-transformer/tree/master/attention/lightning_indexer) fusion operators only when the absorption matrix is available. By introducing these two fusion operators, you can significantly reduce memory access frequency and optimize memory usage. This also improves computational performance, ultimately enhancing the efficiency and reducing the resource overhead of large model training and inference pipelines.

### 2.4 End-to-end network functionality verification

Complete the inference engine adaptation verification and training engine adaptation development. Refer to [Training Configuration Parameters and Metrics](parameter_and_metrics.md) and configure the relevant parameters for the inference engine and training engine based on actual business requirements. Complete the VeRL end-to-end network function integration to ensure stable operation of the entire process.

## 3. Precision Alignment

The pipeline for locating accuracy issues in large language model reinforcement learning is complex and involves many factors. Problems in the **training phase, inference phase, and training-inference consistency** typically introduce various accuracy issues. **Accuracy alignment** is the key to ensuring that the training process is reproducible and issues are debuggable.

For precision alignment during the training and inference phases, refer to the official documentation: [Precision Alignment Guide](../precision_analysis/precision_alignment.md). Therefore, this section does not repeat the basic phase alignment process. Instead, it **focuses on the training-inference consistency scenario** and uses the msprobe precision tool to perform precision alignment practices and troubleshoot issues.

### 3.1 Precision Monitoring Configuration

After the entire network runs successfully, enable the precision monitoring parameter by setting `actor_rollout_ref.rollout.calculate_log_probs=True`. During training, closely monitor the following key metrics to determine training-inference consistency and model training stability:

* **Training and inference consistency reference metrics**:
  * `training/rollout_probs_diff_mean` (mean of rollout probability difference). When the model converges normally, it is recommended to keep this metric within 0.01. If the value remains higher than 0.01 or deviates significantly from the GPU baseline, you can identify a training and inference precision anomaly.
  * `training/rollout_probs_diff_max` (maximum rollout probability difference)
  * `training/rollout_actor_probs_pearson_corr` (Pearson correlation coefficient between rollout and actor probabilities)
* **Model training stability metrics**:
  * `actor/grad_norm`: Check whether it shows an overall downward trend to determine whether the model training converges normally.

In addition, the configuration parameter `trainer.rollout_data_dir=./rollout_dump/` saves the intermediate Rollout results during training. You can manually check the exported Rollout data to verify whether the model replies meet expectations and whether the output contains garbled characters or repeated answers. This further confirms that the inference engine adaptation is correct.

### 3.2 Collecting Precision Data

When training/rollout_probs_diff_mean exceeds the reasonable threshold of 0.01 or deviates significantly from the GPU baseline, use the [msprobe](../../../en/dev_guide/precision_analysis/precision_debugger.md) precision tool to collect data for root cause analysis.

### 3.3 Troubleshooting and Alignment Practices for Training and Inference Differences

After data collection is complete, read the `construction.json` file first to perform module-level data comparison. First, ensure that the input data of `layer.0.input_layernorm` is completely consistent. Then, verify the data module by module and layer by layer to locate where the training and inference outputs first become inconsistent.

For large models, minor numerical differences accumulate and amplify layer by layer. This causes significant differences between training and rollout results. The same token might even have an output probability of 0 during training and 1 during rollout. Therefore, align every difference point to be exactly equal as much as possible.

After locating the difference node, determining the adaptation and modification plan is also a key challenge. Various open-source communities in the industry have multiple different implementations for related modules. To ensure the correctness of the model implementation logic, refer to authoritative source code and technical reports from multiple sources to comprehensively determine the final alignment plan.

#### 3.3.1 Common Training and Inference Inconsistencies

In large language model (LLM) reinforcement learning, you can categorize the typical root causes of training and inference inconsistency into the following five categories:

1. **Framework implementation inconsistency**: This is caused by different implementation logic between training and inference frameworks. Sometimes it is "semantically correct" (for example, different operator splitting methods but mathematically equivalent), and sometimes it is "semantically incorrect" (for example, missing a scaling factor or having an extra operation). You need to strictly verify this by combining the source code and technical reports.
2. **Precision type differences**: For example, the training side uses BF16 throughout, while the inference side implicitly upcasts to FP32 for computation in sensitive operators such as normalization and then downcasts, causing truncation errors.
3. **Hyperparameter inconsistency**: For example, the hardcoded `eps` value in the LayerNorm module is not unified.
4. **Parallelism strategy**: Tensor parallelism during training versus continuous batching during inference causes differences in floating-point accumulation order.
5. **Randomness control**: Implementation deviations of Dropout and sampling strategies between the training and inference phases.

This section lists typical real-world cases of training and inference inconsistencies encountered during the GLM-5 model migration and adaptation process.

#### 3.3.2 Case 1: Inconsistent framework implementation of FFN activation function

Compare from top to bottom, and identify that the output is inconsistent at the MLP activation function of the first layer.

The inference side already uses the NPU-optimized `npu_swiglu` fused operator, but the training side still executes the native GLU small operator implementation.

* **Root cause**: Although the `swiglu` enabling configuration has been added to the VeRL parameters, Megatron-Bridge did not explicitly configure `provider.bias_activation_fusion=True` in the NPU adaptation PR, causing the code to not enter the NPU fused operator branch.
* **Fix**: Add configuration items in Megatron-Bridge so that the training side correctly calls the fused operator:
  ```
  +actor_rollout_ref.actor.megatron.override_transformer_config.swiglu=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True \
  ```

#### 3.3.3 Case 2: Precision of indexer_k_norm is inconsistent with hyperparameters

During the strict alignment process, an inconsistency between the precision type and hyperparameters was found at `indexer_k_norm`:

* **Precision difference**: On the inference side, LayerNorm performs a precision upcast to fp32 using `F.layer_norm( x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)`, while the training-side Megatron implementation uses BF16. Minor differences accumulate across multiple layers and become non-negligible.
* **Fix**: Unify the training-side code to add precision upcast and downcast operations.
* **Hyperparameter difference**: On the GLM5 inference side, vllm inherits the DeepSeek-V3.2 logic, and the EPS value for `k_norm` is hardcoded to `1e-6`. The training engine and the official technical report both use `1e-5`.
* **Fix**: Modify the inference-side EPS to `1e-5` to align with the training side.

```
self.k_norm=LayerNorm(self.head_dim,eps=1e-6 -> 1e-5)
```

#### 3.3.4 Case 3: Missing and redundant logic in the lightning_indexer module

Investigation revealed that the lightning_indexer module has inconsistent implementations between the training and inference sides, specifically:

* **Missing (omission on the inference side)**: The inference side lacks the scaling logic for `weights`. The standard implementations in the Megatron training side, slime, and transformers all include this scaling. Therefore, it is added on the inference side to align the forward pass:

```
weights, _ = self.weights_proj(x)
+weights = weights * (self.n_head**-0.5) * (self.head_dim**-0.5)
```

* **Redundancy (unnecessary and incorrect implementation on the training side)**: The Megatron implementation on the training side includes an extra `rotate_activation` (Hadamard transform). Extensive research confirms that this operation is specifically for quantization scenarios and is incorrectly implemented for the BF16 format. Following [Transformer PR#45017](https://github.com/huggingface/transformers/pull/45017), remove this redundant logic from the training side.

```
class DSAIndexer(MegatronModule):
    def forward_with_scores(
-		q = rotate_activation(q)
-		k = rotate_activation(k)
```

### 3.4 General Routing Stability Solution for Large MoE Models

In a typical RL training process, you use an efficient inference engine such as vLLM to sample data, and then send the sampled data to a training framework such as Megatron to optimize model training.

For regular dense models, implementation and environment differences between inference and training frameworks only cause slight numerical deviations. However, **large-scale MoE models** drastically amplify this problem. The root cause is the MoE dynamic routing mechanism. Minor differences in framework implementation and running environments can assign the same input token to completely different combinations of experts. This leads to entirely different computation paths.

This inconsistency in routing decisions can destabilize the RL training of the MoE model. It causes the "experience" obtained from the inference phase to become completely different for the training phase. This distorts the optimization signal and ultimately leads to catastrophic consequences.

To solve this common problem, the industry introduced the **Routing Replay** mechanism. The core idea is to lock the expert routing path during a specific stage. This prevents minor perturbations from affecting routing decisions and ensures model training stability. Currently, there are two mainstream variants: R2 and R3:

* **(1) Vanilla Routing Replay (R2)**: (This corresponds to `actor_rollout_ref.actor.megatron.router_replay.mode="R2"`, and for VeOmni, it is `actor_rollout_ref.actor.veomni.router_replay.mode="R2"`)

  * **Mechanism**: During the gradient update phase, the expert paths that the training engine calculated in the previous sampling phase are reproduced.
  * **Purpose**: R2 mainly mitigates the impact of **policy staleness** on routing. As the policy updates, the routing calculated in the current forward pass may be inconsistent with the routing used to generate the old data. R2 maintains the coherence of the optimization signal by replaying the old routing.
* **(2) Rollout Routing Replay (R3)**: (corresponds to `actor_rollout_ref.actor.megatron.router_replay.mode="R3"`)

  * **Mechanism**: It captures the routing distribution of the inference engine during sequence generation and directly replays it into the training engine.
  * **Purpose**: It simultaneously addresses the two issues of **training-inference skew** and **policy staleness**. It ensures that the expert path used to calculate the loss during the training phase is absolutely consistent with the expert path during actual inference generation.

Therefore, the Routing Replay mechanism effectively bridges the routing gap between inference and training frameworks, whether using R2 to mitigate outdated strategies or R3 to achieve end-to-end alignment. In the training-inference consistency alignment of **large-scale MoE models**, this mechanism has become a key approach to ensure precision alignment and training stability. Currently, mainstream large models such as DeepSeek-V3.2, GLM-5, and MiMo-V2 have all adopted the Routing Replay technology in R3 mode.

Therefore, for large-scale MoE models, we generally recommend using the more thoroughly aligned R3 mode in actual configurations:

```
actor_rollout_ref.actor.megatron.router_replay.mode="R3" \
actor_rollout_ref.rollout.enable_rollout_routing_replay=True \
```

For the VeOmni backend, use `actor_rollout_ref.actor.veomni.router_replay.mode="R3"` instead. The top-level `actor.router_replay` is removed and no longer takes effect.

## 4. Performance Optimization

When you optimize the training performance of large model RL (reinforcement learning) on the Ascend NPU, refer to the official documentation first for basic configuration tuning: [perf_tuning.rst](https://github.com/verl-project/verl/blob/04833f01/docs/perf/perf_tuning.rst). To achieve more efficient optimization, follow the standardized process of **data collection​​→​bottleneck identification​→configuration tuning→iterative validation**. This process significantly improves the throughput of core stages such as Rollout, Reward, and Update, and effectively reduces resource idling and load imbalance. For specific operations of performance analysis and tuning, strictly refer to the following official guidelines:

1. [Ascend Performance Analysis Guide](../performance/ascend_performance_analysis_guide.md)
2. [Profiling Collection Guide](../performance/ascend_profiling.rst)

### 4.1 Inference Performance Optimization

The rollout phase is the core inference step in large model RL training. Its inference time accounts for the majority of the entire training process. The following are common performance optimization methods for this phase:

1. **Enable graph mode**: Graph mode pre-compiles and optimizes the entire computation graph, enabling deep optimizations such as operator fusion, memory reuse, and constant folding to significantly improve execution efficiency.
2. **Accelerate operator dispatch through CPU core binding**: CPU core binding improves operator dispatch efficiency. Since vllm-ascend v0.18.0rc1, this capability is enabled by default on ARM architecture Ascend servers.
3. **Configure the HCCL communication algorithm to AIV mode**: Set the environment variable `HCCL_OP_EXPANSION_MODE` to `AIV` mode. This specifies that the orchestration and expansion logic of the communication algorithm runs on the Vector Core computing unit on the device side.
4. **Enable asynchronous scheduling**: This eliminates the gap between two consecutive execute_model executions by the Worker. The Worker can directly obtain the completed SchedulerOutput for model inference without blocking to wait for scheduling.

The corresponding configuration parameters are as follows:

```
# Enable graph mode
actor_rollout_ref.rollout.enforce_eager=False
+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[2, 4, 8, 16, 24, 32]"
# CPU binding
++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=True
# Enable asynchronous scheduling
++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=True
```

### 4.2 Training Performance Optimization

The Update phase of large model reinforcement learning training features significant variance in sequence length and high device memory consumption. In addition to basic operator fusion, you need to combine sequence parallelism and device memory-computation tradeoff strategies to overcome the bottleneck. For enabling common training performance optimization features, refer to the [MindSpeed-verl documentation](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/user-guide/verl.md). The core optimization methods include:

1. **Operator fusion**: Enable fused operators such as RoPE, SwiGLU, RMSNorm, and DSA. Operator fusion reduces computing overhead and device memory, improving training efficiency.
2. **Remove padding**: In RL training, response lengths vary, and traditional padding strategies result in significant wasted computation. Enabling Remove padding packs multiple short sequences to fill the Tensor, greatly improving the utilization of NPU computing units (MFU).

## 5. Evaluation and Verification

After training is complete, evaluate and verify the model on the target dataset to ensure the business performance of the migrated model meets the standard. The evaluation steps are the same for different models. The following uses GLM-5 as an example to detail the evaluation process (using the AISBenchmark tool, which supports evaluating multiple inference backends such as vllm/sglang).

The evaluation uses the mathematics dataset aime2025 and the graduate-level professional science dataset gpqa. This verifies that scores improve in the target direction and that catastrophic forgetting does not occur in irrelevant directions.

### 5.1 Installing aisbench

```shell
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark
pip install -e .
```

### 5.2 Download the evaluation dataset

```shell
# On the Linux server, at the tool root path
cd path/to/benchmark/ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime2025.zip
unzip aime2025.zip
rm aime2025.zip
```

### 5.3 Modifying AISBench configuration code to enable vllm/sglang inference evaluation

Open the benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py file. This is the inference evaluation configuration file. Keep the output length `max_out_len` consistent with the `max_response_len` used in training.

```shell
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="/path/to/GLM-5", # Change to the GLM-5 model path
        model="GLM-5",
	    stream=True,
        request_rate = 0,
	    use_timestamp=False,
        max_seq_len=2048,
        retry = 2,
	    api_key="",
        host_ip = "localhost", # IP address of the inference service
        host_port = 12890 , # Port of the inference service
        max_out_len = 8192,  # Maximum output token length
        batch_size=48, # Maximum concurrency for inference
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0,
            seed = 1234,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
```

### 5.4 Starting the Inference Server on Multiple Machines

Refer to the [vllm_ascend GLM5 guide](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.md#multi-node-deployment) to launch the dual-node A3 inference service. Keep `host_port` consistent with the configuration in the previous section. Set `max_model_len` to the sum of `max_prompt_length` and `max_response` used during training.

### 5.5 Starting the vllm evaluation task

Run the following command to start the online inference evaluation task. It calls the deployed vLLM inference backend and loads the corresponding model configuration for automated evaluation:

```
ais_bench --models vllm_api_stream_chat --datasets aime2025_gen_0_shot_chat_prompt
```

After training, the core capability metrics of the model achieve stable improvement: the evaluation scores on the AIME2025 mathematical reasoning dataset steadily increase. Meanwhile, the model also achieves continuous score gains on the GPQA graduate-level professional science dataset. No knowledge degradation or catastrophic forgetting occurs, and the training optimization effect meets expectations.

| Evaluation dataset | GLM5-base | 10step | 15step | 40step | 50step |
| ---------- | --------- | ------ | ------ | ------ | ------ |
| aime2025   | 47.5      | 49.17  | 49.17  | 48.33  | 52.5   |
| gpqa       | 64.65     | 68.81  | 68.43  | 69.07  | 71.21  |

## 6. Summary

This document fully covers the end-to-end practice of migrating large models from GPUs to Ascend NPUs or performing standalone adaptation on NPUs. The workflow consists of five key phases: environment setup, component integration, precision alignment, performance optimization, and evaluation and verification. It provides developers with actionable and reusable operation guides and solutions to common problems.

In the preparation phase, focus on controlling environment dependency versions, model weight precision, and dataset format to lay the foundation for subsequent adaptation. During component integration, follow the principle of verifying individual components before connecting the entire network. Prioritize ensuring the stable adaptation of the inference engine, training engine, and weight conversion tools. For special model structures, complete customized modifications. Precision alignment is the core of migration adaptation. Focus on monitoring training-inference consistency metrics, and resolve common differences in framework implementation and precision types through module-by-module troubleshooting. For MoE models, enable the Routing Replay mechanism to ensure training stability. Performance optimization must follow a standardized process, focusing on the core stages of inference and training. Improve efficiency and reduce resource consumption through graph mode, operator fusion, and other techniques. Finally, verify through standardized evaluation to ensure that the model meets business performance requirements after migration and that no knowledge degradation occurs.

Overall, following this workflow reduces NPU migration and adaptation costs, avoids common pitfalls, and enables large models to run stably and efficiently on Ascend NPUs.
