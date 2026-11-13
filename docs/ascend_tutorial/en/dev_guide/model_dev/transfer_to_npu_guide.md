# Guide to Migrating Models to NPU

Last updated: 05/14/2026

This guide provides developers with complete practical experience for migrating models from GPU to NPU or independently adapting models on NPU, covering the full workflow: preparation, component integration, precision alignment, performance optimization, and long-run evaluation.

## 1. Preparation

Set up a base runtime environment that supports NPU execution, ensuring that the model loads normally and datasets can be read smoothly. This serves as the foundation for subsequent migration debugging and end-to-end business execution.


### 1.1 Software/Hardware Environment and Dependency Configuration

Refer to the official [Ascend Installation Guide](../../get_start/install_guidance.rst). If the versions of the inference engines (vllm, vllm_ascend) and training engines (Megatron, MindSpeed, transformers) that the model depends on differ from those in this tutorial, **use the versions actually adapted for the model**.

### 1.2 Model Weights

BF16 is the **default mixed-precision training data type** for training backends such as FSDP and Megatron in the VeRL framework. The Ascend NPU environment uniformly uses **BF16** as the baseline precision format, and weights need to be aligned and dequantized to BF16. Currently, A2 and A3 models **do not support FP8 precision training** and only support BF16 precision; the Ascend 950PR & 950DT series products will open up FP8 low-precision training capabilities in later versions.

### 1.3 Data Preparation

Following [Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html), preprocess the dataset into parquet format: (1) ensure it contains the necessary fields for computing reinforcement-learning rewards; (2) reading is faster.

## 2. Component Integration

The VeRL framework adopts an architecture that decouples the inference engine, training engine, and weight-synchronization bridge (Checkpoint Engine), enabling deep separation of computation and data, which provides a flexible extension foundation for migrating and adapting models to Ascend NPU. When migrating and adapting models on NPU, it is recommended to first complete the individual adaptation and verification of each component — the inference engine, training engine, and Megatron-Bridge — and then advance to integrating and debugging the full VeRL pipeline once each component runs stably. For the specific feature support of different VeRL inference and training backends on Ascend NPU, refer to the [Ascend Backend Features Guide](../../feature_support/ascend_backend_features.md).

### 2.1 Inference Engine Adaptation

The VeRL inference engine adopts a layered architecture design. Through abstract interfaces and the factory pattern, it flexibly supports multiple mainstream inference backends such as vllm and sglang. During the migration and adaptation from GPU to NPU, the inference engine adaptation is recommended to follow this workflow:

Before running the full VeRL pipeline on NPU, it is recommended to refer to the official model deployment tutorials of [vllm-ascend](https://github.com/vllm-project/vllm-ascend/tree/main/docs/source/tutorials/models) and [sglang](https://github.com/sgl-project/sglang/tree/main/docs/docs/basic_usage), and first get the **single-instance inference pipeline** working. Fully verify **basic inference functionality** such as model loading and initialization, normal tokenizer loading, single-turn/batch generation, stop-token termination, and long-context inference. After the underlying inference engine is stable and usable, integrate it into the VeRL training workflow.

### 2.2 Training Engine Selection and Adaptation

The VeRL mainline code abstracts the training engine as the `Engine` class, decoupling scheduling logic from the underlying training implementation through a standardized interface layer. This architectural design allows multiple training backends such as FSDP and Megatron to be flexibly integrated in a plug-and-play manner without modifying VeRL's core algorithms and scheduling logic, greatly reducing migration and adaptation costs.

Currently, NPU has completed automatic device detection through the `is_npu_available` interface and automatically applies the corresponding NPU device adaptation patches. Now you only need to configure model_engine=fsdp/megatron to switch the training backend to FSDP or Megatron with one click. The system automatically loads the NPU adaptation logic for the corresponding backend without additional code modification. In VeRL, Ascend has adapted and optimized Megatron; for specific feature configuration, refer to the [verl-MindSpeed feature documentation](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/en/user-guide/verl.md).

### 2.3 Megatron-Bridge Adaptation

Megatron-Bridge is mainly used in the VeRL framework to perform bidirectional conversion between the HuggingFace weights that the inference engine depends on and the mcore weights required by Megatron-Core. This feature can be enabled with the following configuration:

```
actor_rollout_ref.actor.megatron.use_mbridge=True
```

Megatron-Bridge has natively adapted a large number of mainstream model structures in the community. For the supported list, refer to [supported model](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/models/README.md). When migrating and adapting models in the Ascend NPU environment, basic configuration can be completed based on existing community capabilities, but some model-specific structures and scenarios still require supplementary customized adaptation.

Taking the DSA (DeepSeek Sparse Attention) sparse attention structure as an example, we introduce the customized adaptation method. Ascend MindSpeed supports DSA capability based on the absorption matrix. This feature requires splitting the original `linear_kv_up_proj` operator in Megatron into two independent operators: `linear_k_up_proj` and `linear_v_up_proj`. The weights required for the split need to be converted from the HuggingFace-format `self_attn.kv_b_proj.weight`, but the aforementioned native PR did not adapt this operator-splitting logic.

Therefore, the relevant weight-conversion logic must be manually adapted to ensure the absorption matrix can be loaded and take effect normally. Only on the basis that the absorption matrix is available can the [sparse\_flash\_attention](https://gitcode.com/cann/ops-transformer/tree/master/attention/sparse_flash_attention) and [lightning\_indexer](https://gitcode.com/cann/ops-transformer/tree/master/attention/lightning_indexer) fusion operators be properly enabled. By introducing these two fusion operators, memory-access frequency can be greatly reduced, memory utilization optimized, and computation performance improved, ultimately improving the operational efficiency of large-model training and inference pipelines while reducing resource overhead.

### 2.4 Full-Pipeline Integration

After completing the inference-engine adaptation verification and training-engine adaptation development, refer to [Training Configuration Parameters and Metrics](parameter_and_metrics.md) to configure the relevant parameters of the inference engine and training engine according to actual business requirements, and complete the full VeRL pipeline integration to ensure the entire process runs stably.

## 3. Precision Alignment

Precision-issue localization for large-model reinforcement learning involves a complex pipeline with many influencing factors. Various precision issues are typically introduced by problems in the **training phase, inference phase, and training-inference consistency**. **Precision alignment** is the core key to ensuring that the training pipeline is reproducible and issues are debuggable.

For precision alignment in the training and inference phases, refer to the official document: [Precision Alignment Guide](../precision_analysis/precision_alignment.md). Therefore, this section does not reiterate the basic phase-alignment workflow, but **focuses on training-inference consistency scenarios**, using the msprobe precision tool to carry out precision-alignment practice and issue localization and troubleshooting.

### 3.1 Precision Monitoring Configuration

After the full pipeline runs through, enable the precision monitoring parameter by setting `actor_rollout_ref.rollout.calculate_log_probs=True`. During training, focus on the following key metrics to judge training-inference consistency and model training stability:

* **Training-inference consistency reference metrics**:
  * `training/rollout_probs_diff_mean` (mean rollout probability difference). Under normal model convergence, this metric is recommended to stay within 0.01; if it consistently exceeds 0.01 or deviates significantly from the GPU baseline, a training-inference precision anomaly can be identified.
  * `training/rollout_probs_diff_max` (max rollout probability difference)
  * `training/rollout_actor_probs_pearson_corr` (Pearson correlation coefficient between rollout and actor probabilities)
* **Model training stability metrics**:
  * `actor/grad_norm`: monitor whether it shows an overall downward trend to judge whether the model is converging normally during training.

In addition, configure the parameter `trainer.rollout_data_dir=./rollout_dump/` to save the intermediate Rollout results during training. By manually inspecting the exported Rollout data, you can verify whether model responses meet expectations and whether there is garbled or repeated output, which further confirms on the surface that the inference engine is adapted correctly.

### 3.2 Collecting Precision Data

When training/rollout_probs_diff_mean exceeds the reasonable threshold of 0.01, or deviates significantly from the GPU baseline, further collect data through the [msprobe](../../../en/dev_guide/precision_analysis/precision_debugger.md) precision tool for root-cause localization.

### 3.3 Training-Inference Discrepancy Investigation and Alignment Practice

After data collection is complete, first read the `construction.json` file for module-level data comparison. Ensure that the input data of `layer.0.input_layernorm` is completely consistent, then verify module by module and layer by layer to locate where the training and inference outputs first diverge.

For large models, tiny numerical differences accumulate and amplify layer by layer, causing the results of training and inference (rollout) to diverge noticeably. It can even happen that the same token has output probabilities of 0 and 1 in training and inference respectively. Therefore, every discrepancy point must be aligned to be exactly equal as much as possible.

After locating the discrepancy node, determining the adaptation and modification plan is also a key challenge. Because various open-source communities have multiple different implementations of related modules, to ensure the correctness of the model implementation logic, you should reference authoritative source code and technical reports from multiple sources and comprehensively determine the final alignment plan.

#### 3.3.1 Common Training-Inference Inconsistencies

In large-model reinforcement-learning practice, the typical root causes of training-inference inconsistency can be categorized into the following five classes:

1. **Framework implementation inconsistencies**: caused by different implementation logic in the training and inference frameworks. Sometimes these are "semantically correct" (for example, different operator-splitting approaches that are mathematically equivalent), and sometimes "semantically incorrect" (for example, a missing scaling factor or an extra operation). Strict verification against source code and technical reports is required.
2. **Precision-type differences**: for example, the training side uses BF16 throughout, while the inference side implicitly upcasts to FP32 for some sensitive operators such as normalization and then downcasts, causing truncation errors.
3. **Hyperparameter inconsistencies**: for example, the hardcoded `eps` value in the LayerNorm module is not unified.
4. **Parallelism strategies**: tensor parallelism during training vs. continuous batching during inference causes differences in floating-point accumulation order.
5. **Randomness control**: implementation deviations of Dropout and sampling strategies between the training and inference phases.

Below are typical real cases of training-inference inconsistency encountered during the GLM-5 model migration and adaptation.

#### 3.3.2 Case 1: Framework Implementation Inconsistency of the FFN Activation Function

Comparing from top to bottom, the output discrepancy was located at the MLP activation function of the first layer.

The inference side already uses the NPU-optimized `npu_swiglu` fusion operator, but the training side still runs the native GLU small-operator implementation.

* **Root cause**: Although the `swiglu` enable configuration has been added to the VeRL parameters, the Megatron-Bridge NPU adaptation PR did not explicitly configure `provider.bias_activation_fusion=True`, so the code did not enter the NPU fusion-operator branch.
* **Fix**: Add the configuration items in Megatron-Bridge so the training side correctly invokes the fusion operator:
  ```
  +actor_rollout_ref.actor.megatron.override_transformer_config.swiglu=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True \
  ```

#### 3.3.3 Case 2: Precision and Hyperparameter Inconsistency of indexer_k_norm

During strict alignment, a precision-type and hyperparameter inconsistency was found at `indexer_k_norm`:

* **Precision difference**: the inference side upcasts to fp32 in LayerNorm with `F.layer_norm( x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)`, while the training-side Megatron implementation uses BF16. Tiny differences become non-negligible after accumulating across many layers.
* **Fix**: Unify the training-side code to add the upcast/downcast operations.
* **Hyperparameter difference**: on the GLM5 inference side, vllm inherits the DeepSeek-V3.2 logic, and the EPS value of `k_norm` is hardcoded to `1e-6`; whereas the training engine and the official technical report uniformly use `1e-5`.
* **Fix**: Modify the inference-side EPS to `1e-5` to align with the training side.

```
self.k_norm=LayerNorm(self.head_dim,eps=1e-6 -> 1e-5)
```

#### 3.3.4 Case 3: Missing and Redundant Logic in the lightning_indexer Module

Investigation found that the lightning_indexer module has inconsistent implementations between the training and inference sides, specifically:

* **Missing (inference-side omission)**: the inference side is missing the scaling logic for `weights`. Referring to the standard implementations in the Megatron training side, slime, and transformers, all include this scaling, so it was added on the inference side to align the forward pass:

```
weights, _ = self.weights_proj(x)
+weights = weights * (self.n_head**-0.5) * (self.head_dim**-0.5)
```

* **Redundant (extra erroneous implementation on the training side)**: the training-side Megatron implementation has an extra `rotate_activation` (Hadamard transform). After consulting extensive materials, this operation was confirmed to be dedicated to quantization scenarios and is an erroneous implementation in BF16 format. Following [Transformer PR#45017](https://github.com/huggingface/transformers/pull/45017), this redundant logic was removed on the training side.

```
class DSAIndexer(MegatronModule):
    def forward_with_scores(
-		q = rotate_activation(q)
-		k = rotate_activation(k)
```

### 3.4 General Routing Stability Solution for Large MoE Models

In a typical RL training pipeline, efficient inference engines such as vLLM are usually used to sample data, and the sampled data is then fed into training frameworks such as Megatron for model training and optimization.

For conventional dense models, implementation and environment differences between the inference and training frameworks only cause minor numerical deviations; but for **large MoE models**, this issue is drastically amplified. The core root cause lies in the MoE dynamic routing mechanism: tiny framework-implementation and runtime-environment differences can cause the same input token to be assigned to completely different expert combinations, thereby following entirely different computation paths.

Such routing-decision inconsistencies may cause unstable RL training of MoE models. It makes the "experience" obtained from the inference phase completely different for the training phase, distorting the optimization signal and ultimately leading to catastrophic consequences.

To solve this general problem, the industry has introduced the **Routing Replay** mechanism. Its core idea is to lock the expert routing paths of a specific phase, shielding routing decisions from the interference of minor perturbations, thereby ensuring the stability of model training. Currently, there are two mainstream variants, R2 and R3:

* **(1) Vanilla Routing Replay (R2)**: (corresponds to `actor_rollout_ref.actor.megatron.router_replay.mode="R2"`, or `actor_rollout_ref.actor.veomni.router_replay.mode="R2"` for VeOmni)
  
  * **Mechanism**: In the gradient-update phase, it reproduces the expert paths computed by the training engine in the previous sampling phase.
  * **Effect**: mainly mitigates the impact of **policy staleness** on routing. As the policy is updated, the routing computed in the current forward pass may differ from the routing when the old data was generated; R2 maintains the coherence of the optimization signal by replaying the old routing.
* **(2) Rollout Routing Replay (R3)**: (corresponds to `actor_rollout_ref.actor.megatron.router_replay.mode="R3"`)
  
  * **Mechanism**: During sequence generation, it captures the routing distribution of the inference engine and replays it directly into the training engine.
  * **Effect**: simultaneously addresses both **training-inference divergence** and **policy staleness**. It ensures that the expert paths used to compute the loss in the training phase are absolutely consistent with the expert paths used when the inference results were actually generated.

Therefore, whether it is R2, which focuses on mitigating policy staleness, or R3, which achieves full-pipeline alignment, the Routing Replay mechanism effectively bridges the routing gap between the inference and training frameworks. In the training-inference consistency alignment of **large MoE models**, this mechanism has become a core means of ensuring precision alignment and training stability. Currently, mainstream large models such as DeepSeek-V3.2, GLM-5, and MiMo-V2 have all adopted the R3-mode Routing Replay technique.

Therefore, for large MoE models, the R3 mode, which provides more thorough alignment, is generally recommended in actual configuration:

```
actor_rollout_ref.actor.megatron.router_replay.mode="R3" \
actor_rollout_ref.rollout.enable_rollout_routing_replay=True \
```

For the VeOmni backend, use `actor_rollout_ref.actor.veomni.router_replay.mode="R3"` instead. The top-level `actor.router_replay` has been removed and is no longer effective.

## 4. Performance Optimization

When optimizing large-model RL (reinforcement learning) training performance on Ascend NPU, basic configuration tuning can first refer to the official document: [perf_tuning.rst](https://github.com/verl-project/verl/blob/04833f01/docs/perf/perf_tuning.rst). For more efficient optimization, it is recommended to follow the standardized workflow of **data collection -> bottleneck localization -> configuration tuning -> iterative verification**. This workflow can significantly improve the throughput of core phases such as Rollout, Reward, and Update, while effectively reducing resource idling and load imbalance. For the specific operations of performance analysis and tuning, strictly follow the official guides below:

1. [Ascend Performance Analysis Guide](../performance/ascend_performance_analysis_guide.md)
2. [Profiling Collection Guide](../performance/ascend_profiling.rst)

### 4.1 Inference Performance Optimization

The Rollout phase is the core inference stage of large-model RL training, and its inference time occupies the vast majority of the entire training pipeline. The following are common performance optimization methods for this phase:

1. **Enable graph mode**: Graph mode compiles and optimizes the entire computation graph in advance, enabling deep optimizations such as operator fusion, memory reuse, and constant folding, which significantly improves execution efficiency.
2. **CPU binding to accelerate operator dispatch**: CPU binding can improve operator-dispatch efficiency. Since vllm-ascend v0.18.0rc1, this capability is enabled by default on ARM-architecture Ascend servers.
3. **Configure the HCCL communication algorithm to AIV mode**: Set the environment variable `HCCL_OP_EXPANSION_MODE` to `AIV` mode to specify that the orchestration and expansion logic of the communication algorithm runs on the Device-side Vector Core compute unit.
4. **Enable async scheduling**: This eliminates the gap between two consecutive execute_model executions by the Worker, allowing the Worker to directly fetch the already-scheduled SchedulerOutput for model inference without blocking and waiting for scheduling.

The corresponding configuration parameters are as follows:

```
# Enable graph mode
actor_rollout_ref.rollout.enforce_eager=False
+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[2, 4, 8, 16, 24, 32]"
# CPU core binding
++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=True
# Enable async scheduling
++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=True
```

### 4.2 Training Performance Optimization

The Update phase of large-model RL training has characteristics such as large sequence-length differences and high memory consumption. In addition to basic operator fusion, it also requires combining sequence parallelism and memory-computation trade-off strategies to break through bottlenecks. Common training performance-optimization features can be enabled by referring to the [MindSpeed-verl documentation](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/en/user-guide/verl.md). The core optimization methods include:

1. **Operator fusion**: Enable fusion operators such as RoPE, SwiGLU, RMSNorm, and DSA. Operator fusion reduces computation overhead and memory usage, improving training efficiency.
2. **Remove padding**: In RL training, Response lengths vary widely, and the traditional Padding strategy leads to a large amount of invalid computation. Enabling Remove padding packs multiple short sequences to fill the Tensor, greatly improving the utilization (MFU) of the NPU compute units.

## 5. Evaluation and Verification

After training is complete, evaluation and verification must be performed on the target dataset to ensure the model's business performance after migration meets the standard. The evaluation steps are the same for different models. The following uses GLM-5 as an example to detail the evaluation workflow (using the AISBenchmark tool, which supports evaluation with multiple inference backends such as vllm/sglang).

The evaluation uses the math dataset aime2025 and the graduate-level professional science dataset gpqa, verifying that scores improve in the target direction while no catastrophic knowledge forgetting occurs in unrelated directions.

### 5.1 Install aisbench

```shell
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark
pip install -e .
```

### 5.2 Download the Evaluation Dataset

```shell
# Inside the Linux server, at the tool root path
cd path/to/benchmark/ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime2025.zip
unzip aime2025.zip
rm aime2025.zip
```

### 5.3 Modify the AISBench Configuration Code to Enable vllm/sglang Inference Evaluation

Open the file benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py. This is the inference evaluation configuration file. The output length `max_out_len` is recommended to match the training `max_response_len`.

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
        host_ip = "localhost", # IP of the inference service
        host_port = 12890 , # Port of the inference service
        max_out_len = 8192,  # Maximum output tokens length
        batch_size=48, # Maximum inference concurrency
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0,
            seed = 1234,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
```

### 5.4 Launch the Multi-Node Inference Server

Refer to the [vllm_ascend GLM5 Guide](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.md#multi-node-deployment) to launch the two-node A3 inference service. `host_port` should be consistent with the configuration in the previous section, and `max_model_len` should be set to the sum of `max_prompt_length` and `max_response` during training.

### 5.5 Launch the vLLM Evaluation Task

Run the following command to launch the online inference evaluation task, which invokes the deployed vLLM inference backend and loads the corresponding model configuration to complete the automated evaluation:

```
ais_bench --models vllm_api_stream_chat --datasets aime2025_gen_0_shot_chat_prompt
```

After training, the model's core capability metrics achieved stable improvement: the evaluation score on the AIME2025 math-reasoning dataset rose steadily, and continuous score gains were also achieved on the GPQA graduate-level professional science dataset, with no knowledge degradation or catastrophic forgetting. The training optimization effect met expectations.

| Evaluation Dataset | GLM5-base | 10step | 15step | 40step | 50step |
| ---------- | --------- | ------ | ------ | ------ | ------ |
| aime2025   | 47.5      | 49.17  | 49.17  | 48.33  | 52.5   |
| gpqa       | 64.65     | 68.81  | 68.43  | 69.07  | 71.21  |

## 6. Summary

This guide comprehensively covers the full workflow practice of migrating large models from GPU to Ascend NPU or independently adapting them on NPU. It is divided into five key phases: environment setup, component integration, precision alignment, performance optimization, and evaluation and verification, providing developers with actionable, reusable operation guides and issue-resolution solutions.

In the preparation phase, focus on controlling environment-dependency versions, model-weight precision, and dataset format to lay the foundation for subsequent adaptation. In the component-integration phase, follow the principle of verifying single components first before full-pipeline integration, prioritizing the stable adaptation of the inference engine, training engine, and weight-conversion tool, and completing customized modifications for special model structures. Precision alignment is the core of migration and adaptation; focus on monitoring training-inference consistency metrics, and resolve common discrepancies such as framework implementation and precision types through module-by-module investigation. MoE models need the Routing Replay mechanism enabled to ensure training stability. Performance optimization should follow the standardized workflow, focusing on the core inference and training phases, and improving efficiency and reducing resource consumption through graph mode and operator fusion. Finally, standardized evaluation and verification ensures that the model's business performance meets the standard after migration, with no knowledge degradation.

Overall, following the workflow in this guide can effectively reduce the cost of NPU migration and adaptation, avoid common pitfalls, and achieve stable, efficient operation of large models on Ascend NPU.
