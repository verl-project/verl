# Transfer to NPU guide

Last updated: 05/14/2026

本文为开发者提供从 GPU 迁移至 NPU或在 NPU 上独立适配模型的完整实践经验，涵盖前期准备、各组件打通、精度对齐、性能优化及长跑评测全流程。

## 一、前期准备

搭建可支持 NPU 运行的基础运行环境，保证模型正常加载、数据集可顺利读取，作为后续迁移调试、业务跑通的基础。


### 1.1 软硬件环境与依赖配置

参照官方文档[install\_guidance.rst](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/get_start/install_guidance.rst)；若模型依赖的推理引擎 vllm、vllm_ascend 和训练引擎Megatron、MindSpeed、transformers 版本与教程存在差异，**以模型实际适配版本为准**。

### 1.2 模型权重

BF16 为 VeRL 框架中 FSDP 与 Megatron 等训练后端**默认混合精度训练数据类型**。昇腾 NPU 环境统一采用 **BF16** 作为基准精度格式，权重需对齐反量化为 BF16。目前 A2、A3 机型**暂不支持 FP8 精度训练**，仅支持 BF16 精度；A5 机型后续版本将开放 FP8 低精度训练能力。

### 1.3 数据准备

数据需参照[Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)将数据集预处理为 parquet 格式：(1) 确保它包含计算强化学习奖励所需的必要字段；(2) 读取速度更快。

## 二、各组件联调打通

VeRL 框架采用推理引擎、训练引擎与权重同步桥接（Checkpoint Engine）相解耦的架构设计，可实现计算与数据的深度分离，为模型向昇腾 NPU 迁移适配提供了灵活的扩展基础。在开展模型在NPU上的迁移与适配工作时，建议优先完成推理引擎、训练引擎、Megatron-Bridge 各组件的单独适配与验证，待各组件运行稳定后，再推进 VeRL 整网链路的打通与调试。关于 VeRL 不同推理、训练后端在昇腾 NPU 上的具体特性支持，可参考[昇腾特性文档](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/feature_support/ascend_backend_features.md)。

### 2.1 推理引擎适配

VeRL 推理引擎采用分层架构设计，通过抽象接口与工厂模式，实现 vllm、sglang 等多种主流推理后端的灵活支持。在完成 GPU 向 NPU 的迁移适配过程中，推理引擎适配推荐按以下流程操作：

在 NPU 上跑通 VeRL 整网链路前，建议参考 [vllm-ascend](https://github.com/vllm-project/vllm-ascend/tree/main/docs/source/tutorials/models)、[sglang](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/basic_usage) 官方模型部署教程，优先调通**单实例推理链路**，完整验证模型加载与初始化、Tokenizer 加载正常、单轮 / 批量生成、停止词终止、长上下文推理等**基础推理功能**，前置底层推理引擎稳定可用后，再接入 VeRL 训练流程。

### 2.2 训练引擎选择与适配

VeRL 主线代码将训练引擎抽象为 `Engine`类，通过标准化接口层实现调度逻辑与底层训练实现的解耦。该架构设计支持 FSDP、Megatron、MindSpeed-LLM 等多种训练后端灵活接入、即插即用，无需修改 VeRL 核心算法与调度逻辑，大幅降低迁移适配成本。

当前 NPU 已通过 `is_npu_available` 接口完成设备自动检测，并自动应用对应的 NPU 设备适配补丁。目前只需通过配置 model_engine=fsdp/megatron，即可一键切换训练后端至 FSDP、Megatron，系统会自动加载对应后端的 NPU 适配逻辑，无需额外修改代码。VeRL中昇腾对Megatron做了适配与优化，具体特性配置参考[verl-MindSpeed特性文档](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/user-guide/verl.md)设置。

### 2.3 Megatron-Bridge适配

Megatron-Bridge 主要用于在 VeRL 框架下，完成推理引擎依赖的 HuggingFace 权重与 Megatron-Core 所需 mcore 权重的双向转换，可通过以下配置启用该功能：

```
actor_rollout_ref.actor.megatron.use_mbridge=True
actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
```

Megatron-Bridge已在社区原生适配大量主流模型结构，支持列表可参考：[supported model](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/models/README.md)，在昇腾 NPU 环境开展模型迁移适配时，可基于社区现有能力完成基础配置，但仍有部分模型特殊结构与场景需要补充定制化适配。

以​DSA （DeepSeek Sparse Attention）稀疏注意力结构为示例，介绍定制化适配的方法。昇腾 MindSpeed 支持基于吸收矩阵的 DSA能力，该特性要求将 Megatron 中原有的 `linear_kv_up_proj` 算子拆分为 `linear_k_up_proj` 与 `linear_v_up_proj` 两个独立算子。拆分所需权重需从 HuggingFace 格式的 `self_attn.kv_b_proj.weight` 转换生成，而上述原生 PR 并未适配该算子拆分逻辑。

因此需手动改造适配相关权重转换逻辑，保障吸收矩阵可正常加载与生效。只有在吸收矩阵可用的基础上，才能正常使能 [sparse\_flash\_attention](https://gitcode.com/cann/ops-transformer/tree/master/attention/sparse_flash_attention) 与 [lightning\_indexer](https://gitcode.com/cann/ops-transformer/tree/master/attention/lightning_indexer) 融合算子；通过引入两个融合算子，可大幅减少内存访问频次、优化内存占用率，同时提升计算性能，最终实现大模型训练与推理链路的运行效率提升及资源开销降低。

### 2.4 整网功能打通

完成推理引擎适配验证、训练引擎适配开发，参照[参数配置说明](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/dev_guide/model_dev/parameter_and_metrics.md)根据实际业务需求，配置推理引擎、训练引擎的相关参数，完成 VeRL 整网功能打通，确保全流程稳定运行。

## 三、精度对齐

大模型强化学习的精度问题定位链路复杂、影响因素繁多，各类精度问题通常由**训练阶段、推理阶段、训推一致性**问题引入。**精度对齐**是保障训练流程可复现、问题可调试的核心关键。

训练与推理阶段的精度对齐，可参考官方文档：[精度对齐文档](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/dev_guide/precision_analysis/precision_alignment_zh.md)。因此本节不再赘述基础阶段对齐流程，将**重点围绕训推一致性场景**，基于 msprobe 精度工具，开展精度对齐落地实践与问题定位排查工作。

### 3.1 精度监控配置

整网跑通后，需启用精度监控参数，设置 `actor_rollout_ref.rollout.calculate_log_probs=True`。在训练过程中需重点观察以下关键指标，以此判断训推一致性及模型训练稳定性：

* **训推一致性参考指标**：
  * `training/rollout_probs_diff_mean`（rollout概率差异均值），模型正常收敛状态下，该指标建议维持在 0.01 以内；若数值持续高于 0.01 或与 GPU 基准存在明显偏离，可判定存在训推精度异常。
  * `training/rollout_probs_diff_max`（rollout概率差异最大值）
  * `training/rollout_actor_probs_pearson_corr`（rollout与actor概率的皮尔逊相关系数）
* **模型训练稳定性指标**：
  * `actor/grad_norm`：需关注其是否呈整体下降趋势，以此判断模型训练是否正常收敛。

此外，配置参数 `trainer.rollout_data_dir=./rollout_dump/` 用于保存训练过程中的 Rollout 中间结果。通过人工核查导出的 Rollout 数据，校验模型回复是否符合预期、输出有无乱码与重复回答现象，可进一步从表象上确认推理引擎适配无误。

### 3.2 采集精度数据

当 training/rollout_probs_diff_mean 超出 0.01 合理阈值、或与 GPU 基准标杆出现明显偏离时，需进一步通过[msprobe](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/dev_guide/precision_analysis/precision_debugger_zh.md)精度工具采集数据做根因定位。

### 3.3 训推差异点排查与对齐实践

完成数据采集后，优先读取 `construction.json` 文件进行模块级数据比对。先保证 `layer.0.input_layernorm` 输入数据完全一致，再逐模块逐层校验，定位训练与推理输出首次出现不一致的位置。

而对于大尺寸模型，微小数值差异会随着逐层累积、放大，导致训练（training）和推理（rollout）的结果差异明显，甚至会出现同一个token训推输出概率分别为0和1的现象，因此需尽可能将每一处差异点对齐至完全相等。

定位到差异节点后，适配修改方案同样是关键难点。由于业内各开源社区对相关模块存在多套不同实现，为保障模型实现逻辑的正确性，需多方参考权威源码与技术报告，综合确定最终对齐方案。

#### 3.3.1 常见训推不一致

在大模型强化学习实践中，可将训推不一致的典型根因归纳为以下五类：

1. **框架实现不一致**：由于训练、推理框架实现逻辑不同导致。有时是“语义正确”的（如算子拆分方式不同但数学等价），有时是“语义错误”的（如遗漏了某个缩放因子，或多了某个操作），需结合源码与技术报告严格鉴定。
2. **精度类型差异**：如训练侧全程 BF16，而推理侧在某些归一化等敏感算子中隐式升精到 FP32 计算再降精，导致截断误差。
3. **超参数不一致**：如LayerNorm模块中硬编码的 `eps` 值未做统一。
4. **并行策略**：训练时的张量并行 vs 推理时的连续批处理，导致浮点累加顺序差异。
5. **随机性控制**：Dropout、采样策略在训推阶段的实现偏差。

下面列举 GLM-5 模型迁移适配过程中遇到的典型训推不一致实际案例。

#### 3.3.2 案例一：FFN激活函数的框架实现不一致

从上往下依次比对，排查到第一层的 MLP 激活函数处输出不一致。

推理侧已正常使用 NPU 优化的 `npu_swiglu` 融合算子，但训练侧仍执行原生 GLU 小算子实现。

* **根因**：尽管已在 Verl 参数中添加了 `swiglu` 使能配置，但 Megatron-Bridge 在 NPU 适配 PR 中，未显式配置 `provider.bias_activation_fusion=True`，导致代码未进入 NPU 融合算子分支。
  ```
  +actor_rollout_ref.actor.megatron.override_transformer_config.swiglu=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True \
  ```
* **修复方案**：在 Megatron-Bridge 中添加配置项使训练侧正确调用融合算子：

#### 3.3.3 案例二：indexer_k_norm 的精度与超参数不一致

在严格对齐过程中，发现 `indexer_k_norm` 处存在精度类型与超参不一致：

* **精度差异**：推理侧在 LayerNorm 中存在升精度到 fp32 操作 `F.layer_norm( x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)`，而训练侧 Megatron 实现为 BF16。微小差异经多层累积不可忽视。
* **修复方案**：统一训练侧代码增加升精降精操作。
* **超参差异**：GLM5 推理侧vllm继承 DeepSeekV32 逻辑，`k_norm` 的 EPS 值被硬编码为 `1e-6`；而训练引擎及官方技术报告统一采用 `1e-5`。
* **修复方案**：将推理侧 EPS 修改为 `1e-5` 与训练侧对齐。

```
self.k_norm=LayerNorm(self.head_dim,eps=1e-6 -> 1e-5)
```

#### 3.3.4 案例三：lightning_indexer 模块逻辑缺失与冗余

排查发现lightning_indexer模块，训练与推理侧该模块具体实现存在不一致，具体表现为：

* **缺失（推理侧遗漏）**：推理侧缺失了 `weights` 的缩放逻辑。参考 Megatron 训练侧、slime 及 transformers 的标准实现，均包含该缩放，故在推理侧补齐以对齐前向：

```
weights, _ = self.weights_proj(x)
+weights = weights * (self.n_head**-0.5) * (self.head_dim**-0.5)
```

* **冗余（训练侧多余错误实现）**：训练侧 Megatron 实现中多出了 `rotate_activation`（哈达玛变换）。经查阅大量资料确认，该操作专用于量化场景，在 BF16 格式中属于错误实现。参考 [Transformer PR#45017](https://github.com/huggingface/transformers/pull/45017)，将训练侧该冗余逻辑移除。

```
class DSAIndexer(MegatronModule):
    def forward_with_scores(
-		q = rotate_activation(q)
-		k = rotate_activation(k)
```

### 3.4 MoE 大模型通用路由稳定方案

在典型的 RL 训练流程中，通常采用 vLLM 等高效推理引擎完成样本采样，再将采样数据送入 Megatron 等训练框架做模型训练优化。

对于常规稠密模型，推理与训练框架间的实现、环境差异仅会产生轻微数值偏差；但**大尺寸 MoE 模型**下该问题会被急剧放大。核心根源在于 MoE 动态路由机制：微小的框架实现、运行环境差异，就可能导致同一输入 Token 被分配至完全不同的专家组合，从而走向截然不同的计算路径。

这种路由决策的不一致，可能会导致MoE模型RL训练不稳定。它使得从推理阶段获取的“经验”对于训练阶段而言变得完全不同，优化信号因此失真，最终导致灾难性的后果。

为了解决这一通用问题，业界引入了 **Routing Replay（路由回放）** 机制。其核心思想是通过锁定特定阶段的专家路由路径，屏蔽微小扰动对路由决策的干扰，从而保证模型训练的稳定性。目前主流包含R2和R3两种变体：

* **（1）Vanilla Routing Replay (R2)**： (对应`actor_rollout_ref.actor.router_replay.mode="R2"`)
  
  * **机制**：在梯度更新阶段，复现训练引擎在上一轮采样阶段计算出的专家路径。
  * **作用**：主要缓解**策略陈旧性**对路由的影响。随着策略的更新，当前前向传播计算出的路由可能与生成旧数据时的路由不一致，R2通过回放旧路由来维持优化信号的连贯性。
* **（2）Rollout Routing Replay (R3)**：(对应`actor_rollout_ref.actor.megatron.router_replay.mode="R3"`)
  
  * **机制**：在序列生成过程中捕捉推理引擎的路由分布，并将其直接重放到训练引擎中。
  * **作用**：同时解决**训练-推理偏差**和**策略陈旧性**两个问题。它确保了训练阶段计算Loss所依据的专家路径，与实际推理生成结果时的专家路径绝对一致。

因此，无论是侧重缓解策略陈旧的 R2，还是实现全链路对齐的 R3，Routing Replay 机制都有效弥合了推理与训练框架间的路由鸿沟。在**大尺寸 MoE 模型**的训推一致性对齐中，该机制已成为保障精度对齐与训练稳定的核心手段。目前，DeepSeek-V3.2、GLM-5、MiMo-V2 等主流大模型均已采用了 R3 模式的 Routing Replay 技术。

因此对于大尺寸 MoE 模型，在实际配置中通常推荐使用对齐更彻底的 R3 模式：

```
actor_rollout_ref.actor.router_replay.mode="R3" \
actor_rollout_ref.rollout.enable_rollout_routing_replay=True \
```

## 四、性能优化

在昇腾 NPU 上进行大模型 RL（强化学习）训练性能优化时，基础配置调优可优先参考官方文档：[perf_tuning.rst](https://github.com/verl-project/verl/blob/04833f01/docs/perf/perf_tuning.rst)。为实现更高效的优化，建议遵循**数据采集​​→​瓶颈定位​→配置调优→迭代验证**的标准化流程，该流程可显著提升 Rollout、Reward、Update 等核心阶段的吞吐量，同时有效降低资源空泡与负载不均问题。性能分析与调优的具体操作，可严格参照以下官方指引：

1. Ascend Performance Analysis Guide：[https://github.com/verl-project/verl/blob/main/docs/ascend\_tutorial/examples/ascend\_performance\_analysis\_guide.md](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/examples/ascend_performance_analysis_guide.md)
2. Profiling 数据采集与使能配置：[https://github.com/verl-project/verl/blob/main/docs/ascend\_tutorial/profiling/ascend\_profiling\_zh.rst](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/profiling/ascend_profiling_zh.rst)

### 4.1 推理性能优化

Rollout 阶段作为大模型 RL 训练的核心推理环节，其推理耗时在整个训练流程中占据绝大部分，以下是该阶段常见的性能优化手段：

1. **启用图模式功能**：图模式将整个计算图提前编译优化，可以实现算子融合、内存复用、常量折叠等深度优化，显著提升执行效率。
2. **CPU 绑核加速算子下发**：通过 CPU 绑核可提升算子下发效率；自 vllm-ascend v0.18.0rc1 版本起，ARM 架构昇腾服务器已默认开启该能力。
3. **HCCL 通信算法配置为 AIV 模式**：将环境变量 `HCCL_OP_EXPANSION_MODE` 设置为 `AIV` 模式，指定通信算法的编排与展开逻辑运行在 Device 侧 Vector Core 计算单元。
4. **启用异步调度**：能够消除 Worker 连续两次 execute_model 执行间隙，让 Worker 可直接获取已调度完成的 SchedulerOutput 进行模型推理，无需阻塞等待调。

对应配置参数如下：

```
# 图模式启用
actor_rollout_ref.rollout.enforce_eager=False +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY" 
+actor_rollout_ref.rollout.engine_k
wargs.vllm.compilation_config.cudagraph_capture_sizes="[2, 4, 8, 16, 24, 32]"
# CPU绑核
++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=True
# 异步调度启用
++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=True
```

### 4.2训练性能优化：

大模型 RL 训练的 Update 阶段具有序列长度差异大、显存消耗高等特点。除了基础的算子融合，还需结合序列并行与显存-计算权衡策略来打破瓶颈。常见训练性能优化特性可参考 [MindSpeed-verl 文档](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/user-guide/verl.md) 完成启用，核心优化手段包括：

1. ​**算子融合**​：启用 RoPE、SwiGLU、RMSNorm、DSA 等融合算子。通过算子融合减少计算开销与显存，提升训练效率。
2. ​**Remove padding**​：RL 训练中各 Response 长度参差不齐，传统 Padding 策略会导致大量无效计算。开启 Remove padding后可将多个短序列打包填满 Tensor，极大提升 NPU 计算单元的利用率（MFU）。

## 五、评测验证

训练完成后，需对目标数据集进行评测验证，确保模型迁移后的业务效果达标。不同模型的评测步骤一致，以下以 GLM-5 为例，详细说明评测流程（采用 AISBenchmark 工具，支持 vllm/sglang 多种推理后端的评估）。

评测采用了数学类的数据集aime2025与研究生级专业理科数据集gpqa，验证在目标方向上分数上升，且无关方向不会出现知识灾难遗忘情况。

### 5.1 安装aisbench

```shell
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark
pip install -e .
```

### 5.2 下载评估数据集

```shell
# linux服务器内，处于工具根路径下
cd path/to/benchmark/ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime2025.zip
unzip aime2025.zip
rm aime2025.zip
```

### 5.3 修改AISBench配置代码使能vllm/sglang推理评测

打开 benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py 文件，这是推理评测配置文件，输出长度`max_out_len`建议与训练的`max_response_len`保持一致。

```shell
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="/path/to/GLM-5", # 修改为 GLM-5 模型路径
        model="GLM-5",
	    stream=True,
        request_rate = 0,
	    use_timestamp=False,
        max_seq_len=2048,
        retry = 2,
	    api_key="",
        host_ip = "localhost", # 推理服务的IP
        host_port = 12890 , # 推理服务的端口
        max_out_len = 8192,  # 最大输出tokens长度
        batch_size=48, # 推理的最大并发数
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0,
            seed = 1234,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
```

### 5.4 多机拉起推理服务端

参考[vllm_ascend GLM5指南](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.md#multi-node-deployment)拉起双机A3推理服务，`host_port`与上一小节配置保持一致，`max_model_len`设置为训练时的`max_prompt_length`与`max_response`之和。

### 5.5 启动vllm评测任务

执行以下命令启动在线推理评测任务，调用已部署的 vLLM 推理后端，加载对应模型配置完成自动化评测：

```
ais_bench --models vllm_api_stream_chat --datasets aime2025_gen_0_shot_chat_prompt
```

模型经过训练后，核心能力指标实现稳定提升：在AIME2025 数学推理数据集上评测得分稳步上涨，同时在GPQA 研究生级专业理科数据集上也实现了持续的分数增益，无知识退化、无灾难性遗忘问题，训练优化效果符合预期。

| 评测数据集 | GLM5-base | 10step | 15step | 40step | 50step |
| ---------- | --------- | ------ | ------ | ------ | ------ |
| aime2025   | 47.5      | 49.17  | 49.17  | 48.33  | 52.5   |
| gpqa       | 64.65     | 68.81  | 68.43  | 69.07  | 71.21  |

## 六、总结

本文完整覆盖了大模型从 GPU 迁移至昇腾 NPU 或在 NPU 上独立适配的全流程实践，主要分为环境搭建、组件联调、精度对齐、性能优化、评测验证五大关键环节，为开发者提供可落地、可复用的操作指南与问题解决方案。

前期准备阶段需重把控环境依赖版本、模型权重精度与数据集格式，为后续适配奠定基础；组件联调环节需遵循先单组件验证后整网打通的原则，优先确保推理、训练引擎及权重转换工具的稳定适配，针对特殊模型结构需完成定制化改造；精度对齐是迁移适配的核心，需重点监控训推一致性指标，通过逐模块排查解决框架实现、精度类型等常见差异，MoE 模型需启用 Routing Replay 机制保障训练稳定；性能优化需遵循标准化流程，聚焦推理与训练核心阶段，通过图模式、算子融合等手段提升效率、降低资源消耗；最终通过标准化评测验证，确保模型迁移后业务效果达标、无知识退化。

整体而言，遵循本文流程可有效降低 NPU 迁移适配成本，规避常见坑点，实现大模型在昇腾 NPU 上的稳定、高效运行。
