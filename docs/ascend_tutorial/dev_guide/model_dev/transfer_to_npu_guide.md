# Transfer to NPU guide

==================================================================================

Last updated: 05/13/2026

本文以 GLM5 为例，为开发者提供从 GPU 迁移至 NPU或在 NPU 上独立适配模型的完整实践经验，涵盖前期准备、各组件打通、精度对齐、性能优化及长跑评测全流程。

## 一、前期准备

搭建可支持 NPU 运行的基础运行环境，保证模型正常加载、数据集可顺利读取，为后续迁移调试、业务跑通的基础。

### 1.1 软硬件环境与依赖配置

参照官方文档 [ascend\_quick\_start.rst](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/quick_start/ascend_quick_start.rst) 完成基础环境搭建；若模型依赖的推理引擎 vllm、vllm_ascend 和训练引擎Megatron、MindSpeed、transformers 版本与教程存在差异，**以模型实际适配版本为准**。

### 1.2 模型权重

BF16 为 VeRL 框架中 FSDP 与 Megatron 训练后端**默认混合精度训练数据类型**。昇腾 NPU 环境统一采用 **BF16** 作为基准精度格式，权重需对齐转换为 BF16。目前 A2、A3 机型**暂不支持 FP8 精度训练**，仅支持 BF16 精度；A5 机型后续版本将开放 FP8 低精度训练能力。

**GLM5 权重链接：**

BF16：[https://huggingface.co/zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5)

### 1.3 数据准备

数据需参照[Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)将数据集预处理为 parquet 格式：(1) 确保它包含计算强化学习奖励所需的必要字段；(2) 读取速度更快。
本文使用[dapo-math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/blob/main/data/dapo-math-17k.parquet)数据集，VeRL已支持该数据集可直接使用

## 二、各组件联调减层打通

VeRL 存量场景已支持 NPU 设备类型自动识别，GPU 环境下的运行脚本迁移至昇腾 NPU 后，原则上无需显式配置 trainer.device=npu 参数；若需使用新增特性，仍可通过设置 trainer.device优先启用，相关特性将逐步适配设备自动识别能力。VeRL 不同推理、训练后端的具体特性支持，可参考[昇腾特性文档](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/features/ascend_backend_features.md)

### 2.1 推理引擎适配

VeRL 推理引擎采用分层架构设计，通过抽象接口与工厂模式，实现 vllm、sglang 等多种主流推理后端的灵活支持。在完成 GPU 向 NPU 的迁移适配过程中，推理引擎适配推荐按以下流程操作：

在 NPU 上跑通 VeRL 整网链路前，建议参考 [vllm-ascend](https://github.com/vllm-project/vllm-ascend/tree/main/docs/source/tutorials/models)、[sglang](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/basic_usage) 官方模型部署教程，优先调通**单实例推理链路**，完整验证模型加载与初始化、Tokenizer 加载正常、单轮 / 批量生成、停止词终止、长上下文推理等**基础推理功能**，确保底层推理底座稳定可用后，再接入 VeRL 训练流程。

我们参照[vllm_ascend GLM5指南](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.md#multi-node-deployment)拉起服务端后参照[ais\_bench.md](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/developer_guide/evaluation/using_ais_bench.md)对aime2025执行了不同输出长度下的评测任务，得到下表，完成了对推理引擎功能的验证。

| length | score |
| ------ | ----- |
| 20k    | 60.00 |
| 70k    | 96.67 |

### 2.2 训练引擎选择与适配

VeRL 主线代码将训练引擎抽象为 `Engine`类，通过标准化接口层实现调度逻辑与底层训练实现的解耦。该架构设计支持 FSDP、Megatron、MindSpeed-LLM 等多种训练后端灵活接入、即插即用，无需修改 VeRL 核心算法与调度逻辑，大幅降低迁移适配成本。

当前 NPU 已通过 `is_npu_available` 接口完成设备自动检测，并自动应用对应的 NPU 设备适配补丁。目前只需通过配置 model_engine=fsdp/megatron/mindspeed，即可一键切换训练后端至 FSDP、Megatron 或 MindSpeed-LLM，系统会自动加载对应后端的 NPU 适配逻辑，无需额外修改代码。

GLM5适配使用旧版的`actor_rollout_ref.actor.strategy=megatron`并使用 `override_transformer_config` 参数传入MindSpeed自定义特性，具体配置参考[MindSpeed-verl文档](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/user-guide/verl.md)设置。

### 2.3 Megatron-Bridge适配

Megatron-Bridge 用于在 VeRL 框架中实现推理引擎所需的 HuggingFace 权重与 Megatron-Core 所需的 mcore 权重之间的转换，可通过以下配置启用该功能：

```
actor_rollout_ref.actor.megatron.use_mbridge=True
actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
```

目前 Megatron-Bridge 已支持 GLM-5 模型（相关支持参考 [PR #2913](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2913)），在 NPU 环境迁移适配 GLM-5 模型时，我们参考该 PR 完成了 Megatron-Bridge 的相关配置。

昇腾 MindSpeed 支持 DSA（Dynamic Sparse Attention）使用吸收矩阵，此特性要求将 Megatron 中的 linear_kv_up_proj 拆分为`linear_k_up_proj`和 `linear_v_up_proj`。该拆分所需的权重，由 HuggingFace 格式中的 `self_attn.kv_b_proj.weight` 转换生成，但原生 PR 未支持该拆分功能。

为此，我们对该部分逻辑进行了手动适配，确保吸收矩阵能够正常使用；只有在吸收矩阵可用的前提下，才能使能[sparse\_flash\_attention](https://gitcode.com/cann/ops-transformer/tree/master/attention/sparse_flash_attention)和 [lightning\_indexer](https://gitcode.com/cann/ops-transformer/tree/master/attention/lightning_indexer)融合算子，从而显著减少内存访问次数、优化内存占用效率、提升计算性能及数值稳定性，进而在大模型训练与推理过程中实现更高的运行效率和更低的资源开销。

### 2.4 整网功能打通

完成推理引擎适配验证、训练引擎选型后，参照官方文档[ascend\_backend\_features.md](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/features/ascend_backend_features.md)根据实际业务需求，配置推理引擎、训练引擎的相关参数，完成 VeRL 整网功能打通，确保全流程稳定运行。

## 三、精度对齐

### 3.1 精度监控配置

整网跑通后，启用精度监控参数，设置`actor_rollout_ref.rollout.calculate_log_probs=True`，在训练过程中重点观察关键指标，以此判断训推一致性及模型训练稳定性：

* 训推一致性参考指标：training/rollout_probs_diff_mean（rollout概率差异均值）、training/rollout_probs_diff_max（rollout概率差异最大值）、training/rollout_actor_probs_pearson_corr（rollout与actor概率的皮尔逊相关系数）
* 模型训练稳定性指标：actor/grad_norm，需关注其是否呈整体下降趋势，以此判断模型训练是否正常收敛

配置参数 trainer.rollout_data_dir=./rollout_dump/，用于保存训练过程中的 Rollout 中间结果；通过核查导出的 Rollout 数据，校验模型回复是否符合预期、输出无乱码与重复回答现象，进一步确认推理引擎适配无误。

### 3.2 采集精度数据

通过观察训推一致性指标`training/rollout_probs_diff_mean`，若该指标高于0.01或与能拿到的GPU标杆差异明显，则需要使用精度工具 msprobe进行进一步排查。

GLM5整网跑通后，初始training/rollout_probs_diff_mean为0.15左右，说明rollout概率差异均值较大，训推不一致程度超出合理范围，需要采集精度数据定位分析问题。

参照msProbe官方指南，分别修改[推理](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/dump/vllm_dump_instruct.md)与[训练](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/dump/verl_megatron_consistency_preprocess_dump.md)代码，将训练输入调整为单prompt格式，统一推理与训练阶段的输入，确保采集到的prefill阶段前向数据具备可比性，然后导出Dump数据。通过分析该数据的差异点，验证推理引擎与训练引擎在模型前向计算逻辑上的一致性，精准排查因训推引擎前向计算差异导致的精度偏差问题。

通过配置以下配置，

1. `max_response_lenth=1`设置为1，使模型仅执行 prefill 阶段，专注比对训推引擎 prefill 前向过程；
2. 模型精简为 5 层结构（3 层 dense+2 层 moe），权重采用 dummy 初始化；
3. 统一训推切分策略：TP64、DP1、EP64

核心配置参数如下：

```
train_batch_size=1
n_resp_per_prompt=1
ppo_mini_batch_size=1
data.max_response_length=1
actor_rollout_ref.rollout.agent.num_workers=1
+actor_rollout_ref.model.override_config.num_hidden_layers=5
actor_rollout_ref.rollout.load_format='dummy'
actor_rollout_ref.rollout.tensor_model_parallel_size=64
actor_rollout_ref.rollout.data_parallel_size=1
actor_rollout_ref.rollout.expert_parallel_size=64
actor_rollout_ref.actor.megatron.tensor_model_parallel_size=64
actor_rollout_ref.ref.megatron.tensor_model_parallel_size=64
actor_rollout_ref.actor.megatron.expert_model_parallel_size=64
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
```

### 3.3 精度对齐-训推差异点排查

#### 3.3.1 定位与修改思路

完成 3.2 节所述前置操作后，以 mix 精度级别采集数据并开展比对。优先读取`construction.json`文件进行模块级数据比对，先保证`layer.0.input_layernorm`输入数据完全一致，再逐模块逐层校验，定位训练与推理输出首次出现不一致的位置。

大尺寸模型中，微小数值差异会经 78 层网络逐层累积放大，因此需尽可能将每一处差异点对齐至完全相等。

定位到差异节点后，适配修改方案同样是关键难点。由于业内各开源社区对相关模块存在多套不同实现，为保障模型实现逻辑的正确性，需多方参考权威源码与技术报告，综合确定最终对齐方案。

#### 3.3.2 FFN激活函数训推实现不一致

从上往下依次比对，排查到第一层的MLP激活函数处存在有输出不一致的地方

```
	elif self.config.bias_activation_fusion:
            *********
            else:
               *********
                elif self.activation_func == F.silu and self.config.gated_linear_unit: 
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                        self.config.cpu_offloading
                        and self.config.cpu_offloading_activations
                        and HAVE_TE,
                    )
                ***************
	else:
            ************
            if self.config.gated_linear_unit:
                def glu(x):
                    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                    if (val := self.config.activation_func_clamp_value) is not None:
                        x_glu = x_glu.clamp(min=None, max=val)
                        x_linear = x_linear.clamp(min=-val, max=val)
                    return self.config.activation_func(x_glu) * (
                        x_linear + self.config.glu_linear_offset
                    )
                intermediate_parallel = glu(intermediate_parallel)
            ******************
```

结合 stack.json 调用栈溯源代码后确认，**Dense层的MLP激活函数实现存在训推不一致**，推理侧已正常使用 NPU 优化的 `npu_swiglu` 融合算子，但​**训练侧未生效**​。

尽管已在 Verl 参数中添加 MindSpeed 相关 `swiglu` 使能配置，但训练侧仍无法调用 `npu_swiglu` 算子。​**根因​**​：Megatron-Bridge 在 NPU 适配 PR 中，​未显式配置 `provider.bias_activation_fusion=True`，导致代码未进入 NPU 融合算子分支，而是执行了原生 GLU 小算子实现。

```
+actor_rollout_ref.actor.megatron.override_transformer_config.swiglu=True \ +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True \
```

修复方案：在 Megatron-Bridge 中添加上述配置项`provider.bias_activation_fusion=True`，使训练侧正确调用 NPU 优化的 npu_swiglu 融合算子。

​修复效果​：精度差异指标 `training/rollout_probs_diff_mean` 从 ​**0.15 降至 0.04**​，训推一致性显著提升，但差异仍偏高，需继续排查其余误差点。

#### 3.3.3 indexer_k_norm推理侧升精操作

后续使用torch.equal对每个操作的输出tensor严格对齐，发现indexer_k_norm处，推理相较于训练，k_norm存在升精度到fp32操作，对于大尺寸模型微小的差异会经过78层逐层累积，所以尽可能的将每一个差异点对到完全相等
**推理：**

```
class LayerNorm(nn.Module):
    def forward(self, x: torch.Tensor):
        return F.layer_norm(
            x.float(), (self.dim,), self.weight, self.bias, self.eps
        ).type_as(x)
```

**训练侧实现为BF16**

```
class FusedLayerNormAffineFunction:
    @staticmethod
    def forward(*args, **kwargs):
        return FusedLayerNormAffineFunction.apply(*args, **kwargs)
```

故采用相同的写法将训练侧代码升精降精度操作

#### 3.3.4 indexer_k_norm训推eps参数不一致

GLM5 推理侧继承 DeepSeekV32 实现逻辑，在 vLLM 框架中`k_norm`的 EPS 值被硬编码为`1e-6`；而训练引擎 Megatron 及官方技术报告统一采用`1e-5`，参数规格不匹配引发精度误差。临时将推理侧 EPS 修改为`1e-5`，与训练侧及标准规范对齐。

```
class Indexer(nn.Module):
    def __init__():
	*********
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
```

#### 3.3.5 **lightning_indexer逻辑不一致**

推理侧缺失weights的缩放`weights, _ = self.weights_proj(x)`，相较于训练侧Megatron的实现，该部分缺失缩放逻辑，参考slime和transformers上的实现均含有该部分，故做补充以确保训推前向对齐

```
weights, _ = self.weights_proj(x)
weights = weights * (self.n_head**-0.5) * (self.head_dim**-0.5)
```

训练侧megatron的实现中相较于vllm_ascend多出来了rotate_activation，使用了哈达玛变换。该部分经过查阅大量资料后，定为用于量化时候的操作，不该在bf16格式中使用，参考[Transformer PR#45017](https://github.com/huggingface/transformers/pull/45017)。

```
class DSAIndexer(MegatronModule):
    def forward_with_scores(
	# =========================================
	# Rotate activation
	# =========================================
	q = rotate_activation(q)
	k = rotate_activation(k)
```

最终经过训推对齐后，`training/rollout_probs_diff_mean`指标从0.15降低到0.014

### 3.4 Moe RoutingReplay适配

在典型的RL流程中，使用高效的推理引擎（如vllm）进行数据采样，再将数据送入训练框架如Megatron进行模型优化。对于标准稠密模型，这种框架差异或许只会带来微小的数值误差；但在MoE模型中，这个问题被急剧放大。其核心在于MoE的路由机制（Routing Mechanism）：微小的环境或实现差异，都可能导致模型为同一个输入token选择完全不同的专家组合，从而走向截然不同的计算路径。这种路由决策的不一致，可能会导致MoE模型RL训练不稳定。它使得从推理阶段获取的“经验”对于训练阶段而言变得完全不同，优化信号因此失真，最终导致灾难性的后果。
因此在训推一致对齐过程中，对于MoE模型还需要开启routing replay。

Routing Replay有R2/R3两种变体。

（1）Vanilla Routing Replay (R2):对应verl中开关`actor_rollout_ref.actor.router_replay.mode="R2"`

机制：回放训练引擎在采样阶段计算出的专家路径。R2 的目标是减轻专家路由对策略陈旧的影响，其方法是在梯度更新阶段，复现训练引擎中 rollout 策略所选择的路由专家

作用：主要减少策略陈旧性对路由的影响。

（2）Rollout Routing Replay (R3): :对应verl中开关`actor_rollout_ref.actor.megatron.router_replay.mode="R3"`

机制：在序列生成过程中捕捉推理引擎的路由分布，并将其直接重放到训练引擎中。

作用：同时减少训练-推理偏差和策略陈旧性。

在本文训推一致对齐过程中，使用以下配置启用R3来进行精度对齐，启用R3策略后，`training/rollout_probs_diff_mean`指标从0.014进一步降低到0.0035

```
actor_rollout_ref.actor.router_replay.mode="R3" \
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True \
```

## 四、性能优化

744B大尺寸模型，最初使用piece-wise图模式，时长为一小时一步，实验成本较大，不利于推进，故将优化重点聚焦于模型运行效率提升，核心目标是缩短单步训练时长、降低实验成本，同时保障模型精度不损失，为后续实验迭代和精度排查提供支撑。

图模式功能适配

```
actor_rollout_ref.rollout.enforce_eager=False +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY" 
+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[2, 4, 8, 16, 24, 32]"
```

启用图模式后，推理时间从一步60分钟降到一步20分钟

## 五、评测验证

训练完成后，需对目标数据集进行评测验证，确保模型迁移后的业务效果达标。不同模型的评测步骤一致，以下以 GLM-5 为例，详细说明评测流程（采用 AISBenchmark 工具，支持 vllm/sglang 多种推理后端的评估）。

评测采用了数学类的数据集aime2025与研究生级专业理科数据集gpqa，验证在目标方向上分数上升，且无关方向不会出现知识灾难遗忘情况

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

打开 benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py 文件，这是推理评测配置文件，输出长度`max_out_len`建议与训练的`max_response_len`保持一致

```shell
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content
from ais_bench.benchmark.clients import OpenAIChatStreamClient, OpenAIChatStreamSglangClient

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-general-chat',
        path="/path/to/GLM-5", # 修改为 GLM-5 模型路径
        model="GLM-5",
        request_rate = 0,
        max_seq_len=2048,
        retry = 2,
        host_ip = "localhost", # 推理服务的IP
        host_port = 12890 , # 推理服务的端口
        max_out_len = 8192,  # 最大输出tokens长度
        batch_size=48, # 推理的最大并发数
        trust_remote_code=False,
        custom_client=dict(type=OpenAIChatStreamSglangClient), #使用sglang客户端
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
ais_bench --models vllm_api_stream_chat --datasets math500_gen_0_shot_cot_chat_prompt
```

模型经过训练后，核心能力指标实现稳定提升：在AIME2025 数学推理数据集上评测得分稳步上涨，同时在GPQA 研究生级专业理科数据集上也实现了持续的分数增益，无知识退化、无灾难性遗忘问题，训练优化效果符合预期。

| 评测数据集 | GLM5-base | 10step | 15step | 40step | 50step |
| --- | --- | --- | --- | --- | --- |
| aime2025 | 47.5 | 49.17 | 49.17 | 48.33 | 52.5 |
| gpqa | 64.65 | 68.81 | 68.43 | 69.07 | 71.21 |
