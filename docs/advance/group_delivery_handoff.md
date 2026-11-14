# group_delivery 三卡 A6000 验证交接

Last updated: 10/27/2026.

这份交接面向接手 GPU 验证的开发者。实现入口是
`trainer.v1.trainer_mode=group_delivery`：分卡、固定 rollout batch、固定策略版本，
完整 GRPO group 到齐后立即 forward/backward；所有 group 都贡献梯度后，只执行一次
optimizer step、scheduler step 和权重同步。失败时不丢弃慢组、不补抽新 prompt、不提交部分梯度。

**当前状态：CPU 验证完成情况见下文；CUDA、真实 Megatron 通信和 ALFWorld 实体环境尚未实测。**
三卡机器首先验证数值正确性，再验证真实 agentic 长尾 workload 的重叠收益，不能把 mock 测试当成 GPU 验收。

## 代码与重复工作核查

基线：`verl-project/verl` 的 `4d3ed12a7617c6b78ee0ac8f27dd7848bb0d11ee`。
分支：`group-delivery`；功能和配置名称统一为 `group_delivery`。
上游评审：[verl-project/verl#8466](https://github.com/verl-project/verl/pull/8466)。
这个分支包含实现、测试、ALFWorld 配方及两份 group_delivery 文档。

另一个 checkout 获取本分支：

```bash
git fetch origin pull/8466/head:group-delivery-validation
git switch group-delivery-validation
git rev-parse HEAD
```

记录实际检出的 commit，按下文准备 Linux/GPU 环境；不要复制开发机的 macOS `.venv`。
离线迁移也可以导出 `git diff --binary 4d3ed12a7617c6b78ee0ac8f27dd7848bb0d11ee HEAD > group_delivery.patch`，
在同一基线上先 `git apply --check group_delivery.patch`，再 `git apply group_delivery.patch`。

2026-10-27 检索了公开 PR/issue 的同步、分卡、group、streaming、gradient accumulation、
on-policy pipeline 等关键词，覆盖打开和关闭状态，并检查主要候选的具体 diff。
结论是**有相关工作，未找到完整实现本次固定 batch 提前 backward、最终一次 step 语义的候选**；
这不是“没有任何类似 PR”的绝对保证，提交上游前需刷新检索。

| 相关工作 | 核查结论与复用边界 |
| --- | --- |
| [verl #6717](https://github.com/verl-project/verl/pull/6717) | 已有 Tinker zero-grad / forward-backward / optimizer-step 分离接口；本实现直接复用。它不负责 group 到达调度和整批提交屏障。 |
| [verl #8201](https://github.com/verl-project/verl/pull/8201)、[#6025](https://github.com/verl-project/verl/pull/6025) | group batching、可变 trajectory 数和 DP 分配相关；更新仍按 mini-batch，不是本次边界。 |
| [verl #8203](https://github.com/verl-project/verl/pull/8203) | StreamingDataloader 的生产、补充与缓存结构相关；未看到拆分 optimizer 更新的实现。后续若合入，需把本模式的采样接入迁移到新数据层。 |
| [verl #6868](https://github.com/verl-project/verl/pull/6868)、[#6874](https://github.com/verl-project/verl/pull/6874) | fully async / rollout 调度；每个 mini-batch 更新与延后权重同步不能替代梯度累积。 |
| [verl-recipe #93](https://github.com/verl-project/verl-recipe/pull/93) | AsyncFlow 的 staleness=0 是相关方案；候选代码仍调用每批 update_actor，且不是本次 Megatron group 累积路径。 |

核心改动位于：

- `verl/trainer/ppo/v1/trainer_group_delivery.py`：复用分卡资源初始化和标准 PPO 单组计算，控制固定 batch 屏障。
- `verl/trainer/ppo/v1/group_delivery.py`：只交付该 batch、该策略版本的完整 group。
- `verl/workers/engine_workers_group_delivery.py`、`verl/workers/utils/group_gradient_accumulator.py`：复用 Tinker 接口，保存每组已经归约的 FP32 梯度。
- `verl/workers/engine/megatron/transformer_impl.py`：静态 CP token 计数保留余数，避免 group 切分改变归一化分母。
- `examples/group_delivery/`：独立的 ALFWorld 验证配方，不向训练核心加入环境依赖。

## 训练语义和支持边界

每次 Megatron schedule 都会 finalize 梯度。直接把上一组梯度留在 native buffer 中，可能重复归约或缩放。
这里将每组 finalize 后的梯度拷到额外 FP32 accumulator，再清空 native buffer。
最后把整批加权平均写回，交给原生 optimizer 做全局 norm、clip 和 update。
`token-mean` 按有效 response token 数加权；sequence mean 按至少有一个有效 response token 的样本数加权。
工具观察与 padding 不计入有效 response token。

所有 DP/TP/PP/CP rank 按同一 group 顺序进入 collective。DP lane 不够时补零 loss 的样本，
不能让不同 DP rank 各自从队列取不同 group。TP/PP/静态 CP 与组合沿用 Megatron 的 process groups。
actor/reference 的 DP 对齐分别计算；同进程的 Megatron actor/reference 仍须遵循上游共享 model-parallel
初始化的限制，不能据此配置两个独立的 TP/PP/CP 拓扑。通信发生在每组，额外内存约为一份本 rank 原生 FP32 梯度 buffer；
所以小 group、通信瓶颈或 PP 空泡可能抵消 rollout 重叠收益，不能预先承诺加速。

当前限定：BF16、原生 Megatron DDP/普通 Adam 或 SGD、GRPO、vanilla PPO loss、`ppo_epochs=1`、
`ppo_mini_batch_size=train_batch_size`。支持 token/sequence mean、熵正则及 reference KL loss。
`seq-mean-token-sum-norm` 必须指定固定正 `loss_scale_factor`。

以下组合会拒绝启动：动态 CP、VPP、Megatron FSDP、precision-aware optimizer、参数/optimizer offload、
FP8/FP4/QAT、MTP、非零 MoE aux/z loss、expert bias/quantile 状态更新、Sinkhorn、critic、distillation、
KL-in-reward、动态 group filtering、rollout IS/RS、score centering、可变 n，以及一个 session 返回多个 trajectory。
普通多轮工具调用最终返回一个 trajectory 可以使用。EP 仅在上述 MoE 限制内沿用原生归约，尚无本次 GPU 验收结果。

checkpoint 只在完整 optimizer update 后保存。不支持恢复半个累积窗口；出错后从上一完整 checkpoint 重启。
第 k 个训练 step 的 prompt tag 应为 k，trajectory 的 min/max policy version 都应为 k-1。
训练中间不刷新 rollout 权重。日志中的初始权重同步不计入每个 batch 的一次同步。

## 三卡拓扑和模型

主模型选 [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)。
官方配置为 28 层、16 attention heads、8 KV heads，适合 TP=2 或 PP=2 的验证。
使用 BF16，关闭 thinking，先测 8K 总上下文、group n=2、每 batch 8 个 prompt。
这是系统验收模型；ALFWorld 成功率和工具调用格式仍要通过 pilot 验证。

假设机器为每卡 48 GB 的 RTX A6000，先用 `nvidia-smi` 确认。
DP=2 时，按 1.7B 参数估算，BF16 参数、原生 FP32 梯度、额外 accumulator、DP 分片的 master weights/Adam
合计约 27 GB/训练卡，尚未包括 activation、通信 buffer、CUDA runtime 和临时张量。
启动脚本开启全层重计算；若 OOM，先把 response 长度从 6144 降到 2048，再逐步升回。
不要直接换 4B 全参训练，也不要打开当前不支持的 offload 来掩盖显存问题。

| 用途 | 训练 GPU | rollout GPU | DP | TP | PP | CP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 主验收 | 2 | 1 | 2 | 1 | 1 | 1 |
| TP 验收 | 2 | 1 | 1 | 2 | 1 | 1 |
| PP 验收 | 2 | 1 | 1 | 1 | 2 | 1 |
| CP 验收 | 2 | 1 | 1 | 1 | 1 | 2 |

三卡无法同时验证 TP2×PP2 或 DP2×TP2 的分卡训练：它们至少需要 4 张训练卡加 1 张 rollout 卡。
TP2×PP2×CP2 至少需要 8 张训练卡。数值回归脚本包含这些拓扑，卡数不够时明确 skip；
不能将 skip 写成通过。DP4 的 n=2 测试专门覆盖完全零 loss 的 DP lane。

## 环境准备

以下命令都在迁移后的仓库根目录执行。不要复制开发机的 macOS `.venv`。

```bash
uv venv --python 3.12
uv sync --locked --extra megatron --extra vllm --extra test
source .venv/bin/activate
mkdir -p artifacts/group_delivery
nvidia-smi > artifacts/group_delivery/nvidia-smi.txt
nvidia-smi topo -m > artifacts/group_delivery/topology.txt
uv pip freeze > artifacts/group_delivery/python-packages.txt
python -c 'import torch, megatron.core, transformer_engine, vllm; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'
```

跟随当前仓库 `uv.lock` 和 `pyproject.toml` 的组合，不单独升级 Megatron-Core/Bridge。
本基线使用 Megatron-Core 0.19.2 接口；GPU 安装版本以 lock 为准。
驱动须满足实际安装 CUDA runtime 的要求。先处理 import、CUDA kernel 和 NCCL 初始化失败，再进入算法测试。

```bash
hf download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B
export MODEL_PATH="$PWD/models/Qwen3-1.7B"
```

记录下载时的 Hugging Face commit，后续 A/B 使用同一份 checkpoint。模型目录、ALFWorld 数据、checkpoint、日志不提交仓库。

ALFWorld 采用[官方 text 环境](https://github.com/alfworld/alfworld)，单独安装，避免和训练环境依赖冲突：

```bash
uv venv .venv-alfworld --python 3.10
uv pip install --python .venv-alfworld/bin/python alfworld fastapi uvicorn
export ALFWORLD_DATA="$PWD/data/alfworld"
.venv-alfworld/bin/alfworld-download
uv pip freeze --python .venv-alfworld/bin/python > artifacts/group_delivery/alfworld-packages.txt
ALFWORLD_MAX_STEPS=50 .venv-alfworld/bin/python examples/group_delivery/alfworld_server.py
```

最后一条在单独终端常驻运行；只监听 `127.0.0.1:8765`。训练与服务应处于同一网络命名空间。
每个 trajectory 独占一个环境，所有 tool calls 共享它；成功、异常或取消时关闭。
不使用可视化 THOR，不需要环境占用额外 GPU，不向模型暴露 expert plan。
若训练进程被强杀，重启本次环境服务清理遗留 session。

## 先做数值验收

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m pytest -q -s tests/models/test_group_delivery_megatron.py \
  | tee artifacts/group_delivery/megatron-numerics.log
```

该测试本地生成很小的 Qwen3，不下载模型；使用真实 Megatron、distributed optimizer 和固定响应/advantage，
对比整批一次 forward/backward 与不同完成顺序的逐组累积，连续跑两个 update。
检查 owned FP32 gradient shards、clip 前 global norm 和 FP32 optimizer update。
梯度相对 L2 误差阈值 2%，update 3%，norm 2%，用于容纳 BF16 计算顺序差异；
应同时观察实际误差，不能为通过而任意放宽阈值。两张可见 GPU 应运行 8 个 case，另外 8 个大拓扑 case skip。
CUDA 可用但没有安装 Megatron 时测试会失败。

这一关不通过，不进入收益测试。尤其排查 CP 分母、TP 重复归约、PP 最后 stage、tied embedding、
DP reduce-scatter owned shard，以及第二个 step 的 bucket 状态。

## Agentic 重长尾数据

采用 ALFWorld 的六类真实多步任务：取放、开灯观察、清洁、加热、冷却、取两个物体。
模型必须实际调用环境工具，reward 来自环境的 `won`，不靠文本答案或随机 delay。
数据集名称本身不能证明当前模型、token cap 和并发设置下具有“很重长尾”。
因此使用**实际 rollout 计时筛选的固定压力测试子集**，同时保留自然抽样 pilot 的分布报告。
该子集用于系统测试，不能据此声称标准 ALFWorld benchmark 分数。

先抽 256 个训练任务，held-out `valid_unseen` 独立保留：

```bash
python examples/group_delivery/prepare_alfworld.py \
  --data-root "$ALFWORLD_DATA" --output data/alfworld_pilot --count 256
export DATA_DIR="$PWD/data/alfworld_pilot"
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2 bash tests/special_e2e/run_v1_group_delivery.sh \
  trainer.val_only=True trainer.val_before_train=True trainer.resume_mode=disable \
  data.val_files="$DATA_DIR/pilot.parquet" \
  trainer.experiment_name=alfworld_pilot \
  2>&1 | tee artifacts/group_delivery/pilot.log
```

pilot 只生成、调用环境和评分，不训练。每个 prompt 采样两次，按 group 最慢 session 的时延统计。
日志有 `alfworld_episode` JSON（game、uid、session、时延、动作数、成功、cap、token 数）。
若 rollout 错误、全部不调用工具或全部 reward=0，先修配方/提示和模型能力，不能当作学习正确性的证明。
环境 crash 必须导致 batch 失败；达到 episode cap 则是正常失败样本，reward=0。

```bash
python examples/group_delivery/prepare_alfworld.py \
  --data-root "$ALFWORLD_DATA" --output data/alfworld_tail \
  --pilot-log artifacts/group_delivery/pilot.log --count 256 --minimum-tail-ratio 5
export DATA_DIR="$PWD/data/alfworld_tail"
```

筛选按训练任务实测时延：80% 从较快一半采样，20% 从最慢十分之一采样；允许重复游戏但每次是独立 group。
只使用完整 n=2 的 pilot groups。写出 parquet 和 manifest，要求重采样 workload 的 pilot 估计
`P95/P50 >= 5`；达不到就报错，不注入 sleep 伪造长尾。
这只是筛选依据，后续正式运行仍须重新统计 group 时延，模型更新、prefix cache 和资源竞争都可能改变分布。
若大量样本撞同一 token/turn cap，应报告截断比例，不能把统一截断后的平坦分布叫重长尾。

## 两步 smoke 和拓扑轮换

保持 ALFWorld 服务运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash tests/special_e2e/run_v1_group_delivery.sh \
  trainer.resume_mode=disable trainer.experiment_name=alfworld_dp2 \
  2>&1 | tee artifacts/group_delivery/smoke-dp2.log

TP=2 CUDA_VISIBLE_DEVICES=0,1,2 bash tests/special_e2e/run_v1_group_delivery.sh \
  trainer.resume_mode=disable trainer.experiment_name=alfworld_tp2
PP=2 CUDA_VISIBLE_DEVICES=0,1,2 bash tests/special_e2e/run_v1_group_delivery.sh \
  trainer.resume_mode=disable trainer.experiment_name=alfworld_pp2
CP=2 CUDA_VISIBLE_DEVICES=0,1,2 bash tests/special_e2e/run_v1_group_delivery.sh \
  trainer.resume_mode=disable trainer.experiment_name=alfworld_cp2
```

验收条件：每 step `training/delivered_groups=8`、`training/optimizer_steps=1`；
日志恰好 8 对 backward begin/end，最后一个 backward end 后才有 optimizer begin/end 和 weight sync。
至少一个稳态 step 的 `training/groups_backward_started_during_rollout > 0`，证明 backward 真正在其它组生成时开始。
检查 reward 有组内方差、梯度有限、权重有更新，工具观察 mask=0，未出现版本错误、丢组或补抽。
人工结束环境服务的失败注入应让训练失败且没有本批 optimizer_end；从上一完整 checkpoint 能继续。

独立做一次 checkpoint 恢复：把两步 smoke 的 `global_step_2` 路径传给
`trainer.resume_mode=resume_path trainer.resume_from_path=... trainer.total_training_steps=3`。
恢复后第 3 步只消费新 batch，没有旧 prompt 遗留。性能测试关闭保存，避免 checkpoint 时间干扰。

## 重叠收益的对照实验

先用 DP2。相同模型、数据文件、batch、n、seed、并发、截断、GPU 分配，分别跑：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash tests/special_e2e/run_v1_group_delivery.sh \
  trainer.total_training_steps=20 trainer.save_freq=-1 trainer.resume_mode=disable \
  trainer.experiment_name=alfworld_delivery \
  2>&1 | tee artifacts/group_delivery/delivery.log

CUDA_VISIBLE_DEVICES=0,1,2 bash tests/special_e2e/run_v1_group_delivery.sh \
  trainer.v1.group_delivery.wait_for_full_batch=True \
  trainer.total_training_steps=20 trainer.save_freq=-1 trainer.resume_mode=disable \
  trainer.experiment_name=alfworld_wait_all \
  2>&1 | tee artifacts/group_delivery/wait-all.log
```

wait-all 只关闭提前交付，仍使用相同的逐组计算和一次 optimizer step，因而能隔离重叠收益。
它不是普通 `sync` 大 batch 的吞吐基线：正式评估还应报告每组通信/packing 相对整批执行的额外成本。
两个在线运行的采样结果可能不同，不能拿在线随机 rollout 做逐元素梯度等价验证；数值测试使用上一节的固定数据。

排除前 2 个 warmup step，至少重复 3 次，报告：

- step wall time 的中位数/P95、有效生成 token/s、rollout group 时延 P50/P95/P99/max。
- 有 rollout 尚未完成时开始 backward 的 group 数、累计等待时间、optimizer 与权重同步时间。
- 两张训练卡/一张 rollout 卡的利用率和峰值显存、CPU 环境占用。
- 成功率、组内 reward 方差、turn/token cap 比例、实际 `P95/P50`；没有尾部分布不算长尾验收通过。

只在这些记录齐全时报告加速比；速度没有提升也属于有效结果，需判断是 group 太小、PP 空泡、通信成本还是 rollout 卡吞吐瓶颈。

## CPU 验证记录

本地 Python 3.12、macOS CPU；没有 CUDA/Megatron、没有下载 ALFWorld 游戏。
下面的环境测试使用有状态 fake environment 验证 HTTP/session 生命周期、异常清理和 tail selection，
不证明真实 ALFWorld 的依赖安装或任务成功率。

```bash
RAY_ADDRESS=local RAY_TMPDIR=/tmp/verl-group-delivery-ray python -m pytest -q \
  tests/workers/test_group_gradient_accumulator_on_cpu.py \
  tests/workers/test_group_delivery_worker_on_cpu.py \
  tests/trainer/ppo/v1/test_group_delivery_on_cpu.py \
  tests/special_standalone/test_group_delivery_alfworld_on_cpu.py \
  tests/trainer/ppo/v1/test_separate_async_step_switch_on_cpu.py \
  tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py \
  tests/models/test_group_delivery_megatron.py
```

2026-10-27 实测：**179 passed，16 skipped，49.75 s**。
其中本功能 CPU 测试 63 项，已有 separate-async 回归 70 项，已有 ReplayBuffer 回归 46 项。
16 项 skip 全部是没有 CUDA 的 Megatron 数值测试，不计入通过数。
Python 3.12.14、PyTorch 2.14.1（macOS CPU）、TensorDict 0.10.0、Ray 2.59.0、TransferQueue 0.1.11；
GPU 环境仍使用上文锁定依赖，不能复制本机 CPU 包组合。
启动脚本已通过 Hydra 配置解析和 group_delivery 配置约束检查，`uv pip check` 通过。
仓库规定的完整 `pre-commit run` 检查通过，包括 Ruff、格式、mypy、生成配置一致性、
文档、license、device API、测试目录结构和 compileall。`git diff --check` 通过。

若准备提交上游，先按仓库 `AGENTS.md` 由提交人逐行 review 并运行相关测试；
PR 必须说明与已有工作的差异、测试结果和 AI 辅助情况。本次未创建 PR。
