# Verl real W4A4 提交与启动踩坑记录（2026-09-01）

本文记录当天所有已遇到的问题、证据、处理方式和验证状态。后续诊断必须在这里追加，不能仅保留在聊天或临时日志中。

## 不变量

- 训练：Megatron，48 个 routed-expert MLP 层 W4A4；attention/dense BF16。
- Rollout：vLLM 0.26，`nvfp4_per_token`，native `reload_weights`。
- R3 on；Slime 三项 loss 为 0/3；token-mean；TIS token level。
- CUDA graph 开启，模式为 `FULL_DECODE_ONLY`；`max_num_seqs=128`。
- 正式规模为 8 nodes × 4 GPUs，EP=4，32 prompts × 16 responses，full Adam。
- 分区候选顺序：`36x2-a01r,tcpo,batch`。
- 分支中不得出现三项 loss 的实现或提交历史。

## 镜像血缘，避免“为什么今天才建镜像”的混淆

这不是今天才第一次实现或运行 W4A4。已跑通 20 steps 的 v8 是基线运行时；今天构建的是在该已验证镜像上的增量诊断层。

1. v8：vLLM 0.26 + MCore 0.18 + TE 2.18，已完成 20-step 训练。
2. v10：以 v8 为 base，只加入 vLLM 合入 0.26 之后的 #50029 和 #50074 精确 backport。
3. v11：以 v10 为 base，只给 FlashInfer TRTLLM NVFP4 MoE 的两个调用点加 `enable_pdl=False`，用于单变量诊断。没有升级任何依赖，也没有关闭 CUDA graph。

镜像均为版本化只写一次；不能覆盖旧镜像。这样能确保 job、source commit、lock hash、镜像 checksum 一一对应。

## Job 时间线

| Job | 阶段 | 结果 | 关键结论 |
| --- | --- | --- | --- |
| 2694027 | v8，8-node 20 steps | COMPLETED，43m | 原方案确实能完整启动、训练、refit、保存 full-Adam checkpoint；不是“从未跑通”。 |
| 2694199 | v9 long 首次启动 | 卡在 rollout server 启动 | 32 个 server 中偶发少数不 ready。 |
| 2694385 | v9 long 替代启动 | 再次卡住 | 换了一组节点仍复现，排除单一坏节点。 |
| 2694200/1/2 | v9 下游 chain | 已取消 | 上游没有 checkpoint/W&B，不允许下游误启动。 |
| 2694765 | v10 probe | COMPLETED，50s | 基础镜像/架构/工具可用。 |
| 2694768 | v10 build | COMPLETED，1m31s | #50029/#50074 精确写入，依赖版本未漂移。 |
| 2694779 | v10 preflight | COMPLETED，2m52s | 48 passed, 5 skipped；TE、packing、reload lifecycle、R3 合同通过。 |
| 2694788 | v10 8-node startup | 30/32 ready，取消 | #50029 并不能修复启动挂死；#50074 仍是 reload 正确性必需项。 |
| 2694812 | v11 probe | COMPLETED，48s | PDL-off 诊断链的 base 可用。 |
| 2694814 | v11 build | COMPLETED，1m31s | v11 镜像和 checksum 已生成，两个 PDL-off marker 存在。 |
| 2694819 | v11 preflight | COMPLETED，3m09s | 两处 PDL API/源码断言、TE W4A4 两轮前反向、packing/reload、R3 均通过；48 passed, 5 skipped。 |
| 2694835 | v11 8-node 3-step | COMPLETED，21m53s | 32/32 server；CG/max128；3 steps；多次 native refit；R3 48 layers；399 GiB full-Adam checkpoint；Slurm exit 0。 |

## 坑 1：正式 long job 没有 W&B，不等于训练跑了很久

### 现象

Slurm job 处于 RUNNING，但 W&B 没有 run、没有曲线。

### 证据与结论

W&B 初始化发生在 32 个 vLLM server 全部 ready 之后。挂死时 actor 状态是 `launch_server FINISHED 30, RUNNING 2` 或 `FINISHED 31, RUNNING 1`，所以训练 step 0 尚未开始，W&B 也不会注册。这类 job 的 Slurm elapsed 不能解释为训练耗时。

### 处理

- 启动阶段同时看 Ray actor 状态、每张 GPU 显存、GPU utilization 和 actor 日志，不能只看 Slurm RUNNING。
- 未产生 checkpoint/W&B 的上游 job 必须取消其 dependency chain。

## 坑 2：偶发卡死不是坏节点，也不是单纯的在线权重量化慢

### 现场

- 两次发生在不同分区/不同节点组合。
- 32 个 `WorkerDict` 都完成 model init。
- 卡住的 server worker 保持约 99–100% CPU/GPU utilization，显存约 21.7 GiB；成功 worker 随后到约 121–128 GiB 并 ready。
- 卡住前最后一条稳定日志为 `TRT-LLM fused MoE cooperative launch SM allocation...`；成功 worker 随后进入 attention/KV profile 和 API ready。

### 已排除

- 单一坏节点：同一节点上的其他 GPU 能启动，且复现节点每次不同。
- 只由 vLLM v0.26 的 whole-tensor expert packing 引起：加入 #50029 后 v10 仍然 30/32 卡住。

### 当前最强假设

FlashInfer TRTLLM NVFP4 MoE 在 SM100、128 bucket/concurrency、PDL/cooperative-launch 路径上的间歇 kernel hang。上游已有同类报告：`flashinfer_trtllm` 在 64/128 concurrency 出现 hang/静默错误，而 `flashinfer_cutlass` 不复现。

### 单变量验证

v11 只给两处 TRTLLM NVFP4 MoE API 显式传 `enable_pdl=False`。CUDA graph、`max_num_seqs=128`、backend、quantization、reload 和训练合同全部不变。只有 32/32 startup 通过后，才能把 PDL 视为强因果；一次通过后还需重复启动或完成 multi-step reload，防止把概率事件误判成修复。

### v11 验证结果

Job 2694835 在 v10 曾经稳定暴露 30/32 的同一 8-node 规模上完成：

- `vLLMHttpServer.launch_server` 32/32 FINISHED；`actor_rollout_init_model` 32/32 FINISHED。
- 32 个 server 都完成 `FULL_DECODE_ONLY` CUDA graph capture 和 48-layer W4A4 rollout attestation。
- 初始 refit 和 step 后 refit 都接收完整的 18,432 个 BF16 expert weights，并使用 native reload/post-finalize ACK。
- 完成 3 个 global steps、R3 48-layer replay、最终 validation 和 global_step_3 full-Adam checkpoint。
- checkpoint 约 399 GiB；optimizer `dist_ckpt` 35 files、366,450,559,551 bytes。

因此 PDL 是本次 startup hang 的强因果变量。正式化前仍建议再跑一个有非零 advantage 的短实验，因为 2694835 只覆盖了零更新路径。

## 坑 3：#50029 与 #50074 的作用不能混为一谈

- vLLM #50029：改为逐 expert 直接 packing，减少整块 FP32/BF16 临时量并改善精度/显存。v10 证明它不是这次 startup hang 的充分修复，但仍应保留。
- vLLM #50074：reload 时复用同一个 MoE kernel object，避免 CUDA graph 持有旧 kernel/旧权重引用。它解决的是 refit 后的正确性和非有限输出风险，是 native reload 路径的必要项，也应保留。

## 坑 4：诊断不能通过关 CUDA graph 或改小 max_num_seqs 偷跑

之前关 CG 会显著改变性能和真实路径；用户明确要求 CG 必须打开，`max_num_seqs` 也必须与 BF16 对齐。因此：

- 所有正式和诊断配置固定 `enforce_eager=False`、`FULL_DECODE_ONLY`。
- 固定 `max_num_seqs=128`。
- PDL 是 MoE kernel 的 launch-overlap 开关，不等于 CUDA graph；v11 关闭 PDL 不改变 CG 合同。

## 坑 5：共享 `/tmp` 被上游 partial clone 的按需 fetch 填满

### 现象

追 vLLM 全历史时出现 `No space left on device`，`/tmp` 2 GiB 达到 100%。原因是 partial clone 的 `git log -S` 触发 promisor remote 补拉大量对象。

### 处理

- 已删除本次创建的 `/tmp/vllm_nvfp4_history_20260901` 和 `/tmp/flashinfer_0614_probe`，释放空间。
- 后续大型上游 clone 放到 Lustre 的明确临时目录；优先使用 tag 下的目标文件或 GitHub 官方页面，避免在共享 `/tmp` 做全历史 blob fetch。

## 坑 6：pre-commit 的唯一失败来自登录环境缺 hydra

针对新增文件的 ruff、format、mypy、license、device API、compile 等均通过。`autogen-trainer-cfg` 在登录环境报 `ModuleNotFoundError: hydra`，这是已知 host 环境缺包，并非改动导致配置漂移。提交时显式 `SKIP=autogen-trainer-cfg`；真正的 config、pytest、GPU 回归全部放在 scheduler preflight 中执行，不能在登录节点运行 pytest/Ray/GPU 测试。

## 坑 7：v11 复用 submit harness 的两个纯脚本错误

两次都在 `sbatch` 前失败，没有创建 scheduler job：

1. `readonly BUNDLE` 后再赋值会报 `readonly variable`。正确写法是先赋值，再 `readonly BUNDLE`。
2. v10 的 `submit.sh` 文件 mode 不是 executable，直接 `exec path` 报 `Permission denied`。v11 改为 `exec bash path`，不依赖被复用脚本的 executable bit。

这两项已分别由 commit `4f3f2124` 和 `729ff9b3` 修复。以后复用 bundle 时必须先在登录节点做 `bash -n`、`shellcheck`，并实际走一次不提交或 probe 入口验证 wrapper。

## 坑 8：全 -1 reward 会让 3-step 看起来“训练没动”

Job 2694835 的 512 个 responses 每一步都达到 20,480 token 上限，reward 全为 -1。prompt group 内 reward 相同，GRPO advantage 因而全为 0；对应结果是：

- `actor/loss=0`、`actor/grad_norm=0`；
- step 后 refit 的 `changed=0`，packed fingerprint 不变；
- response length 一直为 20,480；final AIME accuracy 为 0。

这不是 native reload 失败：初始 dummy rollout model 的 `refit=0 changed=1`，后续每轮仍完成 32/32 export/reload/ACK，只是 optimizer 合法地产生零更新。这个 job 可以证明启动、CG、W4A4 rollout、refit、R3、训练循环和 checkpoint 闭环，不能证明 reward/response 曲线会正常学习。下一次短测必须使用能产生非零 reward variance 的数据/奖励配置，并检查 `grad_norm>0` 和至少一次 step 后 `changed=1`。

## 坑 9：训练成功后的 teardown traceback 不能误判成训练失败

2694835 在 W&B 同步和 Ray job success 之后，清理 DataLoader/Ray descendants 时出现：

- `DataLoader worker ... killed by signal: Killed`；
- W&B service teardown `BrokenPipeError`；
- `srun: forcing job termination`。

这些发生在以下语义 gate 之后：Training Progress 100%、Final validation、W&B finished、full-Adam checkpoint PASS、Ray job succeeded。train harness 会在这些 gate 全部通过后接受 descendant cleanup signal；该 job 的 Slurm 状态最终为 `COMPLETED 0:0`。判断标准必须是最终 Slurm exit + semantic/checkpoint gates，不能只搜索 traceback 字样。

## 下一步验收门

1. 已完成：v11 preflight PASS，并打印两处 `enable_pdl` API/源码断言。
2. 已完成：v11 8-node startup 32/32 ready，保持 CG on 和 max 128。
3. 已完成零更新路径：multi-step refit、R3、有限值扫描和 full-Adam checkpoint。
4. 待完成：使用非零 reward variance 的短测验证 optimizer 实际更新，以及 step 后 reload `changed=1`。
5. 待完成：再重复一次 32-server startup 或用上述非零更新短测同时覆盖概率性 startup gate。
6. 通过后从诊断 bundle 派生正式、全新命名的 long-run chain；不要直接把 `diag` bundle 当正式 recipe。

## 关键路径

- v10 日志：`ray_log/verl_real_nvfp4_r3_nativeonline_post026_20260901_v10/`
- v11 日志：`ray_log/verl_real_nvfp4_r3_nativeonline_pdl_off_diag_20260901_v11/`
- v11 state：`run_state/verl_real_nvfp4_r3_nativeonline_pdl_off_diag_20260901_v11/`
- v11 bundle：`examples/real_nvfp4/jobs/r3_nativeonline_pdl_off_diag_20260901_v11/`
- PDL patch：`examples/real_nvfp4/runtime_backports/disable_vllm_trtllm_nvfp4_moe_pdl.py`
