Ascend Quickstart
=================

**Last updated:** 05/13/2026.

关键更新
--------

- 2026/05/13：将 quick start 和 install guidance 分开。
- 2025/12/11：verl 存量场景目前支持自动识别 NPU 设备类型，GPU 脚本在昇腾上运行，原则上不再需要显式设置 ``trainer.device=npu`` 参数，新增特性通过设置 ``trainer.device`` 仍可优先使用，逐步适配自动识别能力。

硬件支持
--------

- Atlas 200T A2 Box16
- Atlas 900 A2 PODc
- Atlas 800T A3

Ascend Quickstart with vLLM Backend
===================================

基础验证场景
------------

如需快速验证环境和基础链路，可以使用 Qwen2.5-0.5B GRPO 场景。

该场景用于检查：

- verl 入口是否可用；
- 数据是否可读取；
- actor、rollout、reference worker 是否能初始化；
- vLLM-Ascend rollout 是否能生成；
- 训练链路是否能完成首个 step。

准备 GSM8K 数据
---------------

.. code-block:: bash

   python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

生成文件：

.. code-block:: text

   ~/data/gsm8k/train.parquet
   ~/data/gsm8k/test.parquet

启动 Qwen2.5-0.5B GRPO 基础验证
--------------------------------

.. code-block:: bash

   set -x

   python3 -m verl.trainer.main_ppo \
       algorithm.adv_estimator=grpo \
       data.train_files=$HOME/data/gsm8k/train.parquet \
       data.val_files=$HOME/data/gsm8k/test.parquet \
       data.train_batch_size=128 \
       data.max_prompt_length=512 \
       data.max_response_length=128 \
       data.filter_overlong_prompts=True \
       data.truncation='error' \
       actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
       actor_rollout_ref.actor.optim.lr=5e-7 \
       actor_rollout_ref.model.use_remove_padding=False \
       actor_rollout_ref.actor.entropy_coeff=0.001 \
       actor_rollout_ref.actor.ppo_mini_batch_size=64 \
       actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
       actor_rollout_ref.actor.use_kl_loss=True \
       actor_rollout_ref.actor.kl_loss_coef=0.001 \
       actor_rollout_ref.actor.kl_loss_type=low_var_kl \
       actor_rollout_ref.model.enable_gradient_checkpointing=True \
       actor_rollout_ref.actor.fsdp_config.param_offload=False \
       actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
       actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
       actor_rollout_ref.rollout.enable_chunked_prefill=False \
       actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
       actor_rollout_ref.rollout.name=vllm \
       actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
       actor_rollout_ref.rollout.n=5 \
       actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
       actor_rollout_ref.ref.fsdp_config.param_offload=True \
       algorithm.kl_ctrl.kl_coef=0.001 \
       trainer.critic_warmup=0 \
       trainer.logger=console \
       trainer.project_name='verl_grpo_example_gsm8k' \
       trainer.experiment_name='qwen2_5_0_5b_grpo_vllm_ascend' \
       trainer.n_gpus_per_node=8 \
       trainer.nnodes=1 \
       trainer.save_freq=-1 \
       trainer.test_freq=5 \
       trainer.total_epochs=1 $@

关键配置
--------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 配置项
     - 说明
   * - ``actor_rollout_ref.rollout.name=vllm``
     - 使用 vLLM 作为 rollout 后端
   * - ``actor_rollout_ref.rollout.tensor_model_parallel_size``
     - rollout 推理侧张量并行大小
   * - ``actor_rollout_ref.rollout.gpu_memory_utilization``
     - rollout 推理侧可使用的设备显存比例
   * - ``actor_rollout_ref.rollout.n``
     - 每个 prompt 生成的 response 数量
   * - ``trainer.n_gpus_per_node``
     - 昇腾场景中表示每节点使用的 NPU 数量
   * - ``trainer.nnodes``
     - 节点数量

Ascend Quickstart with SGLang Backend
=====================================

最佳实践
--------

我们提供 `最佳实践 <https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/model_support/examples/ascend_sglang_best_practices.rst>`_ 作为参考。

环境变量与参数
--------------

当前 NPU 上支持 SGLang 后端必须添加以下环境变量。

.. code-block:: bash

   # 支持 NPU 单卡多进程
   # https://www.hiascend.com/document/detail/zh/canncommercial/850/commlib/hcclug/hcclug_000091.html
   export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
   export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

   # 规避 Ray 在 device 侧调用无法根据 is_npu_available 接口识别设备可用性
   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

   # 根据当前设备和需要卡数定义
   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

   # 使能推理 EP 时需要
   export SGLANG_DEEPEP_BF16_DISPATCH=1

当前 verl 已解析推理常见参数，详见 `async_sglang_server.py <https://github.com/verl-project/verl/blob/main/verl/workers/rollout/sglang_rollout/async_sglang_server.py>`_ 中 ``ServerArgs`` 初始化传参。

其他 `SGLang 参数 <https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md>`_ 均可通过 ``engine_kwargs`` 进行参数传递。

vLLM 后端脚本转换为 SGLang
--------------------------

vLLM 后端推理脚本转换为 SGLang，需要添加或修改以下参数。

.. code-block:: bash

   # 必须
   actor_rollout_ref.rollout.name=sglang \
   +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend" \

   # 可选
   # 使能推理 EP，详细使用方法见：
   # https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README_CN.md
   ++actor_rollout_ref.rollout.engine_kwargs.sglang.deepep_mode="auto" \
   ++actor_rollout_ref.rollout.engine_kwargs.sglang.moe_a2a_backend="deepep" \

   # MoE 模型多 DP 时必须设置为 True
   +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_attention=False \

   # chunked_prefill 默认关闭
   +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=-1