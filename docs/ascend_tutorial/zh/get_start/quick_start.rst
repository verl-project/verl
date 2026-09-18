昇腾快速上手说明
=================

**Last updated:** 2026/07/14.

关键更新
--------

- 2026/06/30：新增覆盖四种常用训推后端组合，便于用户在 quickstart 阶段快速选择合适的启动脚本。
- 2026/05/13：将 quick start 和 install guidance 分开。
- 2025/12/11：verl 存量场景目前支持自动识别 NPU 设备类型，GPU 脚本在昇腾上运行，原则上不再需要显式设置 ``trainer.device=npu`` 参数，新增特性通过设置 ``trainer.device`` 仍可优先使用，逐步适配自动识别能力。


目录
--------

- `硬件支持 <#硬件支持>`_
- `快速开始 <#快速开始>`_
   - `环境准备 <#环境准备>`_ 
   - `权重准备 <#权重准备>`_
   - `数据准备 <#数据准备>`_
   - `运行方式 <#运行方式>`_
- `附录 <#附录>`_
   - `SGLang 后端使能说明 <#SGLang-后端使能说明>`_
   - `vLLM 后端脚本转换为 SGLang <#vLLM-后端脚本转换为-SGLang>`_

硬件支持
--------

- Atlas 200T A2 Box16
- Atlas 900 A2 PODc
- Atlas 800T A3



快速开始
---------------------------------

本文面向 Ascend NPU 环境，提供基于 GSM8K 和 Qwen3-0.6B 的最小化 GRPO 训练验证流程，涵盖四种训推后端组合，帮助用户快速上手。

验证范围：

- 环境入口：verl 入口是否可用；
- 数据读取：GSM8K数据集是否正确解析；
- 组件初始化：actor、rollout、reference worker 是否能初始化；
- 模型推理：vLLM-Ascend/sglang rollout 是否能生成；
- 端到端链路：训练流程是否能完成首个 step。
  
配置提示：

- 默认配置：示例脚本均使用 Qwen3-0.6B 模型和 GSM8K 数据集。
- 硬件适配：由于 A3 每卡含 2 Die，运行示例需将 ``n_gpus_per_node`` 参数设置为16（A2 每卡含 1 die）

环境准备
~~~~~~~~~~~~~~~

运行本文脚本前，请确认已完成 verl Ascend 环境安装。环境安装详见 `昇腾安装指南 <./install_guidance.rst>`_ 。

权重准备
~~~~~~~~~~~~~~~

1. 请自行从Hugging Face上下载 Qwen3-0.6B 模型权重。

2. 脚本中的默认读取权重路径为 ``~/models/Qwen/Qwen3-0.6B``，建议将权重放在该路径下。若路径不同请修改脚本中的MODEL_PATH指向本地路径。

数据准备
~~~~~~~~~~~~~~~

1. 请自行从Hugging Face上下载 GSM8K 原始数据集。

2. 执行以下命令（请根据实际数据集路径调整命令）

   .. code-block:: bash

      python3 examples/data_preprocess/gsm8k.py --local_dataset_path /download/path/hf_data/gsm8k/

   生成如下文件：

      .. code-block:: text

         ~/data/gsm8k/train.parquet
         ~/data/gsm8k/test.parquet

运行方式
~~~~~~~~~~~~~~~

1. 进入项目目录： ``cd /your/path/verl``。

2. 使能CANN环境：执行以下命令。（若您自定义了 CANN 的路径，请根据实际路径调整命令。）

   .. code-block:: bash

      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      source /usr/local/Ascend/nnal/atb/set_env.sh

3. 运行脚本：当前提供四种常用训推后端组合，请参考以下表格选择对应脚本：

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 60

   * - 组合
     - 训练后端
     - rollout 后端
     - 运行方式
   * - vLLM + FSDP2
     - FSDP2
     - vLLM-Ascend
     - bash tests/special_npu/quick_start/run_qwen3_0_6b_fsdp2_vllm_ascend.sh
   * - vLLM + Megatron
     - Megatron
     - vLLM-Ascend
     - bash tests/special_npu/quick_start/run_qwen3_0_6b_megatron_vllm_ascend.sh
   * - SGLang + FSDP2
     - FSDP2
     - SGLang
     - bash tests/special_npu/quick_start/run_qwen3_0_6b_fsdp2_sglang_ascend.sh
   * - SGLang + Megatron
     - Megatron
     - SGLang
     - bash tests/special_npu/quick_start/run_qwen3_0_6b_megatron_sglang_ascend.sh

脚本内具体参数说明详见 `训练配置参数与指标说明 <../dev_guide/model_dev/parameter_and_metrics.md>`_。

多节点任务拉起详见 `多机任务拉起操作指南 <../model_support/examples/multi-machine_task_startup_practice.rst>`_。

附录
-------------------------------------------

SGLang 后端使能说明
~~~~~~~~~~~~~~~~~~~~~~~~

当前 verl 已解析推理常见参数，详见 `async_sglang_server.py <../../../../verl/workers/rollout/sglang_rollout/async_sglang_server.py>`_ 中 ``ServerArgs`` 初始化传参。

其他 `SGLang 参数 <https://github.com/sgl-project/sglang/blob/v0.5.10/docs/advanced_features/server_arguments.md>`_ 均可通过 ``engine_kwargs`` 进行参数传递。


vLLM 后端脚本转换为 SGLang
~~~~~~~~~~~~~~~~~~~~~~~~

如需自行将 vLLM 后端推理脚本转换为 SGLang，需要添加或修改以下参数。

.. code-block:: bash

   # 必须
   actor_rollout_ref.rollout.name=sglang \
   +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend" \

   # 可选
   # 使能推理 EP，详细使用方法见：
   # https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md
   ++actor_rollout_ref.rollout.engine_kwargs.sglang.deepep_mode="auto" \
   ++actor_rollout_ref.rollout.engine_kwargs.sglang.moe_a2a_backend="deepep" \

   # MoE 模型多 DP 时必须设置为 True（默认为False）
   +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_attention=False \   