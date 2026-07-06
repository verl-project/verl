Ascend Quickstart
=================

**Last updated:** 07/02/2026.

关键更新
--------

- 2026/06/30：新增覆盖四种常用训推后端组合，便于用户在 quickstart 阶段快速选择合适的启动脚本。
- 2026/05/13：将 quick start 和 install guidance 分开。
- 2025/12/11：verl 存量场景目前支持自动识别 NPU 设备类型，GPU 脚本在昇腾上运行，原则上不再需要显式设置 ``trainer.device=npu`` 参数，新增特性通过设置 ``trainer.device`` 仍可优先使用，逐步适配自动识别能力。


目录
--------

- `硬件支持 <#硬件支持>`_
- `Qwen3-0.6B GSM8K GRPO Quick Start <#qwen3-06b-gsm8k-grpo-quick-start>`_
   - `权重准备 <#权重准备>`_
   - `数据准备 <#数据准备>`_
   - `运行方式 <#运行方式>`_
- `SGLang 后端使能说明 <#sglang-后端使能说明>`_
   - `vLLM 后端脚本转换为 SGLang <#vllm-后端脚本转换为-sglang>`_

硬件支持
--------

- Atlas 200T A2 Box16
- Atlas 900 A2 PODc
- Atlas 800T A3



Qwen3-0.6B GSM8K GRPO Quick Start
---------------------------------

本文面向 Ascend NPU 环境，提供基于 GSM8K 和 Qwen3-0.6B 的最小 GRPO 训练验证流程。

文档覆盖四种常用训推后端组合，便于用户在 quickstart 阶段快速选择合适的启动脚本。

运行本文脚本前，请确认已完成 verl Ascend 环境安装。
环境安装详见 `install_guidance <./install_guidance.rst>`_ 。

四个脚本均默认使用 ``Qwen/Qwen3-0.6B`` 和 GSM8K 数据集进行基础链路验证。

主要用于检查：

- verl 入口是否可用；
- 数据是否可读取；
- actor、rollout、reference worker 是否能初始化；
- vLLM-Ascend/sglang rollout 是否能生成；
- 训练链路是否能完成首个 step。

权重准备
~~~~~~

权重需自行从huggingface上下载

脚本中的默认读取权重路径为 ``~${HOME}/models/Qwen/Qwen3-0.6B``

建议将权重放在该路径下，或者修改脚本中MODEL_PATH指向本地路径


数据准备
~~~~~~

.. code-block:: bash

   python3 examples/data_preprocess/gsm8k.py --local_dataset_path /download/path/hf_data/gsm8k/

gsm8k原始数据集需自行从huggingface上下载

生成文件：

.. code-block:: text

   ~/data/gsm8k/train.parquet
   ~/data/gsm8k/test.parquet

运行方式
~~~~~~

相关脚本均已放置于 ``tests/special_npu/quick_start/`` 路径下

首先进入verl路径： ``cd /your/path/verl``

使能CANN环境: 如果您自定义了CANN的路径，请根据自定义路径修改以下使能命令

.. code-block:: bash

   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh

Quick Start 当前提供四种常用训推后端组合。用户可根据训练后端和 rollout 后端选择对应脚本

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

脚本内具体参数说明详见 `训练配置参数与指标说明 <https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/dev_guide/model_dev/parameter_and_metrics.md>`_

SGLang 后端使能说明
-------------------------------------------

当前 verl 已解析推理常见参数，详见 `async_sglang_server.py <https://github.com/verl-project/verl/blob/main/verl/workers/rollout/sglang_rollout/async_sglang_server.py>`_ 中 ``ServerArgs`` 初始化传参。

其他 `SGLang 参数 <https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md>`_ 均可通过 ``engine_kwargs`` 进行参数传递。

vLLM 后端脚本转换为 SGLang
~~~~~~~~

如需自行将 vLLM 后端推理脚本转换为 SGLang，需要添加或修改以下参数。

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


