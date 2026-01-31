Ascend Quickstart
===================================

Last updated: 12/11/2025.

我们在 verl 上增加对华为昇腾设备的支持。


关键更新
----------------------------------

2025/12/11：verl 存量场景目前支持自动识别 NPU 设备类型， GPU 脚本在昇腾上运行，原则上不再需要显式设置 trainer.device=npu 参数，新增特性通过设置 trainer.device 仍可优先使用，逐步适配自动识别能力。

    [说明] 自动识别 NPU 设备类型的前提，是运行程序所在环境包含 torch_npu 软件包。如不包含该软件包，仍需显式指定 trainer.device=npu 参数。

硬件支持
-----------------------------------

Atlas 200T A2 Box16

Atlas 900 A2 PODc

Atlas 800T A3


安装流程
-----------------------------------


DockerFile镜像构建 & 使用
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如需要通过 DockerFile 构建镜像，或希望使用基于 verl 构建的镜像，请参考 `文档 <https://github.com/volcengine/verl/tree/main/docs/ascend_tutorial/dockerfile_build_guidance.rst>`_ 。


安装基础环境
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. 基础环境涉及以下软件包，请参考 `文档 <https://gitcode.com/Ascend/pytorch>`_ 安装。

    +---------------+----------------------+
    | software      | version              |
    +---------------+----------------------+
    | Python        | >= 3.10, <3.12       |
    +---------------+----------------------+
    | CANN          | == 8.3.RC1           |
    +---------------+----------------------+
    | torch         | == 2.7.1             |
    +---------------+----------------------+
    | torch_npu     | == 2.7.1             |
    +---------------+----------------------+

2. （可选）在 x86 平台安装时，pip 需要配置额外的源，指令如下：

    .. code-block:: bash

        pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"


安装其他软件包
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

基础环境准备完毕后，需要通过指令安装以下软件包：

    +---------------+----------------------+
    | torchvision   | == 0.22.1            |
    +---------------+----------------------+
    | triton-ascend | == 3.2.0rc4          |
    +---------------+----------------------+
    | transformers  | latest release       |
    +---------------+----------------------+

    安装指令：

    .. code-block:: bash

        # 安装torchvision，版本需要和torch匹配
        pip install torchvision==0.22.1

        # 清理环境上可能存在的历史triton/triton-ascend软件包残留
        pip uninstall -y triton triton-ascend

        # 安装triton-ascend，不需要单独安装triton
        pip install triton-ascend==3.2.0rc4


安装 vllm & vllm-ascend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. 需确保CANN ascend-toolkit 和 nnal 环境变量被激活，对于CANN默认安装路径 /usr/local/Ascend 而言，激活指令如下：

    .. code-block::

        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        source /usr/local/Ascend/nnal/atb/set_env.sh

2. vllm 源码安装指令：

    .. code-block:: bash

        git clone --depth 1 --branch v0.11.0 https://github.com/vllm-project/vllm.git
        cd vllm && VLLM_TARGET_DEVICE=empty pip install -v -e . && cd ..

3. vllm-ascend 源码安装指令：

    .. code-block:: bash

        git clone --depth 1 --branch v0.11.0rc1 https://github.com/vllm-project/vllm-ascend.git
        cd vllm-ascend && pip install -v -e . && cd ..


安装 MindSpeed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MindSpeed 源码安装指令：

    .. code-block:: bash

        # 下载 MindSpeed，切换到指定commit-id，并下载 Megatron-LM
        git clone https://gitcode.com/Ascend/MindSpeed.git
        cd MindSpeed && git checkout f2b0977e && cd ..
        git clone --depth 1 --branch core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git

        # 安装 MindSpeed & Megatron
        pip install -e MindSpeed

        # 将 Megatron-LM 源码路径配置到 PYTHONPATH 环境变量中
        export PYTHONPATH=$PYTHONPATH:"$(pwd)/Megatron-LM"

        # （可选）如希望 shell 关闭，或系统重启后，PYTHONPATH 环境变量仍然生效，建议将它添加到 .bashrc 配置文件中
        echo "export PYTHONPATH=$PYTHONPATH:\"$(pwd)/Megatron-LM\"" >> ~/.bashrc

        # 安装 mbridge
        pip install mbridge

MindSpeed 对应 Megatron-LM 后端使用场景，使用方式如下：

    1. 使能 verl worker 模型 ``strategy`` 配置为 ``megatron`` ，例如 ``actor_rollout_ref.actor.strategy=megatron``。

    2. MindSpeed 自定义入参可通过 ``override_transformer_config`` 参数传入，例如对 actor 模型开启 FA 特性可使用 ``+actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True``。

    3. 更多特性信息可参考 `MindSpeed & verl 文档 <https://gitcode.com/Ascend/MindSpeed/blob/master/docs/user-guide/verl.md>`_ 。


安装verl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git clone --depth 1 https://github.com/volcengine/verl.git
    cd verl && pip install -r requirements-npu.txt && pip install -v -e . && cd ..


昇腾暂不支持生态库说明
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

verl 中昇腾暂不支持生态库如下：

    +---------------+----------------+
    | software      | description    |
    +---------------+----------------+
    | flash_attn    | not supported  |
    +---------------+----------------+
    | liger-kernel  | not supported  |
    +---------------+----------------+

    1. 不支持通过 flash_attn 使能 flash attention 加速，支持通过 transformers 使用。
    2. 不支持 liger-kernel 使能。


快速开始
-----------------------------------
正式使用前，建议您通过对Qwen2.5-0.5B GRPO的训练尝试以检验环境准备和安装的正确性。

1.下载数据集并将数据集预处理为parquet格式，以便包含计算RL奖励所需的必要字段

    .. code-block:: bash

        python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

2.执行训练

    .. code-block:: bash

        set -x

        export VLLM_ATTENTION_BACKEND=XFORMERS

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
            trainer.experiment_name='qwen2_7b_function_rm' \
            trainer.n_gpus_per_node=8 \
            trainer.nnodes=1 \
            trainer.save_freq=-1 \
            trainer.test_freq=5 \
            trainer.total_epochs=1 $@



算法支持现状
-----------------------------------

**表1** RL类算法

    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+
    | algorithm             |         model           | download link                                                    |   actor.strategy  |   rollout.name    |   shell location                                                                                                                             |     hardware             |
    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+
    |   GRPO                | Qwen2.5-7B-instruct     |`7B <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>`_           |        FSDP       |    vllm-ascend    |`qwen2_5_7b_grpo_npu <https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen2_5_7b_grpo_npu.sh>`_                        |    Atlas 200T A2 Box16   |
    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+
    |   GRPO                | Qwen2.5-32B-instruct    |`32B <https://huggingface.co/Qwen/Qwen2.5-32B-Instruct>`_         |        FSDP       |    vllm-ascend    |`qwen2_5_32b_grpo_npu <https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen2_5_32b_grpo_npu.sh>`_                      |    Atlas 200T A2 Box16   |
    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+
    |   GRPO                | Qwen2.5-VL-3B-instruct  |`3B <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>`_        |        FSDP       |    vllm-ascend    |`qwen2_5_vl_3b_npu <https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen2_5_vl_3b_npu.sh>`_                            |    Atlas 200T A2 Box16   |
    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+
    |   GRPO                | Qwen2.5-VL-7B-instruct  |`7B <https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct>`_        |        FSDP       |    vllm-ascend    |`qwen2_5_vl_7b_npu <https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen2_5_vl_7b_npu.sh>`_                            |    Atlas 200T A2 Box16   |
    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+
    |   GRPO                | Qwen2.5-VL-32B-instruct |`32B <https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct>`_      |        FSDP       |    vllm-ascend    |`qwen2_5_vl_32b_npu <https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen2_5_vl_32b_npu.sh>`_                          |    Atlas 200T A2 Box16   |
    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+
    |   GRPO                | Qwen3-4B                |`4B <https://huggingface.co/Qwen/Qwen3-4B>`_                      |        FSDP       |    vllm-ascend    |`qwen3-4B_npu <https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen3_4b_grpo_vllm_1k_npu.sh>`_                         |    Atlas 800T A3         |
    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+
    |   GRPO                | Qwen3-8B                |`8B <https://huggingface.co/Qwen/Qwen3-8B>`_                      |        FSDP       |    vllm-ascend    |`qwen3_8b_vllm_npu <https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen3-8b_npu.sh>`_                                 |    Atlas 200T A2 Box16   |
    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+
    |   GRPO                | Qwen3-8B                |`8B <https://huggingface.co/Qwen/Qwen3-8B>`_                      |        FSDP       |    sglang         |`qwen3_8b_sglang_npu <https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen3_8b_grpo_sglang_32k_spmd_npu.sh>`_          |    Atlas 200T A2 Box16   |
    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+
    |   GRPO                | Qwen3-32B               |`32B <https://huggingface.co/Qwen/Qwen3-32B>`_                    |        FSDP       |    vllm-ascend    |`qwen3-32B_npu <https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen3-32b_npu.sh>`_                                    |    Atlas 200T A2 Box16   |
    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+
    |   GRPO                | DeepSeekv3-671B         |`671B <https://huggingface.co/deepseek-ai/DeepSeek-V3>`_          |        Megatron   |    vllm-ascend    |`deepseek_v3_megatron_npu <https://github.com/verl-project/verl-recipe/blob/main//r1_ascend/run_deepseekv3_671b_grpo_megatron_npu.sh>`_       |    Atlas 200T A2 Box16   |
    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+
    |   DAPO                | Qwen2.5-7B-instruct     |`7B <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>`_           |        FSDP       |    vllm-ascend    |`qwen2.5_7b_npu <https://github.com/verl-project/verl-recipe/blob/main//dapo/run_dapo_qwen2.5_7b_npu.sh>`_                                    |    Atlas 200T A2 Box16   |
    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+
    |   DAPO                | Qwen2.5-32B             |`32B <https://huggingface.co/Qwen/Qwen2.5-32B>`_                  |        FSDP       |    vllm-ascend    |`qwen2.5_32b_npu <https://github.com/verl-project/verl-recipe/blob/main//dapo/run_dapo_qwen2.5_32b_npu.sh>`_                                  |    Atlas 200T A2 Box16   |
    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+
    |   DAPO                | Qwen3-8B-base           |`8B <https://huggingface.co/Qwen/Qwen3-8B>`_                      |        FSDP       |    vllm-ascend    |`qwen3_8b_npu <https://github.com/verl-project/verl-recipe/blob/main//dapo/run_dapo_qwen3_8b_base_npu.sh>`_                                   |    Atlas 200T A2 Box16   |
    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+
    |   DAPO                | Qwen3-14B-base          |`14B <https://huggingface.co/Qwen/Qwen3-14B>`_                    |        FSDP       |    vllm-ascend    |`qwen3_14b_npu <https://github.com/verl-project/verl-recipe/blob/main//dapo/run_dapo_qwen3_14b_base_npu.sh>`_                                 |    Atlas 200T A2 Box16   |
    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+
    |   DAPO                | Qwen3-30B-A3B-base      |`30B <https://huggingface.co/Qwen/Qwen3-30B-A3B>`_                |        FSDP       |    vllm-ascend    |`qwen3_30b_fsdp_npu <https://github.com/verl-project/verl-recipe/blob/main//dapo/run_dapo_qwen3_moe_30b_base_fsdp_npu.sh>`_                   |    Atlas 200T A2 Box16   |
    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+
    |   DAPO                | Qwen3-30B-A3B-base      |`30B <https://huggingface.co/Qwen/Qwen3-30B-A3B>`_                |        Megatron   |    vllm-ascend    |`qwen3_30b_megatron_npu <https://github.com/verl-project/verl-recipe/blob/main//dapo/run_dapo_qwen3_moe_30b_megatron_npu.sh>`_                |    Atlas 200T A2 Box16   |
    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+
    |   PPO                 | Qwen3-8B                |`8B <https://huggingface.co/Qwen/Qwen3-8B>`_                      |        FSDP       |    vllm-ascend    |`qwen3_8b_ppo_npu <https://github.com/volcengine/verl/blob/main/examples/ppo_trainer/run_qwen3-8b_npu.sh>`_                                   |    Atlas 900 A2 PODc     |
    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+
    |   One_Step_Off_Policy | Qwen3-8B                |`8B <https://huggingface.co/Qwen/Qwen3-8B>`_                      |        FSDP2      |    vllm-ascend    |`qwen3_8b_fsdp2_npu <https://github.com/verl-project/verl-recipe/blob/main//one_step_off_policy/shell/grpo_qwen3_8b_gsm8k_fsdp2_8_8_npu.sh>`_ |    Atlas 800T A3         |
    +-----------------------+-------------------------+------------------------------------------------------------------+-------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+--------------------------+

**表2** SFT类算法

    +-----------+-------------------------+------------------------------------------------------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+----------------------+
    | algorithm |         model           |  download link                                                   |   actor.strategy  |    shell location                                                                                                                            |     hardware         |
    +-----------+-------------------------+------------------------------------------------------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+----------------------+
    |  SFT-PEFT | Qwen3-8B                |`8B <https://huggingface.co/Qwen/Qwen3-8B>`_                      |        FSDP       |`sft_peft_sp2_npu <https://github.com/volcengine/verl/blob/main/examples/sft/gsm8k/run_qwen3_8b_sft_peft_sp2_npu.sh>`_                        |   Atlas 900 A2 PODc  |
    +-----------+-------------------------+-------------------------+----------------------------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+----------------------+
    | ReTool-SFT| Qwen2-7B-instruct       |`7B <https://huggingface.co/Qwen/Qwen2-7B-Instruct>`_             |        FSDP       |`qwen2_7b_sft_npu <https://github.com/verl-project/verl-recipe/blob/main/retool/run_qwen2_7b_sft_npu.sh>`_                                    |   Atlas 900 A2 PODc  |
    +-----------+-------------------------+-------------------------+----------------------------------------+-------------------+----------------------------------------------------------------------------------------------------------------------------------------------+----------------------+


声明
-----------------------------------
verl中提供的ascend支持代码、Dockerfile、镜像皆为参考样例，如在生产环境中使用请通过官方正式途径沟通，谢谢。
