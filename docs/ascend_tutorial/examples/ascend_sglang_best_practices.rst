Ascend SGLang Best Practice
===================================

Last updated: 01/27/2026.

.. _Qwen3-30B: https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3moe-30b_sglang_megatron_npu.sh
.. _Qwen2.5-32B: https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen2-32b_sglang_fsdp_npu.sh
引言
----------------------------------

SGLang 是当前主流的高性能开源推理引擎, 昇腾已经全面原生支持该推理引擎在verl中使用,
仅需简单的构建流程，开发者即可完成环境构建，本文将提供两个经典用例来帮助开发者了解以下内容：

1. 环境构建
2. 模型训练与评估
3. 性能采集

两个用例模型脚本以及其需要的硬件条件各自如下：

+----------------------+---------------------+----------+------------------------+
| 模型                 | NPU型号             | 节点数量 | 训推后端               |
+======================+=====================+==========+========================+
| `Qwen3-30B`_         | Atlas 800T A3       | 1        | SGLang + Megatron      |
+----------------------+---------------------+----------+------------------------+
| `Qwen2.5-32B`_       | Atlas 900 A2        | 2        | SGLang + FSDP          |
+----------------------+---------------------+----------+------------------------+

环境构建
-----------------------------------
我们在quickstart中提供了两种构建环境的方法, 1.从镜像文件DockerFile进行构建 2.从自定义Conda环境进行构建

在本实践中, 我们额外指定verl 的commit id 以避免引入其他问题

.. code-block:: bash

    cd verl
    git checkout 772c224
模型训练与评估
-----------------------------------
1.模型数据准备
^^^^^^^^^^^
`Qwen3-30B`_
^^^^^^^^^^^
**下载模型权重**

--local-dir: 模型保存路径

.. code-block:: bash

  export HF_ENDPOINT=https://hf-mirror.com
  hf download --resume-download Qwen/Qwen3-30B-A3B --local-dir /path/to/local_dir

**下载数据集**

.. code-block:: bash

  git clone https://www.modelscope.cn/datasets/AI-ModelScope/DAPO-Math-17k.git

**HuggingFace To Megatron权重转换(可选)**

.. code-block:: bash

  python scripts/converter_hf_to_mcore.py \
      --hf_model_path Qwen/Qwen3-30B-A3B \
      --output_path Qwen/Qwen3-30B-A3B-mcore \
      --use_cpu_initialization    # Only work for MoE models
*注:verl当前已支持mbridge进行灵活的hf和mcore之间的权重转换,可以修改以下相关参数直接加载hf权重*

.. code-block:: bash

    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    actor_rollout_ref.actor.megatron.use_mbridge=True

`Qwen2.5-32B`_
^^^^^^^^^^^
**下载模型权重**

--local-dir: 模型保存路径

.. code-block:: bash

  export HF_ENDPOINT=https://hf-mirror.com
  hf download --resume-download Qwen/Qwen2.5-32B --local-dir /path/to/local_dir

**下载及处理数据集**

.. code-block:: bash

    wget https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/resolve/main/deepscaler.json
    python recipe/r1_ascend/json_to_parquet.py --output_dir ./data/deepscaler --json_path path/to/deepscaler.json --train_data_ratio 0.9

2.训练
^^^^^^^^^^^
根据开发者实际路径配置情况修改模型训练脚本中的以下参数

.. code-block:: bash 

    # Model Weights Paths
    MODEL_PATH=Qwen/Qwen3-30B-A3B
    MCORE_MODEL_PATH=Qwen/Qwen3-30B-A3B-mcore
    RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
    CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

    # File System Paths
    TRAIN_FILE=$RAY_DATA_HOME/dataset/dapo-math-17k.parquet
    TEST_FILE=$RAY_DATA_HOME/dataset/aime-2024.parquet

    #保存频率，-1默认不保存，如需评测请修改此参数
    trainer.save_freq=-1

对于单机任务 `Qwen3-30B`_ , 可以直接bash执行verl仓上示例脚本

.. code-block:: bash 

  bash examples/grpo_trainer/run_qwen3moe-30b_sglang_megatron_npu.sh
对于多节点任务 `Qwen2.5-32B`_ ，我们推荐使用以下脚本进行大规模多节点训练拉起

.. code-block:: bash

  pkill -9 python
  ray stop --force
  rm -rf /tmp/ray
  export RAY_DEDUP_LOGS=0
  export HYDRA_FULL_ERROR=1
  # TASK_QUEUE_ENABLE，下发优化，图模式设置为1，非图模式设置为2
  export TASK_QUEUE_ENABLE=1
  export HCCL_ASYNC_ERROR_HANDLING=0
  export HCCL_EXEC_TIMEOUT=3600
  export HCCL_CONNECT_TIMEOUT=3600
  
  export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
  export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
  export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
  # 修改为当前需要跑的用例路径
  DEFAULT_SH="./run_*.sh"
  echo "Use $DEFAULT_SH"
  
  ulimit -n 32768
  mkdir logs
  
  NNODES=2
  NPUS_PER_NODE=8
  # 修改为对应主节点IP
  MASTER_ADDR="IP FOR MASTER NODE"
  # 修改为当前节点的通信网卡
  SOCKET_IFNAME="Your SOCKET IFNAME"
  export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
  export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
  # 获取当前IP
  CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
  if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
    # 主节点启动
    ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'
  
    while true; do
        ray_status_output=$(ray status)
        npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
        npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
        device_count=$((npu_count_int / $NPUS_PER_NODE))
  
        # 判断device_count 是否与 NNODES 相等
        if [ "$device_count" -eq "$NNODES" ]; then
            echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
            ray status
            bash $DEFAULT_SH
            break
        else
            echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
            sleep 5
        fi
    done
  else
    # 子节点尝试往主节点注册 ray 直到成功
    while true; do
        # 尝试连接 ray 集群
        ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP
  
        # 检查连接是否成功
        ray status
        if [ $? -eq 0 ]; then
            echo "Successfully connected to the Ray cluster!"
            break
        else
            echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
            sleep 5
        fi
    done
  fi
  
  sleep 600

DEFAULT_SH:修改为训练所用配置 sh 文件路径。在此案例中修改为 `Qwen2.5-32B`_ 路径。
          
NNODES 和 NPUS_PER_NODE:修改为使用节点数量和每个节点 NPU 数量。在此案例中分别为2和8。
          
MASTER_ADDR:修改为对应主节点 IP。即所有节点的 MASTER_ADDR 应该相同。
          
SOCKET_IFNAME, HCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME: 修改为对应通信网卡，通信网卡可以通过以下命令获取：
          
.. code-block:: bash
          
  ifconfig |grep "$(hostname -I |awk '{print $1}'|awk -F '.' '{print $0}')" -B 1|awk -F ':' '{print$1}' | head -1 | tail -1

3.模型评估
^^^^^^^^^^^

不同模型步骤一致,仅以Qwen3-30b为例列举

我们通过 AISBenchmark 评估模型,该工具支持vllm/sglang多种推理后端的评估

**安装方法**

.. code-block:: bash

  git clone https://gitee.com/aisbench/benchmark.git
  cd benchmark
  pip install -e .

**下载评估数据集**

.. code-block:: bash

  cd path/to/benchmark/ais_bench/datasets
  wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip
  unzip math.zip
  rm math.zip

**修改AISBench配置代码使能sglang推理评测**

打开 benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py 文件，这是推理配置文件

.. code-block:: bash

    from ais_bench.benchmark.models import VLLMCustomAPIChatStream
    from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content
    from ais_bench.benchmark.clients import OpenAIChatStreamClient, OpenAIChatStreamSglangClient

    models = [
        dict(
            attr="service",
            type=VLLMCustomAPIChatStream,
            abbr='sgl-api-stream-chat',
            path="/path/to/Qwen3-30B", # 修改为 Qwen3-30B 模型路径
            model="qwen3-30b",
            request_rate = 0,
            max_seq_len=2048,
            retry = 2,
            host_ip = "localhost", # 推理服务的IP
            host_port = 8005, # 推理服务的端口
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


**启动sglang_server服务**

.. code-block:: bash

    python -m sglang.launch_server --model-path "/path/to/Qwen3-30B"  --tp-size 4 --dp-size 1 --port 8005 

**启动sglang_client评测**

.. code-block:: bash

    ais_bench --models vllm_api_stream_chat --datasets math500_gen_0_shot_cot_chat_prompt

**评测结果**

经过训练,模型在Math-500上的评分显著上升

+------+----------------------+---------+----------+------+----------------------+
| iter | dataset              | version | metric   | mode | sgl-api-stream-chat  |
+======+======================+=========+==========+======+======================+
|   0  | math_prm800k_500     | c4b6f0  | accuracy | gen  | 	84.4             |
+------+----------------------+---------+----------+------+----------------------+
|  150 | math_prm800k_500     | c4b6f0  | accuracy | gen  |     91.7             |
+------+----------------------+---------+----------+------+----------------------+

性能采集
-----------------------------------
关于NPU profiling的详细文档请参考 `ascend_profiling_zh <https://github.com/volcengine/verl/blob/main/docs/ascend_tutorial/ascend_profiling_zh.rst>`_

在 `Qwen3-30B`_ 的脚本中提供了基本的采集性能选项PROF_CONFIG，默认设置 global_profiler.steps=null 关闭采集， 开发者可根据实际需要进行参数修改

采集完成后，开发者可以使用 `MindStudio Insight <https://www.hiascend.com/document/detail/zh/mindstudio/830/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html>`_ 进行数据解析

注: verl框架侧进行采集全量 Profiling 产生海量且重复的算子记录，可以根据文档修改代码仅采集关键阶段