Ascend vLLM Best Practice
===================================

Last updated: 06/06/2026.

.. _Qwen3-30B: https://github.com/verl-project/verl/blob/release/v0.7.1/examples/grpo_trainer/run_qwen3moe-30b_grpo_megatron_vllm_npu.sh
.. _doclink: https://github.com/verl-project/verl/blob/c98cb8cc/docs/ascend_tutorial/examples/ascend_vllm_best_pratice.rst
引言
----------------------------------

vLLM 是当前主流的高性能开源推理引擎, 昇腾已经全面原生支持该推理引擎在verl中使用,
仅需简单的构建流程，开发者即可完成环境构建，本文将提供两个经典用例来帮助开发者了解以下内容：

1. 环境构建
2. 模型训练与评估 
3. 性能采集

用例模型脚本以及其需要的硬件条件如下：

- 注:verl近期进行了脚本清理与命名变更, 推荐根据 commit id c98cb8cc 对应文档 `doclink`_  及对应脚本进行构建避免链接失效

+----------------------+---------------------+----------+----------------------------+
| 模型                 | NPU型号             | 节点数量 | 训推后端                   |
+======================+=====================+==========+============================+
| `Qwen3-30B`_         | Atlas 800T A3       | 1        | vLLM + Megatron            |
+----------------------+---------------------+----------+----------------------------+


环境构建
-----------------------------------
我们在 `install_guidance <../../get_start/install_guidance.rst>`_ 中提供了两种构建环境的方法, 1.从镜像文件DockerFile进行构建 2.从自定义Conda环境进行构建

在本实践中, 我们额外指定verl 的commit id 以避免引入其他问题

.. code-block:: bash

    cd verl
    git checkout release/v0.7.1
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
  huggingface-cli download --resume-download Qwen/Qwen3-30B-A3B-Base --local-dir /path/to/local_dir

**下载数据集**

.. code-block:: bash

  git clone https://www.modelscope.cn/datasets/modelscope/gsm8k.git

**HuggingFace To Megatron权重转换(可选)**

.. code-block:: bash

  python scripts/converter_hf_to_mcore.py \
      --hf_model_path Qwen/Qwen3-30B-A3B-Base \
      --output_path Qwen/Qwen3-30B-A3B-Base-mcore \
      --use_cpu_initialization    # Only work for MoE models
*注:verl当前已支持mbridge进行灵活的hf和mcore之间的权重转换,可以修改以下相关参数直接加载hf权重*

.. code-block:: bash

    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    actor_rollout_ref.actor.megatron.use_mbridge=True

2.训练
^^^^^^^^^^^
根据开发者实际路径配置情况修改模型训练脚本中的以下参数

.. code-block:: bash 

    # Model Weights Paths
    MODEL_PATH=Qwen/Qwen3-30B-A3B-Base
    MCORE_MODEL_PATH=Qwen/Qwen3-30B-A3B-Base-mcore
    RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
    CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

    # File System Paths
    TRAIN_FILE=$RAY_DATA_HOME/dataset/gsm8k/test.parquet
    TEST_FILE=$RAY_DATA_HOME/dataset/gsm8k/test.parquet

    #保存频率，-1默认不保存，如需评测请修改此参数
    trainer.save_freq=-1

对于单机任务 `Qwen3-30B`_ , 可以直接bash执行verl仓上示例脚本，如:

.. code-block:: bash 

  bash examples/grpo_trainer/run_qwen3moe-30b_grpo_megatron_vllm_npu.sh
如果您想扩展至多节点 ，我们推荐使用以下脚本进行大规模多节点训练拉起

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
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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

DEFAULT_SH:修改为训练所用配置 sh 文件路径。
          
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
  pip install math_verify latex2sympy2_extended

**下载评估数据集**

.. code-block:: bash

  cd /examples/benchmark/ais_bench/datasets
  mkdir aime/
  cd aime/
  wget https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip
  unzip aime.zip
  rm aime.zip

**修改AISBench配置代码使能vllm推理评测**

.. code-block:: bash

   vim /examples/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py

python文件内容如下，host_port需与服务端的port一致，根据模型配置修改max_seq_len和max_out_len，推理示例设置为2k推20k：

.. code-block:: bash

  from ais_bench.benchmark.models import VLLMCustomAPI

  models = [
      dict(
          attr="service",
          type=VLLMCustomAPI,
          abbr='vllm-api-general',
          path="/path/to/Qwen3-30B", # 修改为 Qwen3-30B 模型路径
          model="qwen3-30b",
          request_rate = 0,
          retry = 2,
          host_ip = "localhost", # 推理服务的IP
          host_port = 6380,
          max_seq_len = 2048, # 最大输入tokens长度
          max_out_len = 20480, # 最大输出tokens长度
          batch_size=48, # 推理的最大并发数
          trust_remote_code=False,
          generation_kwargs = dict(
              temperature = 0.5,
              top_k = 10,
              top_p = 0.95,
              seed = None,
              repetition_penalty = 1.03,
          )
      )
  ]


**启动vllm_server服务**

通过以下命令拉起NPU服务端，需要修改的参数：model和tensor-parallel-size。

/path/to/Qwen3-30B/：保存训练后权重的huggingface模型地址；
tensor-parallel-size：张量并行副本数，TP建议和训练时infer的配置保持一致；
data-parallel-size：数据并行副本数，DP建议和训练时infer的配置保持一致，默认为1；
port：可任意设置空闲端口；

.. code-block:: bash

  cd /path/to/vllm
  vllm serve /path/to/Qwen3-30B/ \
      --served-model-name auto \
      --gpu-memory-utilization 0.9 \
      --max-num-seqs 24 \
      --max-model-len 10240 \
      --max-num-batched-tokens 10240 \
      --enforce-eager \
      --trust-remote-code \
      --distributed_executor_backend=mp \
      --tensor-parallel-size 4 \
      --data-parallel-size 1 \
      --generation-config vllm \
      --port 6380


**启动vllm_client评测**

.. code-block:: bash

  cd /examples/benchmark
  ais_bench --models vllm_api_general --datasets aime2024_gen


**评测结果**

经过训练,模型在aime2024上的评分显著上升

+------+----------+---------+----------+------+-----------------------+
| iter | dataset  | version | metric   | mode | vllm-api-stream-chat  |
+======+==========+=========+==========+======+=======================+
|   0  | aime2024 | a4b6f0  | accuracy | gen  | 85.4                  |
+------+----------+---------+----------+------+-----------------------+
|  150 | aime2024 | a4b6f0  | accuracy | gen  | 91.2                  |
+------+----------+---------+----------+------+-----------------------+

性能采集
-----------------------------------
关于NPU profiling的详细文档请参考 `ascend_profiling_zh <../../dev_guide/performance/ascend_profiling_zh.rst>`_

采集完成后，开发者可以使用 `MindStudio Insight <https://www.hiascend.com/document/detail/zh/mindstudio/830/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html>`_ 进行数据解析

注: verl框架侧进行采集全量 Profiling 产生海量且重复的算子记录，可以根据文档修改代码仅采集关键阶段
