Ascend vLLM Best Practices
===================================

Last updated: 06/06/2026.

.. _Qwen3-30B: https://github.com/verl-project/verl/blob/release/v0.7.1/examples/grpo_trainer/run_qwen3moe-30b_grpo_megatron_vllm_npu.sh
.. _doclink: https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/zh/model_support/examples/ascend_vllm_best_practices.rst
Introduction
----------------------------------

vLLM is a mainstream high-performance open-source inference engine. Ascend now fully and natively supports this inference engine for use in verl. With a simple build process, you can complete the environment setup. This guide provides two classic examples to help you understand the following:

1. Environment setup
2. Model training and evaluation 
3. Performance collection

The use case model scripts and their required hardware conditions are as follows:

- Note: verl recently performed script cleanup and naming changes. To avoid link failures, build using the documentation `doclink`_ and the corresponding scripts for commit ID c98cb8cc.

+----------------------+---------------------+----------+----------------------------+
| Model                | NPU Model           | Nodes    | Training/Inference Backend |
+======================+=====================+==========+============================+
| `Qwen3-30B`_         | Atlas 800T A3       | 1        | vLLM + Megatron            |
+----------------------+---------------------+----------+----------------------------+


Environment Setup
-----------------------------------
We provide two methods for setting up the environment in the `Ascend Installation Guide <../../get_start/install_guidance.rst>`_: 1. Build from a Dockerfile 2. Build from a custom Conda environment

In this practice, additionally specify the verl commit ID to avoid introducing other issues.

.. code-block:: bash

    cd verl
    git checkout release/v0.7.1
Model Training and Evaluation
-----------------------------------
1. Model Data Preparation
^^^^^^^^^^^
`Qwen3-30B`_
^^^^^^^^^^^^
**Download Model Weights**

--local-dir: Path to save the model

.. code-block:: bash

  export HF_ENDPOINT=https://hf-mirror.com
  huggingface-cli download --resume-download Qwen/Qwen3-30B-A3B-Base --local-dir /path/to/local_dir

**Download the dataset**

.. code-block:: bash

  git clone https://www.modelscope.cn/datasets/modelscope/gsm8k.git

**HuggingFace To Megatron Weight Conversion (Optional)**

.. code-block:: bash

  python scripts/converter_hf_to_mcore.py \
      --hf_model_path Qwen/Qwen3-30B-A3B-Base \
      --output_path Qwen/Qwen3-30B-A3B-Base-mcore \
      --use_cpu_initialization    # Only work for MoE models
*Note: verl currently supports using mbridge for flexible weight conversion between HF and mcore. You can modify the following related parameters to directly load HF weights.*

.. code-block:: bash

    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    actor_rollout_ref.actor.megatron.use_mbridge=True

2. Training
^^^^^^^^^^^
Modify the following parameters in the model training script based on your actual path configuration.

.. code-block:: bash 

    # Model Weights Paths
    MODEL_PATH=Qwen/Qwen3-30B-A3B-Base
    MCORE_MODEL_PATH=Qwen/Qwen3-30B-A3B-Base-mcore
    RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
    CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

    # File System Paths
    TRAIN_FILE=$RAY_DATA_HOME/dataset/gsm8k/test.parquet
    TEST_FILE=$RAY_DATA_HOME/dataset/gsm8k/test.parquet

    #Save frequency. -1 means no saving by default. Modify this parameter if you need to evaluate.
    trainer.save_freq=-1

For single-machine tasks `Qwen3-30B`_, you can directly run the sample script from the verl repository using bash, for example:

.. code-block:: bash 

  bash examples/grpo_trainer/run_qwen3moe-30b_grpo_megatron_vllm_npu.sh
If you want to scale to multiple nodes, we recommend using the following script to launch large-scale multi-node training.

.. code-block:: bash

  pkill -9 python
  ray stop --force
  rm -rf /tmp/ray
  export RAY_DEDUP_LOGS=0
  export HYDRA_FULL_ERROR=1
  # TASK_QUEUE_ENABLE, dispatch optimization, set to 1 for graph mode, set to 2 for non-graph mode
  export TASK_QUEUE_ENABLE=1
  export HCCL_ASYNC_ERROR_HANDLING=0
  export HCCL_EXEC_TIMEOUT=3600
  export HCCL_CONNECT_TIMEOUT=3600

  export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
  export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
  export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  # Change to the path of the test case you need to run
  DEFAULT_SH="./run_*.sh"
  echo "Use $DEFAULT_SH"

  ulimit -n 32768
  mkdir logs

  NNODES=2
  NPUS_PER_NODE=8
  # Change to the master node IP address
  MASTER_ADDR="IP FOR MASTER NODE"
  # Change to the communication network interface of the current node
  SOCKET_IFNAME="Your SOCKET IFNAME"
  export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
  export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
  # Get the current IP address
  CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
  if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
    # Start the master node
    ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

    while true; do
        ray_status_output=$(ray status)
        npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
        npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
        device_count=$((npu_count_int / $NPUS_PER_NODE))

        # Check whether device_count equals NNODES
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
    # The worker node attempts to register with Ray on the master node until successful
    while true; do
        # Attempt to connect to the Ray cluster
        ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

        # Check whether the connection is successful
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

DEFAULT_SH: Change this to the path of the .sh configuration file used for training.

NNODES and NPUS_PER_NODE: Modify them to use the number of nodes and the number of NPUs per node. In this case, the values are 2 and 8, respectively.

MASTER_ADDR: Change to the corresponding master node IP. That is, the MASTER_ADDR of all nodes should be the same.

SOCKET_IFNAME, HCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME: Set to the corresponding communication NIC. You can obtain the communication NIC using the following command:

.. code-block:: bash

  ifconfig |grep "$(hostname -I |awk '{print $1}'|awk -F '.' '{print $0}')" -B 1|awk -F ':' '{print$1}' | head -1 | tail -1

3. Model Evaluation
^^^^^^^^^^^^^^^^^^^

The steps are the same for different models. This section lists only Qwen3-30b as an example.

You can use AISBenchmark to evaluate the model. This tool supports evaluating multiple inference backends, such as vLLM and SGLang.

**Installation method**

.. code-block:: bash

  git clone https://gitee.com/aisbench/benchmark.git
  cd benchmark
  pip install -e .
  pip install math_verify latex2sympy2_extended

**Download the evaluation dataset**

.. code-block:: bash

  cd /examples/benchmark/ais_bench/datasets
  mkdir aime/
  cd aime/
  wget https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip
  unzip aime.zip
  rm aime.zip

**Modify the AISBench configuration code to enable vllm inference evaluation**

.. code-block:: bash

   vim /examples/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py

The Python file content is as follows. The host_port must match the server port. Modify max_seq_len and max_out_len according to the model configuration. The inference example is set to 2k input and 20k output:

.. code-block:: bash

  from ais_bench.benchmark.models import VLLMCustomAPI

  models = [
      dict(
          attr="service",
          type=VLLMCustomAPI,
          abbr='vllm-api-general',
          path="/path/to/Qwen3-30B", # Change to the Qwen3-30B model path
          model="qwen3-30b",
          request_rate = 0,
          retry = 2,
          host_ip = "localhost", # IP address of the inference service
          host_port = 6380,
          max_seq_len = 2048, # Maximum input token length
          max_out_len = 20480, # Maximum output token length
          batch_size=48, # Maximum concurrency for inference
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


**Start the vllm_server service**

Use the following command to start the NPU server. The parameters to modify are model and tensor-parallel-size.

/path/to/Qwen3-30B/: The Hugging Face model path that stores the post-training weights;
tensor-parallel-size: The number of tensor parallelism replicas. It is recommended that the TP configuration remains consistent with the infer configuration during training;
data-parallel-size: The number of data parallelism replicas. It is recommended that the DP configuration remains consistent with the infer configuration during training. The default value is 1;
port: You can set any idle port;

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


**Start the vllm_client evaluation**

.. code-block:: bash

  cd /examples/benchmark
  ais_bench --models vllm_api_general --datasets aime2024_gen


**Evaluation results**

After training, the model's score on aime2024 has significantly improved.

+------+----------+---------+----------+------+-----------------------+
| iter | dataset  | version | metric   | mode | vllm-api-stream-chat  |
+======+==========+=========+==========+======+=======================+
|   0  | aime2024 | a4b6f0  | accuracy | gen  | 85.4                  |
+------+----------+---------+----------+------+-----------------------+
|  150 | aime2024 | a4b6f0  | accuracy | gen  | 91.2                  |
+------+----------+---------+----------+------+-----------------------+

Performance Profiling
-----------------------------------
For detailed documentation on NPU profiling, refer to the `Profiling Guide <../../dev_guide/performance/ascend_profiling.rst>`_.

After data collection is complete, you can use `MindStudio Insight <https://www.hiascend.com/document/detail/en/mindstudio/830/GUI_baseddevelopmenttool/MindStudioInsight/Insight_userguide_0002.html>`_ to parse the data.

Note: Collecting full Profiling on the verl framework generates massive and duplicate operator records. You can modify the code according to the documentation to collect only key stages.
