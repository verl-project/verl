Ascend SGLang Best Practices
===================================

Last updated: 06/02/2026.

.. _Qwen3-30B: https://github.com/verl-project/verl/blob/main/examples/ascend_extras/grpo_trainer/run_qwen3_30b_a3b_megatron.sh
.. _doclink: https://github.com/verl-project/verl/blob/c98cb8cc/docs/ascend_tutorial/examples/ascend_sglang_best_practices.rst
Introduction
----------------------------------

SGLang is currently a mainstream high-performance open-source inference engine. Ascend has fully and natively supported this inference engine for use in verl.
With only a simple build process, developers can complete the environment setup. This document will help developers understand the following:

1. Environment Setup
2. Model Training and Evaluation
3. Performance Collection

The example model script and its required hardware conditions are as follows:

- Note: verl recently performed script cleanup and naming changes. It is recommended to build based on the documentation and scripts corresponding to commit id c98cb8cc `doclink`_ to avoid broken links.

+----------------------+---------------------+--------+------------------------------+
| Model                | NPU Model           | Nodes  | Training/Inference Backend   |
+======================+=====================+========+==============================+
| `Qwen3-30B`_         | Atlas 800T A3       | 1      | SGLang + Megatron            |
+----------------------+---------------------+--------+------------------------------+

Environment Setup
-----------------------------------
We provide two methods for building the environment in the `Ascend Installation Guide <../../get_start/install_guidance.rst>`_: 1. Build from a Dockerfile image file 2. Build from a custom Conda environment.

In this practice, we additionally specify the verl commit id to avoid introducing other issues.

.. code-block:: bash

    cd verl
    git checkout 772c224
Model Training and Evaluation
-----------------------------------
1. Model Data Preparation
^^^^^^^^^^^^^^^^^^^^^^^^^
`Qwen3-30B`_
^^^^^^^^^^^^^
**Download Model Weights**

--local-dir: The path where the model is saved.

.. code-block:: bash

  export HF_ENDPOINT=https://hf-mirror.com
  huggingface-cli download --resume-download Qwen/Qwen3-30B-A3B --local-dir /path/to/local_dir

**Download Dataset**

.. code-block:: bash

  git clone https://www.modelscope.cn/datasets/AI-ModelScope/DAPO-Math-17k.git

**HuggingFace To Megatron Weight Conversion (Optional)**

.. code-block:: bash

  python scripts/converter_hf_to_mcore.py \
      --hf_model_path Qwen/Qwen3-30B-A3B \
      --output_path Qwen/Qwen3-30B-A3B-mcore \
      --use_cpu_initialization    # Only work for MoE models
*Note: verl performs weight conversion between hf and mcore through Megatron-Bridge and MindSpeed-Bridge. You can modify the following related parameters to load hf weights directly.*

.. code-block:: bash

    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    actor_rollout_ref.actor.megatron.use_mbridge=True

2. Training
^^^^^^^^^^^
Modify the following parameters in the model training script according to your actual path configuration.

.. code-block:: bash 

    # Model Weights Paths
    MODEL_PATH=Qwen/Qwen3-30B-A3B
    MCORE_MODEL_PATH=Qwen/Qwen3-30B-A3B-mcore
    RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
    CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

    # File System Paths
    TRAIN_FILE=$RAY_DATA_HOME/dataset/dapo-math-17k.parquet
    TEST_FILE=$RAY_DATA_HOME/dataset/aime-2024.parquet

    #Save frequency, -1 means no save by default; modify this parameter if evaluation is needed
    trainer.save_freq=-1

For single-node tasks `Qwen3-30B`_, you can directly run the example script from the verl repository with bash.

.. code-block:: bash 

  bash examples/grpo_trainer/run_qwen3moe-30b_sglang_megatron_npu.sh
If you want to scale to multiple nodes, we recommend using the following script to launch large-scale multi-node training.

.. code-block:: bash

  pkill -9 python
  ray stop --force
  rm -rf /tmp/ray
  export RAY_DEDUP_LOGS=0
  export HYDRA_FULL_ERROR=1
  # TASK_QUEUE_ENABLE: dispatch optimization, set to 1 for graph mode, set to 2 for non-graph mode
  export TASK_QUEUE_ENABLE=1
  export HCCL_ASYNC_ERROR_HANDLING=0
  export HCCL_EXEC_TIMEOUT=3600
  export HCCL_CONNECT_TIMEOUT=3600
  
  export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
  export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
  export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  # Change to the path of the use case you want to run
  DEFAULT_SH="./run_*.sh"
  echo "Use $DEFAULT_SH"
  
  ulimit -n 32768
  mkdir logs
  
  NNODES=2
  NPUS_PER_NODE=8
  # Change to the corresponding master node IP
  MASTER_ADDR="IP FOR MASTER NODE"
  # Change to the current node's communication network interface
  SOCKET_IFNAME="Your SOCKET IFNAME"
  export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
  export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
  # Get the current IP
  CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
  if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
    # Master node startup
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
    # Worker node tries to register with the master node until success
    while true; do
        # Try to connect to the ray cluster
        ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP
  
        # Check whether the connection succeeded
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

DEFAULT_SH: Change to the path of the configuration sh file used for training.
          
NNODES and NPUS_PER_NODE: Change to the number of nodes and the number of NPUs per node. In this case, they are 2 and 8 respectively.
          
MASTER_ADDR: Change to the corresponding master node IP. That is, the MASTER_ADDR of all nodes should be the same.
          
SOCKET_IFNAME, HCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME: Change to the corresponding communication network interface. The communication network interface can be obtained with the following command:
          
.. code-block:: bash
          
  ifconfig |grep "$(hostname -I |awk '{print $1}'|awk -F '.' '{print $0}')" -B 1|awk -F ':' '{print$1}' | head -1 | tail -1

3. Model Evaluation
^^^^^^^^^^^^^^^^^^^

The steps are the same for different models; here we use Qwen3-30b as an example.

We evaluate the model using AISBenchmark, which supports evaluation with both vllm and sglang inference backends.

**Installation**

.. code-block:: bash

  git clone https://gitee.com/aisbench/benchmark.git
  cd benchmark
  pip install -e .
  pip install math_verify latex2sympy2_extended

**Download Evaluation Dataset**

.. code-block:: bash

  cd path/to/benchmark/ais_bench/datasets
  wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip
  unzip math.zip
  rm math.zip

**Modify AISBench Configuration to Enable SGLang Inference Evaluation**

Open the benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py file, which is the inference configuration file.

.. code-block:: bash

    from ais_bench.benchmark.models import VLLMCustomAPIChatStream
    from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content
    from ais_bench.benchmark.clients import OpenAIChatStreamClient, OpenAIChatStreamSglangClient

    models = [
        dict(
            attr="service",
            type=VLLMCustomAPIChatStream,
            abbr='sgl-api-stream-chat',
            path="/path/to/Qwen3-30B", # Change to the Qwen3-30B model path
            model="qwen3-30b",
            request_rate = 0,
            max_seq_len=2048,
            retry = 2,
            host_ip = "localhost", # IP of the inference service
            host_port = 8005, # Port of the inference service
            max_out_len = 8192,  # Maximum output tokens length
            batch_size=48, # Maximum inference concurrency
            trust_remote_code=False,
            custom_client=dict(type=OpenAIChatStreamSglangClient), # Use sglang client
            generation_kwargs = dict(
                temperature = 0,
                seed = 1234,
            ),
            pred_postprocessor=dict(type=extract_non_reasoning_content)
        )
    ]


**Start sglang_server Service**

.. code-block:: bash

    python -m sglang.launch_server --model-path "/path/to/Qwen3-30B"  --tp-size 4 --dp-size 1 --port 8005 

**Start sglang_client Evaluation**

.. code-block:: bash

    ais_bench --models vllm_api_stream_chat --datasets math500_gen_0_shot_cot_chat_prompt

**Evaluation Results**

After training, the model's score on Math-500 improved significantly.

+------+----------------------+---------+----------+------+----------------------+
| iter | dataset              | version | metric   | mode | sgl-api-stream-chat  |
+======+======================+=========+==========+======+======================+
|   0  | math_prm800k_500     | c4b6f0  | accuracy | gen  | 	84.4             |
+------+----------------------+---------+----------+------+----------------------+
|  150 | math_prm800k_500     | c4b6f0  | accuracy | gen  |     91.7             |
+------+----------------------+---------+----------+------+----------------------+

Performance Collection
-----------------------------------
For detailed documentation on NPU profiling, refer to the `Profiling Collection Guide <../../dev_guide/performance/ascend_profiling.rst>`_.

The script for `Qwen3-30B`_ provides a basic performance collection option PROF_CONFIG. By default, global_profiler.steps=null disables collection. Developers can modify the parameters according to their actual needs.

After collection is complete, developers can use `MindStudio Insight <https://www.hiascend.com/document/detail/en/mindstudio/830/GUI_baseddevelopmenttool/MindStudioInsight/Insight_userguide_0002.html?framework=mindspore>`_ for data parsing.

Note: Collecting full Profiling on the verl framework side generates massive and repetitive operator records. You can modify the code according to the documentation to collect only key stages.
