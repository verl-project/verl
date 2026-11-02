Multi-Node Task Startup Guide
===================================

Last updated: 07/28/2026.

Introduction
----------------------------------

In large-scale model training scenarios, a single machine often cannot meet the computing power requirements, and multi-node collaborative training is needed. verl implements distributed scheduling based on the Ray framework. Developers must correctly start the Ray cluster on multiple nodes and configure Ascend NPU-related environment variables to successfully launch multi-node training tasks.

This document will help developers understand the following:

1. Prerequisites
2. Multi-Node Task Startup

Prerequisites
-----------------------------------

1. Environment and Network Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before multi-node training, ensure that all nodes meet the following conditions:

- All nodes have completed environment setup according to the `Ascend Installation Guide <../../get_start/install_guidance.rst>`_, and key components such as verl, Ray, PyTorch, torch-npu, and CANN have consistent versions across nodes
- The training network segment is interconnected between all nodes, with access to Ray ports, Dashboard ports, and the HCCL port range configured later. ``ping`` only verifies basic connectivity; if the cluster has a firewall enabled, you must also confirm that TCP ports are not blocked
- The training script paths and model/data/checkpoint paths are consistent across all nodes (a shared file system such as NFS is recommended)
- All nodes have completed NPU driver and CANN software stack installation, and ``npu-smi info`` can properly identify the devices
- The system time of all nodes should be kept as synchronized as possible to avoid timeline confusion during log analysis and task troubleshooting

2. Obtain the Communication Network Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multi-node communication depends on correct network interface configuration. First, check the available network interfaces and their IPv4 addresses on each node:

.. code-block:: bash

  ip -o -4 addr show scope global | awk '{print $2, $4}'

Select the network interface for multi-node training communication and record the interface name for each node. If the master node IP is already determined, you can also check the network interface used to access the master node with the following command on each node:

.. code-block:: bash

  MASTER_ADDR="IP FOR MASTER NODE"
  ip route get "$MASTER_ADDR" | awk '{for (i = 1; i <= NF; i++) if ($i == "dev") {print $(i + 1); exit}}'

The ``HCCL_SOCKET_IFNAME``, ``GLOO_SOCKET_IFNAME`` in subsequent configurations, and ``SOCKET_IFNAME`` in the startup script should use this network interface name.

3. Confirm Node Roles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A multi-node cluster contains one **Master node** and several **Worker nodes**:

- **Master node**: Starts the Ray Head service, handles cluster scheduling, and triggers the training task after all worker nodes have joined
- **Worker nodes**: Register with the master node and wait for task scheduling after joining the Ray cluster

Select one node as the master node and record its IP address.

Multi-Node Task Startup
-----------------------------------

1. Environment Variable Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure the following environment variables on **all nodes**:

.. code-block:: bash

  # Ray log deduplication and detailed error output
  export RAY_DEDUP_LOGS=0
  export HYDRA_FULL_ERROR=1

  # Ascend NPU dispatch optimization, set to 1 for graph mode, 2 for non-graph mode
  export TASK_QUEUE_ENABLE=1

  # HCCL communication timeout configuration (unit: seconds), increase appropriately based on model size
  export HCCL_ASYNC_ERROR_HANDLING=0
  export HCCL_EXEC_TIMEOUT=3600
  export HCCL_CONNECT_TIMEOUT=3600

  # HCCL port range configuration, avoid port conflicts
  export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
  export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

  # NPU visible device configuration
  export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

  # Communication network interface configuration, replace with the actual interface name of the current node
  export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
  export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"

  # File descriptor limit
  ulimit -n 32768

  # Optional configuration
  # Disable Hugging Face async weight loading to avoid excessive host memory peaks during model loading in some environments
  export HF_DEACTIVATE_ASYNC_LOAD=1

2. Write the Multi-Node Startup Script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following script can be executed uniformly on all nodes. It automatically determines the master/worker role based on the current node IP:

.. code-block:: bash

  # Clean up Ray processes that may remain from the last training run
  pkill -9 python
  ray stop --force
  rm -rf /tmp/ray

  # ====== Configuration to modify ======
  # Training script path
  DEFAULT_SH="./run_*.sh"
  echo "Use $DEFAULT_SH"

  # Number of nodes and NPUs per node
  NNODES=2
  NPUS_PER_NODE=16

  # Master node IP
  MASTER_ADDR="IP FOR MASTER NODE"

  # Current node's communication network interface
  SOCKET_IFNAME="Your SOCKET IFNAME"
  # ====== End of configuration ======

  # Get current node IP
  CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

  if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
    # ====== Master node ======
    ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

    while true; do
        ray_status_output=$(ray status)
        npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
        npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
        device_count=$((npu_count_int / $NPUS_PER_NODE))

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
    # ====== Worker node ======
    while true; do
        ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

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

**Script Configuration Parameter Description:**

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
   * - ``DEFAULT_SH``
     - The configuration sh file path used for training, for example ``run_qwen3moe-30b_grpo_megatron_vllm_npu.sh``
   * - ``NNODES``
     - Number of nodes participating in training
   * - ``NPUS_PER_NODE``
     - Number of NPUs per node, for example Atlas 800T A3 typically has 16
   * - ``MASTER_ADDR``
     - Master node IP, this parameter must be the same on all nodes
   * - ``SOCKET_IFNAME``
     - Communication network interface name of the current node, may differ across nodes

3. Start Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the above script as ``ray_start.sh`` and execute it on **all nodes**:

.. code-block:: bash

  bash ray_start.sh

Recommended execution order:

1. First start the script on the **master node** and wait for the Ray Head service to be ready
2. Then start the script on each **worker node**; worker nodes will automatically register with the master node
3. After the master node detects that all nodes have joined, it automatically triggers the training task

4. Monitor Training Status
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After training starts, you can monitor it through the following methods:

**Ray Dashboard**

Access ``http://<MASTER_ADDR>:8260`` in a browser to view the Ray cluster status, resource usage, and task execution.

**Command Line**

.. code-block:: bash

  ray status

**Training Logs**

The training log output location depends on the training script pointed to by ``DEFAULT_SH``. If the training script configures a log file, you can view it in real time with the following command:

.. code-block:: bash

  tail -f <TRAINING_LOG_PATH>
