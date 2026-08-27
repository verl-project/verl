多机任务拉起操作指南
===================================

Last updated: 07/28/2026.

引言
----------------------------------

在大规模模型训练场景中, 单机往往无法满足算力需求, 需要多机协同训练。verl 基于 Ray 框架实现分布式调度,
开发者需在多个节点上正确启动 Ray 集群并配置昇腾 NPU 相关环境变量, 才能顺利拉起多机训练任务。

本文将帮助开发者了解以下内容：

1. 前置准备
2. 多机任务拉起

前置准备
-----------------------------------

1.环境与网络配置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

多机训练前, 请确保所有节点满足以下条件：

- 各节点已按照 `install_guidance <../../get_start/install_guidance.rst>`_ 完成环境构建, 且 verl、Ray、PyTorch、torch-npu、CANN 等关键组件版本一致
- 各节点间训练网段网络互通, 可访问 Ray 端口、Dashboard 端口以及后续配置的 HCCL 端口范围。``ping`` 只能验证基础连通性, 若集群开启防火墙, 还需确认 TCP 端口未被拦截
- 各节点的训练脚本路径及模型/数据/checkpoint 路径保持一致(推荐使用共享文件系统如 NFS)
- 各节点均已完成 NPU 驱动与 CANN 软件栈安装, 且 ``npu-smi info`` 可正常识别设备
- 各节点系统时间尽量保持同步, 避免日志和任务排查时出现时间线错乱

2.获取通信网卡
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

多机通信依赖正确的网卡配置, 在每个节点上先查看可用网卡及其 IPv4 地址：

.. code-block:: bash

  ip -o -4 addr show scope global | awk '{print $2, $4}'

选择用于多机训练通信的网卡, 并记录每个节点对应的网卡名称。若已确定主节点 IP, 也可以在各节点上通过以下命令查看访问主节点时使用的网卡：

.. code-block:: bash

  MASTER_ADDR="IP FOR MASTER NODE"
  ip route get "$MASTER_ADDR" | awk '{for (i = 1; i <= NF; i++) if ($i == "dev") {print $(i + 1); exit}}'

后续配置中的 ``HCCL_SOCKET_IFNAME``、``GLOO_SOCKET_IFNAME`` 和启动脚本里的 ``SOCKET_IFNAME`` 应使用该网卡名称。

3.确认节点角色
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

多机集群中包含一个 **主节点(Master)** 和若干 **子节点(Worker)**:

- **主节点**: 启动 Ray Head 服务, 负责集群调度, 等待所有子节点加入后触发训练任务
- **子节点**: 向主节点注册, 加入 Ray 集群后等待任务调度

请选定其中一个节点作为主节点, 并记录其 IP 地址。

多机任务拉起
-----------------------------------

1.环境变量配置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 **所有节点** 上配置以下环境变量:

.. code-block:: bash

  # Ray 日志去重与错误详细输出
  export RAY_DEDUP_LOGS=0
  export HYDRA_FULL_ERROR=1

  # 昇腾 NPU 下发优化, 图模式设置为1, 非图模式设置为2
  export TASK_QUEUE_ENABLE=1

  # HCCL 通信超时配置(单位:秒), 根据模型规模适当调大
  export HCCL_ASYNC_ERROR_HANDLING=0
  export HCCL_EXEC_TIMEOUT=3600
  export HCCL_CONNECT_TIMEOUT=3600

  # HCCL 端口范围配置, 避免端口冲突
  export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
  export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

  # NPU 可见设备配置
  export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

  # 通信网卡配置, 替换为当前节点实际的网卡名称
  export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
  export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"

  # 文件描述符限制
  ulimit -n 32768

  # 可选配置
  # 关闭 Hugging Face 异步权重加载, 避免部分环境下模型加载阶段主机内存峰值过高
  export HF_DEACTIVATE_ASYNC_LOAD=1

2.编写多机启动脚本
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

以下脚本可在所有节点上统一执行, 脚本会根据当前节点 IP 自动判断主/子节点角色:

.. code-block:: bash

  # 清理上次训练可能残留的 Ray 进程
  pkill -9 python
  ray stop --force
  rm -rf /tmp/ray

  # ====== 用户需修改的配置 ======
  # 训练脚本路径
  DEFAULT_SH="./run_*.sh"
  echo "Use $DEFAULT_SH"

  # 节点数量与单节点 NPU 数量
  NNODES=2
  NPUS_PER_NODE=16

  # 主节点 IP
  MASTER_ADDR="IP FOR MASTER NODE"

  # 当前节点的通信网卡
  SOCKET_IFNAME="Your SOCKET IFNAME"
  # ====== 配置结束 ======

  # 获取当前节点 IP
  CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

  if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
    # ====== 主节点 ======
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
    # ====== 子节点 ======
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

**脚本配置参数说明：**

.. list-table::
   :header-rows: 1

   * - 参数
     - 说明
   * - ``DEFAULT_SH``
     - 训练所用配置 sh 文件路径, 如 ``run_qwen3moe-30b_grpo_megatron_vllm_npu.sh``
   * - ``NNODES``
     - 参与训练的节点数量
   * - ``NPUS_PER_NODE``
     - 每个节点的 NPU 数量, 如 Atlas 800T A3 通常为16
   * - ``MASTER_ADDR``
     - 主节点 IP, 所有节点的该参数必须相同
   * - ``SOCKET_IFNAME``
     - 当前节点的通信网卡名称, 各节点可能不同

3.启动训练
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

将上述脚本保存为 ``ray_start.sh``, 在 **所有节点** 上分别执行:

.. code-block:: bash

  bash ray_start.sh

执行顺序建议：

1. 先在 **主节点** 上启动脚本, 等待 Ray Head 服务就绪
2. 再在各个 **子节点** 上启动脚本, 子节点会自动向主节点注册
3. 主节点检测到所有节点加入后, 自动触发训练任务

4.监控训练状态
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

训练启动后, 可通过以下方式监控：

**Ray Dashboard**

浏览器访问 ``http://<MASTER_ADDR>:8260``, 查看 Ray 集群状态、资源使用和任务运行情况。

**命令行查看**

.. code-block:: bash

  ray status

**训练日志**

训练日志输出位置取决于 ``DEFAULT_SH`` 指向的训练脚本。如训练脚本中配置了日志文件, 可使用以下命令实时查看:

.. code-block:: bash

  tail -f <TRAINING_LOG_PATH>
