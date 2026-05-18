NPU 常见问题解答
================

Last updated: 05/13/2026.

本文档总结了在 NPU 上执行 VERL 训练和推理时遇到的常见问题及解决方案。

环境配置问题
------------

### Q1: NPU 设备不可见怎么办？

**问题现象**：torch_npu.npu.is_available() 返回 False

**解决方案**：

.. code-block:: bash

   # 检查设备可见性
   echo $ASCEND_RT_VISIBLE_DEVICES
   
   # 设置可见设备并禁用ray自动设置
   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
   
   # 检查驱动状态
   npu-smi info

调试和诊断
----------

### Q1: 如何启用 NPU 性能分析？

使用 VERL 内置的 profiler：

.. code-block:: shell

   actor_rollout_ref.actor.profiler.tool_config.npu.discrete=true \
   actor_rollout_ref.actor.profiler.tool_config.npu.contents=npu,cpu \
   actor_rollout_ref.actor.profiler.tool_config.npu.level=1 \
   actor_rollout_ref.actor.profiler.tool_config.npu.analysis=true

### Q2: 如何排查 NPU 训练失败的问题？

**排查步骤**：

1. 检查环境变量配置
2. 验证设备可见性
3. 检查 CANN 版本兼容性
4. 查看日志中的具体错误信息
5. 使用最小化示例复现问题

**启用详细日志**：

.. code-block:: bash

   # VERL 框架日志
   export VERL_LOGGING_LEVEL=DEBUG
   
   # 昇腾 NPU 日志（0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR）
   export ASCEND_GLOBAL_LOG_LEVEL=0
   export ASCEND_SLOG_PRINT_TO_STDOUT=1
   
   # HCCL 通信日志
   export HCCL_DEBUG=INFO

常见错误信息
------------

### Q1: "torch_npu detected, but NPU device is not available or visible"

**原因**：NPU 驱动未正确安装或设备不可见

**解决方案**：检查驱动安装状态和 ASCEND_RT_VISIBLE_DEVICES 设置

### Q2: "KeyError: decoder.layers.0.self_attention.q_layernorm.weight"

**原因**：MindSpeed版本过低

**解决方案**：切换MindSpeed至 2.3.0_core_r0.12.1

### Q3: "AssertionError: Weight ... is too large to fit in the bucket"

**问题现象**：在分布式训练权重同步时，出现如下错误：

.. code-block:: text

   AssertionError: Weight model.embed_tokens.weight(torch.Size([151936, 4096]), torch.float32) is too large to fit in the bucket.
   Please increase rollout.update_weights_bucket_megabytes(2048 MB).

**原因**：模型某个权重张量的大小超过了权重传输 bucket 的默认容量（2048 MB）。在 verl 框架中，模型权重通过 bucket（缓冲区）进行分块打包传输。当单个权重张量超过 bucket 大小时，断言检查失败。

**权重大小计算方法**：

权重张量的内存占用（字节）= 各维度大小的乘积 × 每个元素的字节数

其中数据类型对应的字节数为：

- ``torch.float32`` → 4 字节
- ``torch.float16`` / ``torch.bfloat16`` → 2 字节
- ``torch.int8`` → 1 字节

以本例中的 ``model.embed_tokens.weight`` 为例：

.. code-block:: text

   张量形状: torch.Size([151936, 4096])
   数据类型: torch.float32 (4 字节)
   权重大小 = 151936 × 4096 × 4 = 2,483,027,968 字节 ≈ 2369 MB

   默认 bucket 大小 = 2048 MB < 2369 MB → 触发断言失败

**解决方案**：在启动训练时增加 ``update_weights_bucket_megabytes`` 参数，使 bucket 容量大于最大权重张量的内存占用：

.. code-block:: bash

   actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096

**参数值选择建议**：

1. **计算模型中最大权重张量的内存占用**：遍历模型所有参数，找出 ``nbytes`` 最大的那个，将其转换为 MB（除以 1024²）。

2. **向上取整到 2 的幂次**：为便于内存分配和管理，建议将计算结果向上取整到最近的 2 的幂次（如 2048、4096、8192 等）。例如最大权重为 2369 MB，则取 4096 MB。

3. **预留适当余量**：考虑到内存对齐和运行时开销，建议 bucket 大小至少为最大权重大小的 1.2~1.5 倍，再向上取整到 2 的幂次。

4. **注意内存限制**：bucket 大小会直接影响 worker 节点的内存占用，设置过大会导致 OOM。应在满足权重传输需求的前提下，尽量选择较小的值。

**常见模型的推荐值**：

.. list-table::
   :header-rows: 1

   * - 模型规模
     - 典型最大权重形状
     - 推荐 bucket 大小
   * - 7B (Qwen2 等)
     - [151936, 4096] float32
     - 4096 MB
   * - 14B
     - [152064, 5120] float32
     - 4096 MB
   * - 72B
     - [152064, 8192] float32
     - 8192 MB

### Q4: 非共享存储下 checkpoint 加载失败，找不到 common.pt / .metadata / metadata.json

**问题现象**：使用 verl + Megatron 后端在**非共享存储**的多机环境下，保存 checkpoint 正常，但重新加载时报错，提示找不到以下文件：

.. code-block:: text

   FileNotFoundError: common.pt
   FileNotFoundError: .metadata
   FileNotFoundError: metadata.json

**原因**：当前 checkpoint 机制对非共享存储的支持不完善。具体表现为：

- **分布式训练权重是分节点保存的**，每个节点只保存自己负责的分片权重，不会只在主节点保存全部权重。
- 但 ``common.pt``、``.metadata``、``metadata.json`` 等元数据文件**仅保存在执行保存操作的节点上**（通常是 rank 0 所在节点），其他节点本地没有这些文件。
- 加载 checkpoint 时，每个节点都需要读取这些元数据文件来还原模型状态，但非共享存储下其他节点本地路径中不存在这些文件，导致加载失败。

**临时解决方案**：手动将元数据文件从保存节点复制到所有其他节点：

.. code-block:: bash

   # 假设 checkpoint 保存在 rank 0 节点的 /path/to/ckpt/ 目录下
   # 将元数据文件从 rank 0 节点复制到其他所有节点

   # 需要复制的文件
   /path/to/ckpt/common.pt
   /path/to/ckpt/.metadata
   /path/to/ckpt/metadata.json

   # 示例：使用 scp 复制到其他节点
   scp /path/to/ckpt/common.pt node1:/path/to/ckpt/
   scp /path/to/ckpt/.metadata node1:/path/to/ckpt/
   scp /path/to/ckpt/metadata.json node1:/path/to/ckpt/

   # 对所有节点重复上述操作

**注意事项**：

- 每次保存 checkpoint 后都需要重新复制元数据文件，因为保存操作可能会更新这些文件的内容。
- 如果训练过程中频繁保存 checkpoint（如按步数自动保存），建议编写脚本在保存后自动触发复制，避免遗漏。
- 长期方案应等待框架层面支持非共享存储的 checkpoint 加载，使元数据文件能自动同步到所有节点。

参考资料
--------

- `NPU 性能优化指南 <../perf/perf_tuning_on_ascend.rst>`_
- `NPU 快速开始指南 <../start/install.rst>`_
- `NPU CI 指南 <../contribution_guide/ascend_ci_guide_zh.rst>`_
- Ascend NPU 文档: https://www.hiascend.com/document
- CANN 工具包文档: https://www.hiascend.com/software/cann

获取更多帮助
------------

如果以上 FAQ 无法解决您的问题，请：

1. 查看完整的错误日志
2. 在 GitHub Issues 中搜索类似问题
3. 提供详细的错误信息和环境配置
4. 提供最小可复现示例