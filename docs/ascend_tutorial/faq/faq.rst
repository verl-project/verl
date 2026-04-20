NPU 常见问题解答
================

Last updated: 03/26/2026.

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