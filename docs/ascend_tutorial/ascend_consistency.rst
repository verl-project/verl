Align the Inference results of the verl and vLLM frameworks on Ascend devices(zh)
====================================

在昇腾设备上对齐verl和vLLM两个框架下的推理结果。

Last updated: 11/17/2025.

这是一份在昇腾设备上对齐verl和vLLM两个框架下推理结果的教程。

环境变量配置
~~~~~~~~~~~~

在多卡通信情况下：

- HCCL通信下(默认场景):
  
  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export HCCL_DETERMINISTIC="true"
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

- LCCL通信下(通过export HCCL_OP_EXPANSION_MODE="AIV"使能）:

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export LCCL_DETERMINISTIC=1
  -  export ATB_LLM_LCOC_ENABLE=0
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

在单卡无通信情况下：

- HCCL和LCCL通信下:
 
  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

vLLM初始化参数
~~~~~~~~~~~~

需要对 SamplingParams 参数里单独设置seed, 保持vLLM和verl推理结果一致, 举例修改如下：

.. code:: yaml

      sampling_params = SamplingParams(n=1,
                                       logprobs=0,  # can be set to 0 and let actor to recompute
                                       max_tokens=config.response_length,
                                       repetition_penalty=config.get("repetition_penalty", 1.0),
                                       seed=1234)

