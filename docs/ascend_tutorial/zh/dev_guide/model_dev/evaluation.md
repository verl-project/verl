# 模型评测

Last updated: 07/14/2026.

不同模型步骤一致，仅以Qwen3-30B为例列举

我们通过 AISBench 评估模型，该工具支持vllm/sglang多种推理后端的评估

## 1.安装方法

~~~bash
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark
pip install -e .
~~~


## 2.下载评估数据集

~~~bash
cd path/to/benchmark/ais_bench/datasets
wget https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip
unzip math.zip
rm math.zip
~~~

## 3.权重转换

当前verl已经支持mbridge直接保存hf格式模型权重，无需转换即可使用。

如果模型权重不是hf格式，需要先转换为hf格式，再进行评估。

此处参照verl原生[转换方法](../../../../../docs/advance/checkpoint.rst)

## 4.vllm推理评测

**启动vllm_server服务**

通过以下命令拉起推理服务端，需要修改的参数：model和tensor-parallel-size。

model：保存训练后权重转换完的huggingface模型地址；

tensor-parallel-size：张量并行副本数，TP建议和训练时infer的配置保持一致；

data-parallel-size：数据并行副本数，DP建议和训练时infer的配置保持一致，默认为1；

port：可任意设置空闲端口；

~~~bash
vllm serve /path/to/Qwen3-30B/ \
       --served-model-name auto \
       --gpu-memory-utilization 0.9 \
       --max-num-seqs 24 \
       --max-model-len 22528 \
       --max-num-batched-tokens 22528 \
       --enforce-eager \
       --trust-remote-code \
       --distributed_executor_backend=mp \
       --tensor-parallel-size 8 \
       --data-parallel-size 1 \
       --generation-config vllm \
       --port 8080
~~~

**修改AISBench推理配置启动vllm_client评测**

编辑 `benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py`，配置 AISBench 通过流式 Chat 接口调用 vLLM。以下为示例取值，输入和输出的 token 总数须在服务端 `--max-model-len` 限制内。参数说明见 [AISBench 模型配置文档](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/all_params/models.html)。

~~~python
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="",
        model="",
        request_rate = 0,  # 每秒发送的请求数，0 表示不限速
        retry = 2,  # 连接失败后的最大重试次数
        host_ip = "localhost",
        host_port = 8080,  # 与 vLLM 服务端的 --port 一致
        max_out_len = 512,  # 单次响应最多生成的 token 数
        batch_size=1,  # 最多同时发送的请求数
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0.5,  # 采样温度，0 表示贪心解码
            top_k = 10,  # 仅从概率最高的 10 个 token 中采样
            top_p = 0.95,  # 按概率从高到低选取候选 token，直到累计概率达到 0.95
            seed = None,
            repetition_penalty = 1.03,  # 大于 1 时降低已出现 token 再次被选中的概率
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
~~~

另起一个窗口进行评测，开启评测命令：
~~~bash
    ais_bench --models vllm_api_stream_chat --datasets math500_gen_0_shot_cot_chat_prompt
~~~
## 5.sglang推理评测
参照 [昇腾 SGLang 最佳实践](../../model_support/examples/ascend_sglang_best_practices.rst)中评测进行