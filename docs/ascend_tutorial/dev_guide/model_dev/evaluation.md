# 模型评测

不同模型步骤一致,仅以Qwen3-30B为例列举

我们通过 AISBenchmark 评估模型,该工具支持vllm/sglang多种推理后端的评估

## 1.安装方法

~~~bash
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark
pip install -e .
~~~


## 2.下载评估数据集

~~~bash
cd path/to/benchmark/ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip
unzip math.zip
rm math.zip
~~~

## 3.权重转换

当前verl已经支持mbridge直接保存hf格式模型权重,无需转换即可使用.

如果模型权重不是hf格式,需要先转换为hf格式,再进行评估

此处参照verl原生[转换方法](verl\docs\advance\checkpoint.rst)

## 4.vllm推理评测

**启动vllm_server服务**

通过以下命令拉起NPU服务端，需要修改的参数：model和tensor-parallel-size。

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
       --port 6380
~~~

**修改aisbench推理配置启动vllm_client评测**

打开推理配置文件 benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py 

host_port需与服务端的port一致，根据模型配置修改max_seq_len和max_out_len
~~~bash
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="",
        model="",
        request_rate = 0,
        retry = 2,
        host_ip = "localhost",
        host_port = 8080,
        max_out_len = 512,
        batch_size=1,
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            seed = None,
            repetition_penalty = 1.03,
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
**修改AISBench配置代码使能sglang推理评测**
打开推理配置文件 benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py 

~~~python
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
~~~



**启动sglang_server服务**

~~~bash
    python -m sglang.launch_server --model-path "/path/to/Qwen3-30B"  --tp-size 4 --dp-size 1 --port 8005 
~~~

**启动sglang_client评测**

~~~bash
    ais_bench --models vllm_api_stream_chat --datasets math500_gen_0_shot_cot_chat_prompt
~~~