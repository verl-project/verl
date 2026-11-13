# Model Evaluation

Last updated: 07/14/2026.

The steps are the same for different models; Qwen3-30B is used here as an example.

We evaluate the model through AISBench, which supports evaluation with multiple inference backends such as vllm/sglang.

## 1. Installation

~~~bash
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark
pip install -e .
~~~


## 2. Download the Evaluation Dataset

~~~bash
cd path/to/benchmark/ais_bench/datasets
wget https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip
unzip math.zip
rm math.zip
~~~

## 3. Weight Conversion

verl now supports saving HF-format model weights directly using mbridge, so no conversion is needed.

If the model weights are not in HF format, convert them to HF format first before evaluation.

Refer to verl's native [conversion method](../../../../../docs/advance/checkpoint.rst).

## 4. vLLM Inference Evaluation

**Start the vllm_server service**

Launch the inference server with the following command. Parameters to modify: model and tensor-parallel-size.

model: the path to the converted HuggingFace model weights saved after training;

tensor-parallel-size: the number of tensor-parallel replicas. TP is recommended to match the infer configuration used during training;

data-parallel-size: the number of data-parallel replicas. DP is recommended to match the infer configuration used during training, default is 1;

port: any free port can be set;

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

**Modify the AISBench inference configuration and launch the vllm_client evaluation**

Open the inference configuration file benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py

host_port must match the server's port; modify max_seq_len and max_out_len according to the model configuration.
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

Open another terminal to run the evaluation, and launch the evaluation command:
~~~bash
    ais_bench --models vllm_api_stream_chat --datasets math500_gen_0_shot_cot_chat_prompt
~~~
## 5. SGLang Inference Evaluation
Refer to the evaluation section in [Ascend SGLang Best Practices](../../model_support/examples/ascend_sglang_best_practices.rst).
