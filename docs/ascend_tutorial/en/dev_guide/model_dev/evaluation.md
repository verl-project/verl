# Model Evaluation

Last updated: 07/14/2026.

The steps are the same for different models. This section lists only Qwen3-30B as an example.

Evaluate the model using AISBench, which supports the evaluation of multiple inference backends, including vllm and sglang.

## 1. Installation Methods

~~~bash
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark
pip install -e .
~~~~~~~~~~~~~~~~


## 2. Download the evaluation dataset

~~~bash
cd path/to/benchmark/ais_bench/datasets
wget https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip
unzip math.zip
rm math.zip
~~~~~~~~~~~

## 3. Weight Conversion

Currently, verl supports mbridge to directly save model weights in Hugging Face format, so you can use them without conversion.

If the model weights are not in the hf format, convert them to the hf format first, and then perform the evaluation.

For reference, see the native verl [conversion method](../../../../../docs/advance/checkpoint.rst).

## 4.vllm Inference Evaluation

**Starting the vllm_server service**

Start the inference server using the following command. You need to modify the following parameters: model and tensor-parallel-size.

model: The path to the huggingface model saved after the post-training weights are converted;

tensor-parallel-size: The number of tensor parallelism replicas. Keep TP consistent with the infer configuration during training;

data-parallel-size: Number of data parallelism replicas. It is recommended that DP remains consistent with the infer configuration during training. The default value is 1;

port: You can set this to any idle port;

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

**Modify the AISBench inference configuration to start the vllm_client evaluation**

Open the inference configuration file benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py

host_port must match the server port. Modify max_seq_len and max_out_len based on the model configuration.
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

Open another window for evaluation, and run the evaluation command:
~~~bash
    ais_bench --models vllm_api_stream_chat --datasets math500_gen_0_shot_cot_chat_prompt
~~~
## 5.sglang inference evaluation
For the evaluation procedure, refer to [Ascend SGLang Best Practices](../../model_support/examples/ascend_sglang_best_practices.rst).