# Upgrading to vllm >= 0.7

Note: verl+vllm 0.8.3 is now stable. Please see ``docs/README_vllm0.8.md`` for upgrade guide.

## Installation

Note: At time of writing, verl+vllm 0.7.x supports **FSDP** for training and **vLLM** for rollout.

```
# Create the conda environment
conda create -n verl python==3.10
conda activate verl

# Install verl
git clone https://github.com/verl-project/verl.git
cd verl
pip3 install -e .

# Install the latest stable version of vLLM
pip3 install vllm==0.7.3 

# Install flash-attn
pip3 install flash-attn --no-build-isolation

```

Note that if you are installing lower versions of vLLM (0.7.0, 0.7.1, 0.7.2), you need to make some tiny patches manually on vllm (/path/to/site-packages/vllm after installation) after the above steps:

- vllm/distributed/parallel_state.py: Remove the assertion below:

```
if (world_size
        != tensor_model_parallel_size * pipeline_model_parallel_size):
    raise RuntimeError(
        f"world_size ({world_size}) is not equal to "
        f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
        f"pipeline_model_parallel_size ({pipeline_model_parallel_size})")

```

- vllm/executor/uniproc_executor.py: change `local_rank = rank` to `local_rank = int(os.environ["LOCAL_RANK"])`
- vllm/model_executor/model_loader/weight_utils.py: remove the `torch.cuda.empty_cache()` in `pt_weights_iterator`

## Features

### Use cuda graph

After installation, examples using FSDP as training backends can be used. By default, the `enforce_eager` is set to True, which disables the cuda graph. To enjoy cuda graphs and the sleep mode of vLLM>=0.7, add the following lines to the bash script:

```
actor_rollout_ref.rollout.enforce_eager=False \
actor_rollout_ref.rollout.free_cache_engine=True \

```

For a typical job like examples/ppo_trainer/run_qwen3_8b_fsdp.sh, the rollout generation time is 85 seconds with vLLM0.7.0. By enabling the cudagraph, the generation duration is further reduced to 62 seconds.

**Note:** Currently, if the `n` is greater than 1 in `SamplingParams` in vLLM>=0.7, there is a potential performance issue on the stability of rollout generation time (Some iterations would see generation time bursts) using vLLM's V0 Engine.

### Use vLLM V1 Engine

Using the vLLM V1 engine can avoid instability issues and achieve additional performance improvements. To use the V1 engine, you can first uninstall the previously installed vLLM and then follow the steps below to install the newer version.

```
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 2275784
sed -i "903a\    data_parallel_size = world_size // pipeline_model_parallel_size // tensor_model_parallel_size" ./vllm/distributed/parallel_state.py
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

Then you can enable the V1 engine by setting `export VLLM_USE_V1=1`. In some benchmark tests, the V1 engine demonstrates a 1.5x speed improvement over the vLLM V0 engine.
The stable support of the vLLM V1 engine is available on verl main.
