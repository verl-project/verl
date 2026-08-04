# bootcamp Training with Xtuner



## 🚄 Training Tutorial

### 1. Install Dependencies

We utilizes [XTuner](https://github.com/InternLM/xtuner/tree/main) as the training engine. 

You should make sure that InternBootcamp is successfully installed.

```bash
pip install -e $InternBootcamp_path
```

Then install xtuner and its dependencies.

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation
pip install xtuner[all]==0.2.0rc0
```

### 2. Prepare Data


The bootcamp data can be transfered into training format by using examples/xpuyu_usage/xpuyu_data_preprocess.py. 


**Example usage:**
```python
python examples/xpuyu_usage/xpuyu_preprocess.py --src examples/bootcamp_generator_outputs/{%Y-%m-%d-%H:%M:%S}
```



### 3. Prepare your training config

Prepare your training config for starting GRPO training.

An example config is in

```
examples/xpuyu_usage/bootcamp_rl/configs/example_training_config.py
```


### 4. Start Training


```bash
cd examples/xpuyu_usage

GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# Number of GPU workers, for single-worker training, please set to 1
NNODES=${WORLD_SIZE:-1} # modified to adapt cluster

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${RANK:-0} # modified to adapt cluster

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo $DISTRIBUTED_ARGS

torchrun $DISTRIBUTED_ARGS train_grpo.py ./bootcamp_rl/configs/example_training_config.py --work_dir examples/xpuyu_usage/ckpts/experiment_name
```


### 5. Training Curve Visualization

You could use examples/xpuyu_usage/report_to_wandb.py to visualize your training curve.

```bash
python examples/xpuyu_usage/report_to_wandb.py examples/xpuyu_usage/ckpts/{experiment_name}/{timestamp}/rank0.log.jsonl {wandb_project_name}
```


