#!/usr/bin/env python3
"""
Python launcher for VERL math RL training.

Usage:
    python launch_math_rl.py 2 64 --epochs=10
    python launch_math_rl.py 1 64 --epochs=5 --gsm8k=True  # Test on smaller data
    python launch_math_rl.py 2 64 --local=True
"""

import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import fire


def calculate_batch_sizes(nodes: int, batch_size: int):
    """Calculate appropriate batch sizes based on number of nodes."""
    # Each node has 8 GPUs, so total GPUs = nodes * 8
    total_gpus = nodes * 8
    
    # PPO mini batch size should be a divisor of batch_size
    # and reasonable for the number of GPUs
    ppo_mini_batch_size = max(1, min(32, batch_size // 4))
    
    # PPO micro batch size per GPU should be small enough to fit in memory
    # but not too small to be inefficient
    ppo_micro_batch_size_per_gpu = max(1, min(4, batch_size // total_gpus))
    
    return ppo_mini_batch_size, ppo_micro_batch_size_per_gpu


def calculate_max_batched_tokens(max_prompt_length: int, max_response_length: int, rollout_max_bsz: int = 4) -> int:
    """Calculate max_num_batched_tokens based on prompt + response length and rollout batch size."""
    return (max_prompt_length + max_response_length) * rollout_max_bsz


def generate_experiment_name(
    nodes: int, 
    batch_size: int, 
    epochs: int,
    model_name: str,
    gsm8k: bool = False,
    custom_name: Optional[str] = None
) -> str:
    """Generate a descriptive experiment name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.replace("Qwen/", "").replace("Qwen2.5-", "qwen2.5-").lower()
    
    dataset_suffix = ".gsm8k" if gsm8k else ""
    
    if custom_name:
        return f"{custom_name}.{model_short}.n{nodes}.bsz{batch_size}.ep{epochs}{dataset_suffix}.{timestamp}"
    else:
        return f"verl-math-rl.{model_short}.n{nodes}.bsz{batch_size}.ep{epochs}{dataset_suffix}.{timestamp}"


def create_log_directory(experiment_name: str) -> Path:
    """Create log directory structure."""
    log_base = Path.home() / "verl_log"
    log_dir = log_base / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_data_files(data_path: str, gsm8k: bool = False):
    """Get training and validation data file paths."""
    if gsm8k:
        # Use GSM8K data for testing (smaller dataset)
        train_files = f"{data_path}/gsm8k/train.parquet"
        val_files = f"{data_path}/gsm8k/test.parquet"
    else:
        # Use default polaris data
        train_files = f"{data_path}/polaris/train.parquet"
        val_files = f"{data_path}/polaris/test.parquet"
    
    return train_files, val_files


def format_verl_command(
    experiment_name: str,
    train_files: str,
    val_files: str,
    model_path: str,
    train_batch_size: int,
    ppo_mini_batch_size: int,
    ppo_micro_batch_size_per_gpu: int,
    total_epochs: int,
    nodes: int,
    max_prompt_length: int = 2048,
    max_response_length: int = 4096,
    project: str = "ssverl",
    # Performance parameters
    max_num_batched_tokens_override: Optional[int] = None,
    gpu_memory_utilization: float = 0.4,
    tensor_model_parallel_size: int = 1,
    use_remove_padding: bool = True,
    enable_gradient_checkpointing: bool = True,
    param_offload: bool = True,
    optimizer_offload: bool = True,
    use_dynamic_bsz: bool = False,
    ref_log_prob_micro_batch_size_per_gpu: Optional[int] = None,
    rollout_log_prob_micro_batch_size_per_gpu: Optional[int] = None,
    critic_forward_micro_batch_size_per_gpu: Optional[int] = None,
    critic_ppo_micro_batch_size_per_gpu: Optional[int] = None,
    use_liger_kernel: bool = False,
    disable_log_stats: bool = False,
    **kwargs
) -> str:
    """Format the VERL training command with all parameters."""
    
    # Calculate max_num_batched_tokens automatically or use override
    if max_num_batched_tokens_override:
        max_num_batched_tokens = max_num_batched_tokens_override
    else:
        max_num_batched_tokens = calculate_max_batched_tokens(max_prompt_length, max_response_length)
    
    # Base command
    command = f"""python3 -m verl.trainer.main_ppo \\
    algorithm.adv_estimator=grpo \\
    data.train_files={train_files} \\
    data.val_files={val_files} \\
    data.train_batch_size={train_batch_size} \\
    data.max_prompt_length={max_prompt_length} \\
    data.max_response_length={max_response_length} \\
    data.filter_overlong_prompts=True \\
    data.truncation=error \\
    actor_rollout_ref.model.path={model_path} \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size} \\
    actor_rollout_ref.actor.use_kl_loss=False \\
    actor_rollout_ref.actor.kl_loss_coef=0.000 \\
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
    actor_rollout_ref.actor.entropy_coeff=0 \\
    actor_rollout_ref.actor.strategy=fsdp2 \\
    actor_rollout_ref.rollout.name=vllm \\
    actor_rollout_ref.rollout.n=8 \\
    actor_rollout_ref.rollout.enforce_eager=False \\
    actor_rollout_ref.rollout.free_cache_engine=True \\
    actor_rollout_ref.rollout.max_num_batched_tokens={max_num_batched_tokens} \\
    actor_rollout_ref.ref.strategy=fsdp2 \\
    algorithm.use_kl_in_reward=False \\
    trainer.critic_warmup=0 \\
    trainer.logger="['console','wandb']" \\
    trainer.project_name={project} \\
    trainer.experiment_name={experiment_name} \\
    trainer.n_gpus_per_node=8 \\
    trainer.nnodes={nodes} \\
    trainer.save_freq=50 \\
    trainer.max_actor_ckpt_to_keep=3 \\
    trainer.max_critic_ckpt_to_keep=3 \\
    trainer.test_freq=3 \\
    trainer.total_epochs={total_epochs} \\
    trainer.val_before_train=True"""
    
    # Add performance optimization parameters
    # GPU memory utilization
    command += f" \\\n    actor_rollout_ref.rollout.gpu_memory_utilization={gpu_memory_utilization}"
    
    # Tensor model parallel size
    command += f" \\\n    actor_rollout_ref.rollout.tensor_model_parallel_size={tensor_model_parallel_size}"
    
    # Remove padding (sequence packing) - proven to be beneficial
    if use_remove_padding:
        command += f" \\\n    actor_rollout_ref.model.use_remove_padding=True"
    
    # Gradient checkpointing - helps with memory efficiency
    if enable_gradient_checkpointing:
        command += f" \\\n    actor_rollout_ref.model.enable_gradient_checkpointing=True"
        command += f" \\\n    critic.model.enable_gradient_checkpointing=True"
    
    # Parameter and optimizer offloading - helps with memory efficiency
    if param_offload:
        command += f" \\\n    actor_rollout_ref.actor.fsdp_config.param_offload=True"
        command += f" \\\n    actor_rollout_ref.ref.fsdp_config.param_offload=True"
        command += f" \\\n    critic.model.fsdp_config.param_offload=True"
    
    if optimizer_offload:
        command += f" \\\n    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
        command += f" \\\n    critic.model.fsdp_config.optimizer_offload=True"
    
    # LigerKernel optimization
    if use_liger_kernel:
        command += f" \\\n    actor_rollout_ref.model.use_liger_kernel=True"
        command += f" \\\n    critic.model.use_liger_kernel=True"
    
    # Disable log stats for rollout
    if disable_log_stats:
        command += f" \\\n    actor_rollout_ref.rollout.disable_log_stats=True"
    
    # Dynamic batch size parameters
    if use_dynamic_bsz:
        command += f" \\\n    actor_rollout_ref.actor.use_dynamic_bsz=True"
        command += f" \\\n    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"
        
        # Add dynamic batch size token limits
        for key, value in kwargs.items():
            if 'max_token_len_per_gpu' in key:
                if key == 'ppo_max_token_len_per_gpu':
                    command += f" \\\n    actor_rollout_ref.actor.ppo_max_token_len_per_gpu={value}"
                elif key == 'ref_log_prob_max_token_len_per_gpu':
                    command += f" \\\n    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={value}"
                elif key == 'rollout_log_prob_max_token_len_per_gpu':
                    command += f" \\\n    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={value}"
    else:
        # Standard micro batch size parameters
        command += f" \\\n    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={ppo_micro_batch_size_per_gpu}"
        
        # Override micro batch sizes if provided
        if ref_log_prob_micro_batch_size_per_gpu:
            command += f" \\\n    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={ref_log_prob_micro_batch_size_per_gpu}"
        else:
            command += f" \\\n    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={ppo_micro_batch_size_per_gpu * 2}"
        
        if rollout_log_prob_micro_batch_size_per_gpu:
            command += f" \\\n    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={rollout_log_prob_micro_batch_size_per_gpu}"
        else:
            command += f" \\\n    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={ppo_micro_batch_size_per_gpu * 2}"
        
        if critic_forward_micro_batch_size_per_gpu:
            command += f" \\\n    critic.forward_micro_batch_size_per_gpu={critic_forward_micro_batch_size_per_gpu}"
        
        if critic_ppo_micro_batch_size_per_gpu:
            command += f" \\\n    critic.ppo_micro_batch_size_per_gpu={critic_ppo_micro_batch_size_per_gpu}"
    
    return command


def fill_template(
    template_path: Path,
    **kwargs
) -> str:
    """Fill the sbatch template with provided values."""
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    for key, value in kwargs.items():
        placeholder = f"${{{key}}}"
        template_content = template_content.replace(placeholder, str(value))
    
    return template_content


def submit_job(sbatch_content: str, experiment_name: str) -> str:
    """Submit the job using sbatch and return job ID."""
    # Create temporary file for sbatch script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(sbatch_content)
        temp_script_path = f.name
    
    try:
        # Make the script executable
        os.chmod(temp_script_path, 0o755)
        
        # Submit the job
        result = subprocess.run(
            ["sbatch", temp_script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract job ID from output (typically "Submitted batch job 12345")
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted job {job_id} for experiment: {experiment_name}")
        print(f"Log file: ~/verl_log/{experiment_name}/train.log")
        
        return job_id
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise
    finally:
        # Clean up temporary file
        os.unlink(temp_script_path)


def run_local_command(command: str, experiment_name: str, log_dir: Path, conda_env: str = "verl"):
    """Run the command locally instead of submitting to slurm."""
    print(f"Running locally: {experiment_name}")
    print(f"Log directory: {log_dir}")
    
    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"
    
    # Prepare the command with conda activation
    full_command = f"""
    cd {Path.home() / "verl"}
    eval "$(/home/sam/miniconda3/bin/conda shell.bash hook)"
    conda activate {conda_env}
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export VLLM_USE_V1=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
    export HF_HOME=$TRANSFORMERS_CACHE
    
    # Run the command
    {command} 2>&1 | tee {log_file}
    """
    
    print(f"Executing command:")
    print(full_command)
    print(f"Logs will be written to: {log_file}")
    
    # Execute the command
    import subprocess
    try:
        result = subprocess.run(
            full_command,
            shell=True,
            executable="/bin/bash",
            check=False  # Don't raise exception on non-zero exit
        )
        print(f"Command finished with exit code: {result.returncode}")
        return result.returncode
    except Exception as e:
        print(f"Error running command: {e}")
        return 1


def main(
    # Required arguments
    nodes: int = 2,
    batch_size: int = 64,
    # Optional arguments
    epochs: int = 20,
    model_path: str = "Qwen/Qwen2.5-3B-Instruct",
    data_path: str = "$HOME/data",
    experiment_name: Optional[str] = None,
    project: str = "ssverl",
    max_prompt_length: int = 2048,
    max_response_length: int = 4096,
    gsm8k: bool = False,
    dry_run: bool = False,
    local: bool = False,
    # Performance tuning arguments - optimized defaults based on test results
    use_remove_padding: bool = True,  # Proven beneficial
    gpu_memory_utilization: float = 0.4,  # 0.6 caused memory issues
    tensor_model_parallel_size: int = 1,  # Default to 1 for simplicity
    enable_gradient_checkpointing: bool = True,  # Helps with memory
    param_offload: bool = True,  # Helps with memory
    optimizer_offload: bool = True,  # Helps with memory
    use_dynamic_bsz: bool = False,  # Can be enabled for advanced tuning
    use_liger_kernel: bool = False,  # Can be enabled for optimization
    disable_log_stats: bool = False,  # Keep stats enabled by default
    # Micro batch size overrides
    ppo_micro_batch_size_per_gpu: Optional[int] = None,
    ref_log_prob_micro_batch_size_per_gpu: Optional[int] = None,
    rollout_log_prob_micro_batch_size_per_gpu: Optional[int] = None,
    critic_forward_micro_batch_size_per_gpu: Optional[int] = None,
    critic_ppo_micro_batch_size_per_gpu: Optional[int] = None,
    # Dynamic batch size arguments
    ppo_max_token_len_per_gpu: Optional[int] = None,
    ref_log_prob_max_token_len_per_gpu: Optional[int] = None,
    rollout_log_prob_max_token_len_per_gpu: Optional[int] = None,
    # Rollout tuning arguments
    max_num_batched_tokens_override: Optional[int] = None,
    # Conda environment
    conda_env: str = "verl",
):
    """Launch VERL math RL training with optimized defaults for 2-node 3B model."""
    
    # Validate arguments
    if nodes <= 0:
        raise ValueError("Number of nodes must be positive")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    if max_prompt_length <= 0:
        raise ValueError("Max prompt length must be positive")
    if max_response_length <= 0:
        raise ValueError("Max response length must be positive")
    
    # Calculate dynamic batch size parameters
    dynamic_batch_params = {}
    if use_dynamic_bsz:
        # Calculate recommended token lengths based on sequence lengths
        base_token_len = max_prompt_length + max_response_length
        
        # Set default values if not provided
        if ppo_max_token_len_per_gpu is None:
            dynamic_batch_params['ppo_max_token_len_per_gpu'] = base_token_len * 2
        
        if ref_log_prob_max_token_len_per_gpu is None:
            dynamic_batch_params['ref_log_prob_max_token_len_per_gpu'] = base_token_len * 3
        
        if rollout_log_prob_max_token_len_per_gpu is None:
            dynamic_batch_params['rollout_log_prob_max_token_len_per_gpu'] = base_token_len * 3
    
    # Calculate batch sizes
    ppo_mini_batch_size, ppo_micro_batch_size_per_gpu = calculate_batch_sizes(
        nodes, batch_size
    )
    
    # Generate experiment name
    experiment_name = generate_experiment_name(
        nodes, batch_size, epochs, model_path, 
        gsm8k, experiment_name
    )
    
    # Create log directory
    log_dir = create_log_directory(experiment_name)
    
    # Get data files
    train_files, val_files = get_data_files(data_path, gsm8k)
    
    # Format the VERL command
    verl_command = format_verl_command(
        experiment_name=experiment_name,
        train_files=train_files,
        val_files=val_files,
        model_path=model_path,
        train_batch_size=batch_size,
        ppo_mini_batch_size=ppo_mini_batch_size,
        ppo_micro_batch_size_per_gpu=ppo_micro_batch_size_per_gpu,
        total_epochs=epochs,
        nodes=nodes,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        project=project,
        max_num_batched_tokens_override=max_num_batched_tokens_override,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_model_parallel_size=tensor_model_parallel_size,
        use_remove_padding=use_remove_padding,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
        param_offload=param_offload,
        optimizer_offload=optimizer_offload,
        use_dynamic_bsz=use_dynamic_bsz,
        ref_log_prob_micro_batch_size_per_gpu=ref_log_prob_micro_batch_size_per_gpu,
        rollout_log_prob_micro_batch_size_per_gpu=rollout_log_prob_micro_batch_size_per_gpu,
        critic_forward_micro_batch_size_per_gpu=critic_forward_micro_batch_size_per_gpu,
        critic_ppo_micro_batch_size_per_gpu=critic_ppo_micro_batch_size_per_gpu,
        use_liger_kernel=use_liger_kernel,
        disable_log_stats=disable_log_stats,
        **dynamic_batch_params
    )
    
    # Prepare template variables
    template_vars = {
        "JOB_NAME": experiment_name,
        "NUM_NODES": nodes,
        "LOGDIR": str(log_dir),
        "CODE_DIR": str(Path.home() / "verl"),
        "DATA_PATH": data_path,
        "MODEL_PATH": model_path,
        "EXPERIMENT_NAME": experiment_name,
        "VERL_COMMAND": verl_command,
        "CONDA_ENV": conda_env,
    }
    
    # Fill template
    template_path = Path(__file__).parent / "sbatch_template.sh"
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    sbatch_content = fill_template(template_path, **template_vars)
    
    # Print configuration
    print(f"=== VERL Training Configuration ===")
    print(f"Experiment: {experiment_name}")
    print(f"Model: {model_path}")
    print(f"Nodes: {nodes} (Total GPUs: {nodes * 8})")
    print(f"Batch Size: {batch_size}")
    print(f"Mini Batch Size: {ppo_mini_batch_size}")
    print(f"Micro Batch Size per GPU: {ppo_micro_batch_size_per_gpu}")
    print(f"Epochs: {epochs}")
    print(f"Dataset: {'GSM8K' if gsm8k else 'Polaris'}")
    print(f"Performance Settings:")
    print(f"  Remove Padding: {use_remove_padding}")
    print(f"  GPU Memory Utilization: {gpu_memory_utilization}")
    print(f"  Gradient Checkpointing: {enable_gradient_checkpointing}")
    print(f"  Parameter Offload: {param_offload}")
    print(f"  Optimizer Offload: {optimizer_offload}")
    print(f"  Dynamic Batch Size: {use_dynamic_bsz}")
    print(f"  LigerKernel: {use_liger_kernel}")
    
    # Handle local execution
    if local:
        print("Running locally...")
        exit_code = run_local_command(verl_command, experiment_name, log_dir, conda_env)
        print(f"Local execution finished with exit code: {exit_code}")
        return
    
    if dry_run:
        print("Dry run mode - would submit the following sbatch script:")
        print("=" * 80)
        print(sbatch_content)
        print("=" * 80)
        print(f"VERL Command:")
        print(verl_command)
        print("=" * 80)
    else:
        # Submit the job
        job_id = submit_job(sbatch_content, experiment_name)
        print(f"Job submitted successfully with ID: {job_id}")
        print(f"Monitor with: squeue -j {job_id}")
        print(f"Check logs with: tail -f {log_dir}/train.log")


if __name__ == "__main__":
    fire.Fire(main)