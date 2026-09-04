unset ROCR_VISIBLE_DEVICES
module load cuda
export PYTHONPATH=/ocean/projects/med230010p/yji3/BrowseCamp/verl:$PYTHONPATH
export CUDA_HOME=/opt/packages/cuda/v12.6.1
export CUDA_PATH=/opt/packages/cuda/v12.6.1
export LD_LIBRARY_PATH=/opt/packages/cuda/v12.6.1/lib64:/opt/packages/cuda/v12.6.1/nvvm/lib64:/opt/packages/cuda/v12.6.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export PATH=/opt/packages/cuda/v12.6.1/bin:$PATH

# ── 1. checker no triage (step93) ─────────────────────────────
CUDA_VISIBLE_DEVICES=2 python evaluate/evaluate_search_r1.py \
    --repo_root /ocean/projects/med230010p/yji3/BrowseCamp/verl \
    --model_path merged_qwen2.5_7b_combined_search_checker_no_triage_step_93 \
    --test_file /ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/combined__medical/test.parquet \
    --max_samples 100 \
    --eval_batch_size 4 \
    --output_file eval_hybrid_checker_no_triage.json \
    --tool_count_mode both \
    --tag_style auto \
    --prompt_mode clean \
    --tool_config_path /ocean/projects/med230010p/yji3/BrowseCamp/verl/examples/sglang_multiturn/config/tool_config/medical_search_checker_tool_config.yaml \
    --multi_turn_format search_r1_with_checker \
    --tensor_parallel_size 1 \
    --nnodes 1 \
    --n_gpus_per_node 1 \
    --gpu_memory_utilization 0.4 \
    --max_model_len 8000 \
    --max_prompt_length 2048 \
    --max_response_length 2000 \
    --max_assistant_turns 7 \
    --max_tool_response_length 768

# ── 2. checker triage (step93) ────────────────────────────────

unset ROCR_VISIBLE_DEVICES
module load cuda
export PYTHONPATH=/ocean/projects/med230010p/yji3/BrowseCamp/verl:$PYTHONPATH
export CUDA_HOME=/opt/packages/cuda/v12.6.1
export CUDA_PATH=/opt/packages/cuda/v12.6.1
export LD_LIBRARY_PATH=/opt/packages/cuda/v12.6.1/lib64:/opt/packages/cuda/v12.6.1/nvvm/lib64:/opt/packages/cuda/v12.6.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export PATH=/opt/packages/cuda/v12.6.1/bin:$PATH
CUDA_VISIBLE_DEVICES=3 python evaluate/evaluate_search_r1.py \
    --repo_root /ocean/projects/med230010p/yji3/BrowseCamp/verl \
    --model_path merged_qwen2.5_7b_combined_search_checker_triage_step_93 \
    --test_file /ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/combined__medical/test.parquet \
    --max_samples 100 \
    --eval_batch_size 4 \
    --output_file eval_hybrid_checker_triage.json \
    --tool_count_mode both \
    --tag_style auto \
    --prompt_mode clean \
    --tool_config_path /ocean/projects/med230010p/yji3/BrowseCamp/verl/examples/sglang_multiturn/config/tool_config/medical_search_checker_tool_config.yaml \
    --multi_turn_format search_r1_with_checker \
    --tensor_parallel_size 1 \
    --nnodes 1 \
    --n_gpus_per_node 1 \
    --gpu_memory_utilization 0.4 \
    --max_model_len 8000 \
    --max_prompt_length 2048 \
    --max_response_length 2000 \
    --max_assistant_turns 7 \
    --max_tool_response_length 768 \
    --enable_triage \
    --online_escalation

# ── 3. no triage explicit check (step188) ────────────────────
unset ROCR_VISIBLE_DEVICES
module load cuda
export PYTHONPATH=/ocean/projects/med230010p/yji3/BrowseCamp/verl:$PYTHONPATH
export CUDA_HOME=/opt/packages/cuda/v12.6.1
export CUDA_PATH=/opt/packages/cuda/v12.6.1
export LD_LIBRARY_PATH=/opt/packages/cuda/v12.6.1/lib64:/opt/packages/cuda/v12.6.1/nvvm/lib64:/opt/packages/cuda/v12.6.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export PATH=/opt/packages/cuda/v12.6.1/bin:$PATH
CUDA_VISIBLE_DEVICES=2 python evaluate/evaluate_search_r1.py \
    --repo_root /ocean/projects/med230010p/yji3/BrowseCamp/verl \
    --model_path /ocean/projects/med230010p/yji3/BrowseCamp/verl/merged_models/qwen2.5-7b-combined-search-checker-no-triage-explicitcheck-20-13-17-step188 \
    --test_file /ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/combined__medical/test.parquet \
    --max_samples 100 \
    --eval_batch_size 4 \
    --output_file eval_hybrid_no_triage_explicit_step188.json \
    --tool_count_mode both \
    --tag_style auto \
    --prompt_mode explicit_check \
    --tool_config_path /ocean/projects/med230010p/yji3/BrowseCamp/verl/examples/sglang_multiturn/config/tool_config/medical_search_checker_tool_config.yaml \
    --multi_turn_format search_r1_with_checker \
    --tensor_parallel_size 1 \
    --nnodes 1 \
    --n_gpus_per_node 1 \
    --gpu_memory_utilization 0.4 \
    --max_model_len 8000 \
    --max_prompt_length 3072 \
    --max_response_length 2000 \
    --max_assistant_turns 7 \
    --max_tool_response_length 768

# ── 4. triage explicit check (step188) ───────────────────────

unset ROCR_VISIBLE_DEVICES
module load cuda
export PYTHONPATH=/ocean/projects/med230010p/yji3/BrowseCamp/verl:$PYTHONPATH
export CUDA_HOME=/opt/packages/cuda/v12.6.1
export CUDA_PATH=/opt/packages/cuda/v12.6.1
export LD_LIBRARY_PATH=/opt/packages/cuda/v12.6.1/lib64:/opt/packages/cuda/v12.6.1/nvvm/lib64:/opt/packages/cuda/v12.6.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export PATH=/opt/packages/cuda/v12.6.1/bin:$PATH
CUDA_VISIBLE_DEVICES=3 python evaluate/evaluate_search_r1.py \
    --repo_root /ocean/projects/med230010p/yji3/BrowseCamp/verl \
    --model_path /ocean/projects/med230010p/yji3/BrowseCamp/verl/merged_models/qwen2.5-7b-combined-search-checker-triage-explicitcheck-20-15-22-step188 \
    --test_file /ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/combined__medical/test.parquet \
    --max_samples 100 \
    --eval_batch_size 4 \
    --output_file eval_hybrid_triage_explicit_step188.json \
    --tool_count_mode both \
    --tag_style auto \
    --prompt_mode explicit_check \
    --tool_config_path /ocean/projects/med230010p/yji3/BrowseCamp/verl/examples/sglang_multiturn/config/tool_config/medical_search_checker_tool_config.yaml \
    --multi_turn_format search_r1_with_checker \
    --tensor_parallel_size 1 \
    --nnodes 1 \
    --n_gpus_per_node 1 \
    --gpu_memory_utilization 0.4 \
    --max_model_len 8000 \
    --max_prompt_length 3072 \
    --max_response_length 2000 \
    --max_assistant_turns 7 \
    --max_tool_response_length 768 \
    --enable_triage \
    --online_escalation