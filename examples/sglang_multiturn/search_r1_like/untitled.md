CUDA_VISIBLE_DEVICES=2,3 \
bash examples/sglang_multiturn/search_r1_like/run_small_qwen2.5-7b_search_checker_ablation_2gpu.sh \
    checker_guarded \
    2>&1 | tee train_$(date +%d-%H-%M).log