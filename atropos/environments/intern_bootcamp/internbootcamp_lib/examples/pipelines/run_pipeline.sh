#!/bin/bash


# 检查 jq 是否已安装
if ! dpkg -l | grep -q '^ii  jq '; then
    echo "jq 未安装，正在安装..."
    sudo apt-get update
    sudo apt-get install jq -y
fi
# 时间戳
timestamp=$(date +"%Y-%m-%d-%H:%M:%S")
# cipher输入集
cipher_input_file='internbootcamp/libs/data/words_alpha_370000.txt'

tokenizer="your tokenizer path" # tokenizer is used to calculate the sequence length of the prompt
max_prompt_len=4096
max_jobs=64  # 设置最大并发进程数
jobs=()     # 用于存储后台进程的PID

cipher_test_nums_for_single_cipher=0
cipehr_train_nums_for_single_cipher=0

while IFS= read -r line || [ -n "$line" ]; do
    # 跳过空行
    if [ -z "$line" ]; then
        continue
    fi

    # 解析JSON行并提取变量
    bootcamp_name=$(echo "$line" | jq -r '.bootcamp_name')
    declare -i sample_number=$(echo "$line" | jq -r '.sample_number')
    config_file=$(echo "$line" | jq -r '.config_file')
    bootcamp_cls_name=$(echo "$line" | jq -r '.bootcamp_cls_name')

    # 如果 config_file 为 "cipher"，保存 sample_number
    if [[ "$config_file" == "cipher" ]]; then
        cipehr_train_nums_for_single_cipher=$sample_number
        continue
    fi

    # 异步运行Python脚本
    python examples/pipelines/data_generator.py \
        --bootcamp_name "$bootcamp_name" \
        --n $sample_number \
        --save_file "examples/bootcamp_generator_outputs/$timestamp/train/${bootcamp_name}.jsonl" \
        --config_file "examples/pipelines/puzzle_configs/${config_file}_train.json" \
        --bootcamp_cls_name "$bootcamp_cls_name" \
        --tokenizer "$tokenizer" \
        --max_prompt_len $max_prompt_len \
        --shuffle 

    # If there is no problem with the above command, you can use the following line to run it in multiple processes, replacing the above command
    # python examples/pipelines/data_generator.py \
    #     --bootcamp_name "$bootcamp_name" \
    #     --n $sample_number \
    #     --save_file "examples/bootcamp_generator_outputs/$timestamp/train/${bootcamp_name}.jsonl" \
    #     --config_file "examples/pipelines/puzzle_configs/${config_file}_train.json" \
    #     --bootcamp_cls_name "$bootcamp_cls_name" \
    #     --tokenizer "$tokenizer" \
    #     --max_prompt_len $max_prompt_len \
    #     --shuffle &

    pid=$!  # 获取后台进程的PID
    jobs+=("$pid")  # 将PID加入数组

    # 控制并发数量
    while [ ${#jobs[@]} -ge $max_jobs ]; do
        wait -n  # 等待任意一个子进程结束
        # 清理已结束的进程的PID
        new_jobs=()
        for job_pid in "${jobs[@]}"; do
            if kill -0 "$job_pid" 2>/dev/null; then
                new_jobs+=("$job_pid")
            fi
        done
        jobs=("${new_jobs[@]}")
    done
done < examples/pipelines/data_configs/data_config_train.jsonl


while IFS= read -r line || [ -n "$line" ]; do
    # 跳过空行
    if [ -z "$line" ]; then
        continue
    fi

    # 解析JSON行并提取变量
    bootcamp_name=$(echo "$line" | jq -r '.bootcamp_name')
    declare -i sample_number=$(echo "$line" | jq -r '.sample_number')
    config_file=$(echo "$line" | jq -r '.config_file')
    bootcamp_cls_name=$(echo "$line" | jq -r '.bootcamp_cls_name')

    # 如果 config_file 为 "cipher"，保存 sample_number
    if [[ "$config_file" == "cipher" ]]; then
        cipher_test_nums_for_single_cipher=$sample_number
        continue
    fi

    # 异步运行Python脚本
    python examples/pipelines/data_generator.py \
        --bootcamp_name "$bootcamp_name" \
        --n $sample_number \
        --save_file "examples/bootcamp_generator_outputs/$timestamp/test/${bootcamp_name}.jsonl" \
        --config_file "examples/pipelines/puzzle_configs/${config_file}_test.json" \
        --tokenizer "$tokenizer" \
        --bootcamp_cls_name "$bootcamp_cls_name" \
        --max_prompt_len $max_prompt_len \
        --shuffle &
    pid=$!  # 获取后台进程的PID
    jobs+=("$pid")  # 将PID加入数组

    # 控制并发数量
    while [ ${#jobs[@]} -ge $max_jobs ]; do
        wait -n  # 等待任意一个子进程结束
        # 清理已结束的进程的PID
        new_jobs=()
        for job_pid in "${jobs[@]}"; do
            if kill -0 "$job_pid" 2>/dev/null; then
                new_jobs+=("$job_pid")
            fi
        done
        jobs=("${new_jobs[@]}")
    done
done < examples/pipelines/data_configs/data_config_test.jsonl

# 等待所有后台任务完成
wait

# cipher test-set gen 
python examples/pipelines/cipher_data_generator.py \
    --nums $cipher_test_nums_for_single_cipher \
    --split test \
    --timestamp $timestamp \
    --filepath $cipher_input_file

# cipher train——set gen
python examples/pipelines/cipher_data_generator.py \
    --nums $cipehr_train_nums_for_single_cipher \
    --split train \
    --timestamp $timestamp \
    --filepath $cipher_input_file

wait
