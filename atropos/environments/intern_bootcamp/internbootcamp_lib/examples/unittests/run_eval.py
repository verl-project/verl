import asyncio
import openai
import pandas as pd
import argparse
import json
import os
import random
import jsonlines
import logging
from copy import deepcopy
from internbootcamp.bootcamp import *
from internbootcamp.bootcamp_utils.formatted_time import formatted_time
from datetime import datetime, timedelta

# Disable logging
logging.disable(logging.CRITICAL)

TEMPLATE_MAP = {
    "r1": {"chat_template":"<｜begin▁of▁sentence｜><｜User｜>{input}<｜Assistant｜><think>\n","stop_words":["<｜end▁of▁sentence｜>"]}, # r1 new chat template
    "qwen": {"chat_template":"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n","stop_words":["<|im_end|>", "<|endoftext|>"]}, # default qwen template
    "internthinker":{"chat_template":"<|im_start|>system\nYou are an expert reasoner with extensive experience in mathematical and code competitions. You approach problems through systematic thinking and rigorous reasoning. Your response should reflect deep understanding and precise logical thinking, making your solution path and reasoning clear to others. Please put your thinking process within <think>...</think> tags.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n","stop_words":["<|im_end|>", "<|endoftext|>"]},
    "chatml":{"chat_template":"<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n","stop_words":["<|im_end|>", "<|endoftext|>"]}, # No sys prompt chatml
}

# Global file paths, locks, and progress status
progress_file_path = None
progress_file_lock = asyncio.Lock()
progress_status = {}  # Stores the current progress for each file


from datetime import timedelta

def format_progress_bar(current, total, start_time, update_time, bar_length=50):
    """
    Format the progress bar with time statistics.
    """
    # Validate inputs
    if total <= 0:
        print("Total must be greater than 0.")
        return ""
    if current < 0 or current > total:
        print("Current must be between 0 and total.")
        return ""

    # Calculate progress percentage
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    # Calculate time statistics
    elapsed_time = update_time - start_time
    elapsed_time_seconds = int(elapsed_time.total_seconds())  # Extract total seconds from timedelta
    elapsed_time_str = str(timedelta(seconds=elapsed_time_seconds))

    if percent > 0:
        remaining_time = elapsed_time_seconds * (1 - percent) / percent
        remaining_time_str = str(timedelta(seconds=int(remaining_time)))
    else:
        remaining_time_str = "N/A"  # Handle division by zero when percent is 0

    # Format the output string
    if current < total:
        return (
            f"{current}/{total} [{bar}] {percent:.1%} "
            f"Elapsed: {elapsed_time_str} Remaining: {remaining_time_str} "
        )
    else:
        return (
            f"{current}/{total} [{bar}] {percent:.1%} "
            f"Elapsed: {elapsed_time_str} Remaining: {remaining_time_str} "
            f"Completed✔️"
        )


async def update_progress(position, description, total, init_model=False):
    """Update the progress bar with time statistics"""
    update_time = datetime.now()
    global progress_file_path, progress_status
    async with progress_file_lock:
        # Initialize progress for this position if not already done
        if position not in progress_status and not init_model:
            progress_status[position] = {"current": 0, "total": total, "start_time": update_time}
        if init_model:
            current = 0
            start_time = update_time
        else:
            # Increment current progress
            progress_status[position]["current"] += 1 
            current = progress_status[position]["current"]
            total = progress_status[position]["total"]
            start_time = progress_status[position]["start_time"]
        
        # Open the progress file and update the corresponding line
        with open(progress_file_path, 'r+') as f:
            lines = f.readlines()
            # Add empty lines if the file has fewer lines than the position
            while len(lines) < position:
                lines.append("\n")
            # Update the corresponding line content
            lines[position - 1] = f"{description}: {format_progress_bar(current, total, start_time=start_time,update_time=update_time)}\n"
            # Write back to the file
            f.seek(0)
            f.writelines(lines)
            f.truncate()


async def check_model_url_alive(url, api_key, model_name, max_attempts=60, interval=60):
    """
    检查模型 URL 是否存活，并验证指定的 model_name 是否已注册。
    """
    attempt = 0
    print("Checking model URL availability and model registration...")
    while attempt < max_attempts:
        try:
            # 创建 OpenAI 客户端
            async with openai.AsyncOpenAI(base_url=url, api_key=api_key) as client:
                # 获取模型列表
                models = await client.models.list()
                model_ids = [model.id for model in models.data]  # 提取所有模型的 ID
                if model_name in model_ids:
                    print(f"Model '{model_name}' is registered and available after {attempt * interval} seconds.")
                    return True
                else:
                    print(f"Attempt {attempt + 1}: Model '{model_name}' is not registered yet. Retrying in {interval} seconds...")
        except Exception as e:
            print(f"Attempt {attempt + 1}: Model URL not available yet. Error: {str(e)}. Retrying in {interval} seconds...")
        
        # 等待指定的间隔时间后重试
        await asyncio.sleep(interval)
        attempt += 1
    
    # 如果超过最大尝试次数仍未成功，则抛出异常
    raise RuntimeError(f"Model URL or model '{model_name}' did not become available within the maximum allowed time.")


async def process_item(client, item, bootcamp, template, output_dir, semaphore, api_mode, sys_prompt, max_tokens, temperature, timeout, model_name, max_retries, max_retrying_delay, position, total_items):
    async with semaphore:
        chat_template = template["chat_template"]
        stop_words = template["stop_words"]
        for attempt in range(max_retries):
            try:
                if api_mode == "chat_completion":
                    messages = [{"role": "user", "content": item["prompt"]}]
                    if sys_prompt:
                        messages.insert(0, {"role": "system", "content": sys_prompt})
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=timeout,
                    )
                    output = response.choices[0].message.content
                elif api_mode == "completion":
                    response = await client.completions.create(
                        model=model_name,
                        prompt=chat_template.format(input=item["prompt"]),  # Use templated prompt
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=timeout,
                        stop=stop_words if stop_words else None,
                    )
                    output = response.choices[0].text
                else:
                    raise ValueError("Invalid API mode")
                break
            except Exception as e:
                print(f"Retries remaining: {max_retries - attempt - 1}. Error occurred while processing {bootcamp.__name__}:{item['id']}. {e}")
                if attempt == max_retries - 1:
                    await update_progress(position, os.path.basename(item['file_path']).replace(".jsonl", ""), total_items)
                    raise RuntimeError(f"Failed to process {item['id']} after {max_retries} attempts.")
                await asyncio.sleep(min((attempt + 1) ^ 2, max_retrying_delay))
                
        score = bootcamp.verify_score(output, item["ground_truth"], short_penalty=False, format_penalty=False)
        try:
            extracted = bootcamp.extract_output(output)
        except:
            extracted = None
        output_len = response.usage.completion_tokens if 'usage' in response else len(output.split())
        result = {
            "id": item["id"],
            "prompt": item["prompt"],
            "output_len": output_len,
            "score": score,
            "extracted_output": extracted,
            "ground_truth": item["ground_truth"],
            "output": output,
        }
        
        # Save results immediately
        detail_file = os.path.join(output_dir, "details", os.path.basename(item['file_path']))
        os.makedirs(os.path.dirname(detail_file), exist_ok=True)
        async with asyncio.Lock():
            with open(detail_file, 'a') as f:
                try:
                    json.dump(result, f, ensure_ascii=False)
                except Exception as e:
                    print(f"Error in saving details for {bootcamp.__name__} with result: {result}, which is {e}")
                f.write('\n')

        # Update progress bar with time statistics
        await update_progress(position, os.path.basename(item['file_path']).replace(".jsonl", ""), total_items)
        return result

async def evaluate_dataset(file_path, bootcamp, output_dir, template, semaphore, api_mode, sys_prompt, max_tokens, temperature, timeout, position, url, model_name, max_retries, max_retrying_delay,total_file_num,api_key):
    global progress_file_path
    # Load data
    with jsonlines.open(file_path) as reader:
        data = list(reader)
    async with openai.AsyncOpenAI(base_url=url, api_key=api_key) as client:
        tasks = []
        for idx, row in enumerate(data):
            item = {
                "id": idx,
                "file_path": file_path,
                "prompt": row["prompt"],
                "ground_truth": row["ground_truth"],
                "data_source": row["data_source"]
            }
            task = process_item(client, item, bootcamp, template, output_dir, semaphore, api_mode, sys_prompt, max_tokens, temperature, timeout, model_name, max_retries, max_retrying_delay, position, len(data))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
    avg_score = sum(r['score'] for r in results) / len(results)
    avg_len = sum(r['output_len'] for r in results) / len(results)
    meta_info = {
        "bootcamp": bootcamp.__name__,
        "avg_score": avg_score,
        "avg_len": avg_len
    }
    # update main progress bar
    await update_progress(position=1, description="Main Progress",total=total_file_num, init_model=False)

    # save meta info to file
    meta_info_output_file = os.path.join(output_dir, "meta.jsonl")
    os.makedirs(os.path.dirname(meta_info_output_file), exist_ok=True)
    with jsonlines.open(meta_info_output_file, mode='a') as writer:
        writer.write(meta_info)
        
    return bootcamp.__name__, avg_score, avg_len


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://{ip}:{port}/v1',
                        help='Base URL of the OpenAI API compatible service. Default format is http://{ip}:{port}/v1.')
    parser.add_argument('--api_key', default='EMPTY',
                        help='API key for accessing the model service. Set to "EMPTY" if no key is required.')
    parser.add_argument('--model_name', required=True,
                        help='Name of the model to be evaluated, e.g., r1_32B or other custom model name.')
    parser.add_argument('--test_dir', required=True,
                        help='Path to the directory containing test JSONL files for evaluation.')
    parser.add_argument('--max_concurrent_requests', type=int, default=128,
                        help='Maximum number of concurrent requests allowed globally.')
    parser.add_argument('--template', default='chatml',choices=['r1', 'qwen', 'internthinker', 'chatml'],
                        help='Predefined conversation template used to format prompts. Only valid when api_mode is completion.')
    parser.add_argument('--max_tokens', type=int, default=32768,
                        help='Maximum number of tokens the model can generate.')
    parser.add_argument('--temperature', type=float, default=0,
                        help='Controls randomness in text generation. Lower values produce more deterministic outputs.')
    parser.add_argument('--timeout', type=int, default=6000,
                        help='Request timeout in milliseconds.')
    parser.add_argument('--api_mode', default='completion',choices=['completion', 'chat_completion'],
                        help='API mode to use: "completion" for raw text generation or "chat_completion" for chat-style APIs.')
    parser.add_argument('--sys_prompt', type=str,
                        help='System prompt content used in chat_completion mode. If not provided, uses the default from the template (if any).')
    parser.add_argument('--max_retries', type=int, default=16,
                        help='Maximum number of retries for failed requests.')
    parser.add_argument('--max_retrying_delay', type=int, default=60,
                        help='Maximum delay between retries in seconds (using exponential backoff).')
    args = parser.parse_args()
    
    # Check if the model URL is alive before starting evaluation
    await check_model_url_alive(args.url,api_key=args.api_key,model_name=args.model_name, max_attempts=60,  interval=60)
  
    cur_file_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = f"{cur_file_dir}/output/{args.model_name}_{os.path.basename(args.test_dir)}_{formatted_time()}"
    os.makedirs(os.path.join(output_dir, 'details'), exist_ok=True)
    
    # save args to output_dir
    args_dict = vars(args)
    args_output_file = os.path.join(output_dir, "eval_args.json")
    with open(args_output_file, 'w') as f:
        json.dump(args_dict, f, indent=4)

    # Set progress log file path
    global progress_file_path
    progress_file_path = os.path.join(output_dir, "progress.log")
    open(progress_file_path, 'w').close()  # Clear file content

    

    # Notify the user to check detailed outputs and progress
    print(f"\nEvaluating model {args.model_name}. Please check the progress at {progress_file_path}. \nDetailed outputs will be saved in {os.path.join(output_dir, 'details')}.")

    # Collect all tasks
    tasks = []
    global_semaphore = asyncio.Semaphore(args.max_concurrent_requests)
    position = 2
    test_files = os.listdir(args.test_dir)
    total_file_num = len(test_files)
    for file_name in test_files:
        if not file_name.endswith('.jsonl'):
            print(f"Skipping non-JSONL file: {file_name}")
            continue
        file_path = os.path.join(args.test_dir, file_name)
        with jsonlines.open(file_path) as f:
            list_f = list(f)
            cur_file_num = len(list_f)
            data_source = list_f[0]['data_source']
        bootcamp_class = globals().get(f"{data_source}bootcamp")
        if not bootcamp_class:
            print(f"bootcamp class not found: {data_source}bootcamp")
            continue
        # Assign a fixed position for each dataset
        task = evaluate_dataset(
            file_path=file_path,
            bootcamp=bootcamp_class,
            output_dir=output_dir,
            template=TEMPLATE_MAP[args.template],
            semaphore=global_semaphore,
            api_mode=args.api_mode,
            sys_prompt=args.sys_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            position=position,
            url=args.url,
            api_key=args.api_key,
            model_name=args.model_name,
            max_retries=args.max_retries,
            max_retrying_delay=args.max_retrying_delay,
            total_file_num=total_file_num
        )
        # Init progress bar
        await update_progress(position=position, description=os.path.basename(file_path).replace(".jsonl", ""), total=cur_file_num, init_model=True)
        tasks.append(task)
        position += 1
    # Init total progress bar
    await update_progress(position=1, description="Main Progress",total=total_file_num, init_model=True)

    
    # Execute all tasks
    results = await asyncio.gather(*tasks)
    results = sorted(results, key=lambda x: x[0])
    
    # Save results
    df = pd.DataFrame(results, columns=["bootcamp", "Average Score", "Average Output Length"])
    df.loc[len(df)] = ["Total Average", df["Average Score"].mean(), df["Average Output Length"].mean()]
    df.to_excel(os.path.join(output_dir, f"{args.model_name}_scores.xlsx"), index=False)

if __name__ == "__main__":
    asyncio.run(main())