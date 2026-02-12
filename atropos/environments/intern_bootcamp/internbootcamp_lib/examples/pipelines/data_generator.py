import os
import fire
import json
import internbootcamp
import subprocess
from tqdm import tqdm 
import sys


sys.set_int_max_str_digits(128*1024)


def main_pipeline(
    bootcamp_name, n, save_file, 
    config_file=None, bootcamp_cls_name=None, shuffle=False,
    tokenizer=None, max_prompt_len=None
):
    assert bootcamp_cls_name is not None
    assert (tokenizer is not None) == (max_prompt_len is not None), "tokenizer and max_prompt_len should be either both existing or both being None."
    if tokenizer is not None:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        except Exception as e:
            print(f"Error occurred when loading tokenizer.{e}")
            tokenizer = None
            
    if config_file is None:
        print(f"config_file not provided. Will use default configs for {bootcamp_name}.")
        configs = [{}]
    else:
        with open(config_file, "r") as f:
            try:
                configs = json.load(f)
            except:
                import pdb;pdb.set_trace()
    
    if not configs:
        configs = [{}]
    n_per_config = [n // len(configs) for _ in configs]
    n_per_config[-1] = n - sum(n_per_config[:-1])
    
    if bootcamp_cls_name is None:
        assert False, "Deprecated: using bootcamp name to get default bootcamp_cls_name is no not supported" 
        # puzzle_name -> PuzzleNameV2bootcamp
        bootcamp_cls_name = "".join([s.capitalize() for s in bootcamp_name.split('_')]) + "V2bootcamp"
        print(f"bootcamp_cls_name is not provided. Set it as {bootcamp_cls_name} by default")

    while '//' in save_file:
        save_file = save_file.replace("//", '/')
    os.makedirs("/".join(save_file.split('/')[:-1]), exist_ok=True)

    bar = tqdm(total=n, desc=f"{bootcamp_cls_name} Gen...")

    writer = open(save_file, "w")
    for _n, config in zip(n_per_config, configs):
        # print(_n, config)
        bootcamp_cls = getattr(internbootcamp, bootcamp_cls_name)
        
        for key in list(config.keys()):
            try:
                if key not in bootcamp_cls.__init__.__code__.co_varnames:
                    del config[key]
            except Exception as e:
                
                print("bootcamp_name:", bootcamp_cls_name,"+", bootcamp_cls)
        count = 0
        failure = 0
        while count < _n:
            try:
                bootcamp = bootcamp_cls(**config)
                bootcamp_case = bootcamp.case_generator()
                prompt = bootcamp.prompt_func(bootcamp_case)
                if tokenizer is not None:
                    length = len(tokenizer.encode(prompt))
                    if length > max_prompt_len:
                        continue
                failure = 0
                writer.write(json.dumps({
                    "data_source": bootcamp_cls_name.replace("bootcamp", ""),
                    "prompt": prompt.strip(),
                    "ground_truth": bootcamp_case
                }, ensure_ascii=False) + "\n")
                bar.update()
                count += 1
            except Exception as e:
                failure += 1
                if failure > 1000:
                    print(config, f"seems to be a too challenging config to generate cases , because of {e}")
                continue
            
    writer.close()

    # shuffle
    if shuffle:
        subprocess.run(f"shuf {save_file} > {save_file}.tmp", shell=True)
        subprocess.run(f"mv {save_file}.tmp {save_file}", shell=True)



if __name__ == "__main__":
    """
    example usge:
    python examples/pipelines/v2_data_generator.py \
        --bootcamp_name aquarium \
        --n 2048 \
        --save_file ../data/bootcamp_generation/aquarium/train.jsonl \
        --config_file examples/pipelines/v2_data_configs/aquarium_train.jsonl \
        --bootcamp_cls_name AquariumV2bootcamp --shuffle
    """
    fire.Fire(main_pipeline)