import re, os
import datasets

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

data_source = "openai/gsm8k"
instruction_following = 'Let\'s think step by step and output the final answer after "####".'
dataset = datasets.load_dataset(data_source, "main")

train_dataset = dataset["train"]
test_dataset = dataset["test"]



# add a row to each data item that represents a unique id
def make_map_fn(split):
    def process_fn(example, idx):
        question_raw = example.pop("question")

        question = question_raw + " " + instruction_following

        answer_raw = example.pop("answer")
        solution = extract_solution(answer_raw)
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question,
            },
        }
        return data

    return process_fn

train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

train_dataset.to_parquet(os.path.join("local_parquet_dir", "train.parquet"))
test_dataset.to_parquet(os.path.join("local_parquet_dir", "test.parquet"))
