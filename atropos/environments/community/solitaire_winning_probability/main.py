import pandas as pd

splits = {
    "train": "main/train-00000-of-00001.parquet",
    "test": "main/test-00000-of-00001.parquet",
}
df = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["train"])
df.to_csv("local_data.csv", index=False)
print(df.head())
