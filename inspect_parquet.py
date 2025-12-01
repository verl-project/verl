import pandas as pd
import sys

file_path = '/home/4dtc/rio/verl/data/ui_design_parquet/train.parquet'

try:
    with open('inspect_output_train.txt', 'w') as f:
        sys.stdout = f
        df = pd.read_parquet(file_path)
        print(f"Successfully read {file_path}")
        print(f"Shape: {df.shape}")
        print("\nColumns:")
        print(df.columns.tolist())
        
        print("\nFirst row example:")
        first_row = df.iloc[0]
        for col in df.columns:
            print(f"\n--- {col} ---")
            print(first_row[col])
        
except Exception as e:
    with open('inspect_output_train.txt', 'w') as f:
        f.write(f"Error reading parquet file: {e}")
