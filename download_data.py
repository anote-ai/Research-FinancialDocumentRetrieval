from datasets import load_dataset
import pandas as pd

dataset = load_dataset("PatronusAI/financebench", split="train")
df = dataset.to_pandas()
df.to_csv("financebench_sample.csv", index=False)
print(f"Downloaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
print(df.head(2))
