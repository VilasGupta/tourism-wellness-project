import os
import pandas as pd
from huggingface_hub import upload_file
from sklearn.model_selection import train_test_split

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
DATASET_REPO = f"{HF_USERNAME}/tourism-wellness-dataset"

print("Loading raw.parquet...")
df = pd.read_parquet("raw.parquet")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["ProdTaken"])

train_df.to_parquet("train.parquet", index=False)
test_df.to_parquet("test.parquet", index=False)

upload_file("train.parquet", "train.parquet", DATASET_REPO, "dataset", token=HF_TOKEN)
upload_file("test.parquet", "test.parquet", DATASET_REPO, "dataset", token=HF_TOKEN)

print("Train/Test uploaded.")
