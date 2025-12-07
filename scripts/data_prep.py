import os
import pandas as pd
from datasets import load_dataset
from huggingface_hub import upload_file

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
if not HF_USERNAME:
    # fallback: try to infer from token (optional)
    HF_USERNAME = os.getenv("GITHUB_ACTOR")  # not always ideal

DATASET_REPO = f"{HF_USERNAME}/tourism-wellness-dataset"
print("Loading dataset from HF dataset repo:", DATASET_REPO)

# Load via datasets library (works in GitHub Actions)
ds = load_dataset(DATASET_REPO)
# choose the split name that exists — earlier we used 'raw'
if "raw" in ds:
    df = ds["raw"].to_pandas()
else:
    # if dataset was pushed without named split, take the first split
    first_split = list(ds.keys())[0]
    df = ds[first_split].to_pandas()

print("Loaded rows:", len(df))

# Basic cleaning
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Ensure target exists
if "ProdTaken" not in df.columns:
    raise ValueError("Target column 'ProdTaken' not found in dataset")

# Train-test split (stratify if possible)
from sklearn.model_selection import train_test_split
if df["ProdTaken"].nunique() > 1:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["ProdTaken"])
else:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save locally
train_path = "train.parquet"
test_path = "test.parquet"
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
print("Saved train/test parquet:", train_path, test_path)

# Upload splits back to HF dataset repo
print("Uploading train/test splits to HF dataset repo...")
upload_file(
    path_or_fileobj=train_path,
    path_in_repo="train.parquet",
    repo_id=DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN
)
upload_file(
    path_or_fileobj=test_path,
    path_in_repo="test.parquet",
    repo_id=DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN
)
print("Upload complete.")
