import os
from datasets import Dataset
import pandas as pd
from huggingface_hub import HfApi, create_repo, upload_file

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
DATASET_REPO = f"{HF_USERNAME}/tourism-wellness-dataset"

print("Registering dataset to:", DATASET_REPO)

df = pd.read_csv("data/tourism.csv")

api = HfApi()
create_repo(repo_id=DATASET_REPO, repo_type="dataset", exist_ok=True, token=HF_TOKEN)

df.to_parquet("raw.parquet", index=False)

upload_file(
    path_or_fileobj="raw.parquet",
    path_in_repo="raw.parquet",
    repo_id=DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN
)

print("Dataset uploaded successfully.")
