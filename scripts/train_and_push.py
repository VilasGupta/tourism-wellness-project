# scripts/train_and_push.py (fixed upload_file keyword-args)
import os
import joblib
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo, upload_file
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME") or os.getenv("GITHUB_ACTOR")
DATASET_REPO = f"{HF_USERNAME}/tourism-wellness-dataset"
MODEL_REPO = f"{HF_USERNAME}/wellness-model"

print("Loading HF train/test...")
ds = load_dataset(DATASET_REPO)
# pick train/test splits if present, otherwise fallback
if "train" in ds and "test" in ds:
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()
else:
    # if only raw split exists, try to find parquet files
    first_split = list(ds.keys())[0]
    train_df = ds[first_split].to_pandas()
    # best-effort: create a test split if missing
    if "test" not in ds:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df.get("ProdTaken"))
    else:
        test_df = ds["test"].to_pandas()

TARGET = "ProdTaken"
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET].astype(int)
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET].astype(int)

print("Training model...")

model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
params = {"n_estimators":[50,100], "max_depth":[3,5], "learning_rate":[0.05,0.1]}

search = RandomizedSearchCV(model, params, n_iter=3, cv=2, scoring="f1", random_state=42)
search.fit(X_train, y_train)
best = search.best_estimator_

preds = best.predict(X_test)
f1 = float(f1_score(y_test, preds))
print("F1 Score:", f1)

os.makedirs("artifacts", exist_ok=True)
model_path = "artifacts/best_model.joblib"
joblib.dump(best, model_path)

api = HfApi()
create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True, token=HF_TOKEN)

# Corrected upload_file usage with keyword arguments
upload_file(
    path_or_fileobj=model_path,
    path_in_repo="best_model.joblib",
    repo_id=MODEL_REPO,
    repo_type="model",
    token=HF_TOKEN
)

print("Model uploaded to:", MODEL_REPO)
