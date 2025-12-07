import os
import joblib
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo, upload_file
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
DATASET_REPO = f"{HF_USERNAME}/tourism-wellness-dataset"
MODEL_REPO = f"{HF_USERNAME}/wellness-model"

print("Loading HF train/test...")
ds = load_dataset(DATASET_REPO)
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()

TARGET = "ProdTaken"
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
params = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1]
}

search = RandomizedSearchCV(model, params, n_iter=3, cv=2, scoring="f1", random_state=42)
search.fit(X_train, y_train)
best = search.best_estimator_

preds = best.predict(X_test)
f1 = f1_score(y_test, preds)
print("F1 Score:", f1)

os.makedirs("artifacts", exist_ok=True)
joblib.dump(best, "artifacts/best_model.joblib")

api = HfApi()
create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True, token=HF_TOKEN)
upload_file("artifacts/best_model.joblib", "best_model.joblib", MODEL_REPO, "model", token=HF_TOKEN)

print("Model uploaded to:", MODEL_REPO)
