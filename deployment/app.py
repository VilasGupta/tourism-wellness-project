import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# CONFIG - set your model repo id here
HF_MODEL_REPO = "Vilas97Gupta/wellness-model"
MODEL_FILENAME = "best_model.joblib"

@st.cache_data(show_spinner=False)
def download_model():
    # This will download & cache the model file from HF model repo
    path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=MODEL_FILENAME)
    model = joblib.load(path)
    return model

model = download_model()

st.title("VisitWithUs — Wellness Package Predictor")

st.markdown("Enter customer details below and click **Predict**.")

# --- Build inputs (choose fields present in your training data) ---
# Use the raw feature names used during training. If you used one-hot encoding,
# the app should construct the same columns; simpler approach: collect raw inputs,
# then transform exactly as in training. For simplicity, we collect a subset of features here.

age = st.number_input("Age", min_value=18, max_value=100, value=30)
city_tier = st.selectbox("City Tier", options=["1", "2", "3"])
type_of_contact = st.selectbox("Type of Contact", options=["Company Invited", "Self Inquiry"])
gender = st.selectbox("Gender", options=["Male", "Female"])
num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
num_followups = st.number_input("Number of Followups", min_value=0, max_value=20, value=1)
duration = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=60.0, value=5.0)
monthly_income = st.number_input("Monthly Income", min_value=0.0, max_value=1000000.0, value=50000.0)
passport = st.selectbox("Passport", options=[0,1], index=1)
own_car = st.selectbox("Own Car", options=[0,1], index=0)
preferred_star = st.selectbox("Preferred Property Star", options=[1,2,3,4,5], index=3)
pitch_score = st.slider("Pitch Satisfaction Score", min_value=0, max_value=10, value=7)

# Build a dataframe of inputs — keys must match training feature names / preprocessing expectations
input_dict = {
    "Age": [age],
    "CityTier": [int(city_tier)],
    "TypeofContact": [type_of_contact],
    "Gender": [gender],
    "NumberOfPersonVisiting": [num_persons],
    "NumberOfFollowups": [num_followups],
    "DurationOfPitch": [duration],
    "MonthlyIncome": [monthly_income],
    "Passport": [passport],
    "OwnCar": [own_car],
    "PreferredPropertyStar": [preferred_star],
    "PitchSatisfactionScore": [pitch_score]
}

df_input = pd.DataFrame(input_dict)

label_maps = {
    "TypeofContact": {"Company Invited": 1, "Self Inquiry": 0},
    "Gender": {"Male": 1, "Female": 0}
}

# Apply mappings
for col, mapping in label_maps.items():
    if col in df_input.columns:
        df_input[col] = df_input[col].map(mapping)

# Ensure numeric columns are numeric (coerce objects -> NaN), then fill NaNs with 0
for c in df_input.columns:
    if df_input[c].dtype == "object":
        df_input[c] = pd.to_numeric(df_input[c], errors="coerce")

df_input = df_input.fillna(0)

st.subheader("Input preview")
st.dataframe(df_input)

if st.button("Predict"):
    # NOTE: The model was trained on preprocessed features (one-hot). If you encoded features earlier,
    # you must apply the same preprocessing pipeline here. For quick demo we'll try to predict directly,
    # assuming your saved model expects the same raw columns or you embedded preprocessing inside the model.
    try:
        pred = model.predict(df_input)
        proba = model.predict_proba(df_input)[:,1] if hasattr(model, "predict_proba") else None
        st.success(f"Predicted label: {int(pred[0])}")
        if proba is not None:
            st.info(f"Predicted probability of purchase: {float(proba[0]):.3f}")
    except Exception as e:
        st.error("Prediction failed. Ensure the model expects the same input features as provided. Error: " + str(e))
