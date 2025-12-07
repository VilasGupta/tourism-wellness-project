import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# -----------------------------
# CONFIG
# -----------------------------
HF_MODEL_REPO = "Vilas97Gupta/tourism-project"
MODEL_FILENAME = "best_model.joblib"

# -----------------------------
# Download Model
# -----------------------------
@st.cache_data(show_spinner=False)
def download_model():
    path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=MODEL_FILENAME)
    model = joblib.load(path)
    return model

model = download_model()

st.title("VisitWithUs — Wellness Package Predictor")
st.markdown("Enter customer details below and click **Predict**.")

# -----------------------------
# USER INPUTS
# -----------------------------
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

# Raw input dict
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

st.subheader("Input Preview")
st.dataframe(df_input)

# ----------------------------------------------------
# EXPECTED MODEL FEATURES (from your error message)
# ----------------------------------------------------
EXPECTED_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "NumberOfTrips", "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
    "TypeofContact_Self Enquiry",
    "Occupation_Large Business", "Occupation_Salaried", "Occupation_Small Business",
    "Gender_Female", "Gender_Male",
    "ProductPitched_Deluxe", "ProductPitched_King", "ProductPitched_Standard", "ProductPitched_Super Deluxe",
    "MaritalStatus_Married", "MaritalStatus_Single", "MaritalStatus_Unmarried",
    "Designation_Executive", "Designation_Manager", "Designation_Senior Manager", "Designation_VP"
]

# ----------------------------------------------------
# BUILD MODEL INPUT (Fix mismatch)
# ----------------------------------------------------
def build_model_input(df_raw):
    # Start with all features = 0
    row = {f: 0 for f in EXPECTED_FEATURES}

    # Copy numeric features
    numeric_copy = [
        "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
        "NumberOfFollowups", "PreferredPropertyStar", "Passport",
        "OwnCar", "PitchSatisfactionScore", "MonthlyIncome"
    ]

    for col in numeric_copy:
        if col in df_raw.columns:
            row[col] = float(df_raw.at[0, col])

    # Type of Contact → one-hot
    if "TypeofContact" in df_raw.columns:
        val = df_raw.at[0, "TypeofContact"].strip().lower()
        if "self" in val:
            row["TypeofContact_Self Enquiry"] = 1

    # Gender → one-hot
    if "Gender" in df_raw.columns:
        g = df_raw.at[0, "Gender"].strip().lower()
        if g == "male":
            row["Gender_Male"] = 1
        elif g == "female":
            row["Gender_Female"] = 1

    # Missing fields (no UI): leave them as 0:
    # NumberOfTrips, NumberOfChildrenVisiting,
    # Occupation_*, Designation_*, ProductPitched_*, MaritalStatus_*

    # Build final DF in correct order
    return pd.DataFrame([row], columns=EXPECTED_FEATURES)

# ----------------------------------------------------
# PREDICT BUTTON
# ----------------------------------------------------
if st.button("Predict"):
    try:
        model_input = build_model_input(df_input)

        st.subheader("Model-ready Input (Final Features)")
        st.dataframe(model_input)

        pred = model.predict(model_input)
        proba = model.predict_proba(model_input)[:, 1] if hasattr(model, "predict_proba") else None

        st.success(f"Predicted Label: {int(pred[0])}")
        if proba is not None:
            st.info(f"Probability of Conversion: {proba[0]:.3f}")

    except Exception as e:
        st.error("Prediction failed: " + str(e))
