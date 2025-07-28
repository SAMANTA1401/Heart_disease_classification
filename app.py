import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np
from src.utils import feature_engineering, load_object
# Load saved model
from src.path_config import DataTransformationConfig
from src.path_config import ModelTrainerConfig
from src.exception import CustomException
from src.logger import logging
import sys


st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("🫀 Heart Disease Risk Predictor")

st.markdown("Enter patient data below to predict heart disease risk.")

# User input form
with st.form("prediction_form"):
    Age = st.slider("Age", 20, 100, 50)
    Sex = st.selectbox("Sex", ["M", "F"])
    ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    RestingBP = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    Cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=600, value=200)
    FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    MaxHR = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    ExerciseAngina = st.selectbox("Exercise-induced Angina", ["Y", "N"])
    Oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("Predict")

if submitted:
    user_input = pd.DataFrame([{
        "Age": Age,
        "Sex": Sex,
        "ChestPainType": ChestPainType,
        "RestingBP": RestingBP,
        "Cholesterol": Cholesterol,
        "FastingBS": FastingBS,
        "RestingECG": RestingECG,
        "MaxHR": MaxHR,
        "ExerciseAngina": ExerciseAngina,
        "Oldpeak": Oldpeak,
        "ST_Slope": ST_Slope,
    }])

    try:

        # Apply feature engineering
        feature_engineering(user_input)


        # print(user_input)
        # logging.info(user_input)

        # Load model and predict
        preprocessobj = load_object(DataTransformationConfig().preprocessor_obj_file_path)

        processed_input = preprocessobj.transform(user_input)

        model = joblib.load(ModelTrainerConfig.trained_model_file_path)

        prediction = model.predict(processed_input)[0]
        proba = model.predict_proba(processed_input)[0][1]

    except Exception as e:
        raise logging.info(CustomException(e,sys))

    st.subheader("🔍 Prediction Result")
    st.write("**Risk of Heart Disease:**", "Yes" if prediction else "No")
    st.write(f"**Probability:** {proba:.2%}")

# # Optional training endpoint (if you want in-app training)
# st.markdown("---")
# st.subheader("⚙️ Train Model (Admin)")
# if st.button("Train Model"):
#     st.warning("Training logic should be securely implemented.")
#     st.info("Add training function here to re-train and save model.")
