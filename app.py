import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('model.pkl')

st.title("Medical Insurance Cost Predictor")

# Input fields
age = st.slider("Age", 18, 100, 25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
children = st.slider("Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    log_pred = model.predict(input_data)
    pred = np.expm1(log_pred)

    st.success(f"Predicted Insurance Cost: ${pred[0]:,.2f}")
