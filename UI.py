import joblib
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load("24_model.pkl")
scaler = joblib.load("scaling.pkl")  # We need to save and load the same scaler used during training

st.title("Diabetes Prediction System")
st.write("Enter the details below to predict diabetes:")

# Input fields
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
Glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=80)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
Insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=30)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
Age = st.number_input("Age", min_value=0, max_value=120, value=30)

if st.button("Predict"):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    # Apply scaling (fixes incorrect predictions)
    input_data_scaled = scaler.transform(input_data)  # Scale input before prediction

    probability = model.predict_proba(input_data_scaled)[0][1]
    threshold = 0.6
    prediction = 1 if probability >= threshold else 0

    if prediction == 1:
        st.error("The model predicts: *Diabetic*  ")
    else:
        st.success("The model predicts: *Not Diabetic* ")