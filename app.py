import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from streamlit_lottie import st_lottie

# Load saved model, scaler, and expected columns
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

st.set_page_config(page_title="Heart Stroke Prediction", layout="centered")
st.title("💓 Heart Stroke Prediction by Ankit")
st.markdown("Provide the following details to check your heart stroke risk:")
from pathlib import Path

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return f.read()

current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()

lottie_heart = load_lottiefile((current_dir/"animation.json").as_posix())
st_lottie(lottie_heart, speed=1, reverse=False, loop=True, quality="low", height=200, width=200, key=None)


# Layout: Use columns for better UX
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40, help="Your age in years.")
    sex = st.selectbox("Sex", ["M", "F"], help="Your gender.")
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"], help="Type of chest pain experienced.")
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120, help="Your resting blood pressure.")
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200, help="Your cholesterol level.")

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], help="Whether your fasting blood sugar is greater than 120 mg/dL.")
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], help="Results of your resting electrocardiogram.")
    max_hr = st.slider("Max Heart Rate", 60, 220, 150, help="Your maximum heart rate.")
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"], help="Whether you experience angina during exercise.")
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, help="ST depression induced by exercise relative to rest.")
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], help="The slope of the peak exercise ST segment.")

# Advanced options/info
with st.expander("ℹ️ Advanced Options / Info"):
    st.markdown("""
    - **Chest Pain Types:**  
      - ATA: Atypical Angina  
      - NAP: Non-Anginal Pain  
      - TA: Typical Angina  
      - ASY: Asymptomatic  
    - **Resting ECG:**  
      - Normal, ST, LVH (Left Ventricular Hypertrophy)
    - **ST Slope:**  
      - Up, Flat, Down
    """)
    st.info("All inputs are required. Please consult a doctor for medical advice.")

# Input validation
if cholesterol < 120:
    st.warning("Cholesterol seems unusually low. Please check your value.", icon="⚠️")

# Show summary before prediction
with st.expander("🔍 Review Your Inputs"):
    st.write({
        "Age": age,
        "Sex": sex,
        "Chest Pain": chest_pain,
        "Resting BP": resting_bp,
        "Cholesterol": cholesterol,
        "Fasting BS": fasting_bs,
        "Resting ECG": resting_ecg,
        "Max HR": max_hr,
        "Exercise Angina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST Slope": st_slope
    })

# When Predict is clicked
if st.button("Predict"):
    with st.spinner("Analyzing..."):
        # Create a raw input dictionary
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        # Create input dataframe
        input_df = pd.DataFrame([raw_input])

        # Fill in missing columns with 0s
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns
        input_df = input_df[expected_columns]

        # Scale the input
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        # Try to get probability/confidence if available
        try:
            proba = model.predict_proba(scaled_input)[0][prediction]
        except Exception:
            proba = None

        # Result Visualization with Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = proba * 100 if proba is not None else 50,  # Use probability or default value
            delta = {'reference': 50},
            title = {'text': "<b>Heart Disease Risk</b>", 'align': "center"},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "midnightblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': 'green'},
                    {'range': [40, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'red'}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': proba * 100 if proba is not None else 50}}
        ))

        st.plotly_chart(fig, use_container_width=True)

        # Interpretation and Recommendation
        if prediction == 1:
            risk_level = "High"
            recommendation = "It is recommended to consult a healthcare professional for further evaluation."
        else:
            risk_level = "Low"
            recommendation = "Continue to maintain a healthy lifestyle and have regular check-ups."

        st.write(f"**Risk Level:** {risk_level}")
        st.write(f"**Recommendation:** {recommendation}")

        if proba is not None:
            st.write(f"**Confidence:** {proba:.2f}")

        # Optional: Display prediction (0 or 1)
        # st.write(f"**Prediction:** {prediction}")