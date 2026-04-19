import streamlit as st
import pandas as pd
import joblib
import os

# 1. Page Config
st.set_page_config(page_title="Sandal-ML", page_icon="🌳")
st.title("🌳 Sandal-ML: Yield Predictor")

# 2. FORCE LOAD (With error checking)
@st.cache_resource
def load_assets():
    try:
        # We use absolute paths to ensure it finds the files in your folder
        model = joblib.load('sandalwood_model.pkl')
        scaler = joblib.load('sandalwood_scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Critical Error: {e}")
        return None, None

model, scaler = load_assets()

# 3. Sidebar/Inputs
age = st.slider("Tree Age (Years)", 5, 25, 12)
rainfall = st.number_input("Annual Rainfall (mm)", 500, 2500, 1000)
temp = st.slider("Avg Temperature (°C)", 20, 40, 28)
ph = st.slider("Soil pH", 5.0, 9.0, 7.0)
altitude = st.number_input("Altitude (meters)", 300, 1500, 600)

# 4. Prediction Logic
if st.button("Predict Yield"):
    if model is not None and scaler is not None:
        # Prepare data
        features = pd.DataFrame([[age, rainfall, temp, ph, altitude]], 
                                columns=['Age_Years', 'Rainfall_mm', 'Avg_Temp_C', 'Soil_pH', 'Altitude_m'])
        
        # Scale and Predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        
        st.success(f"Estimated Heartwood Yield: {prediction[0]:.2f} Kg")
    else:
        st.error("The model or scaler failed to load. Please check your folder.")