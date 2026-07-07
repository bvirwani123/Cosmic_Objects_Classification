import streamlit as st
import numpy as np
import joblib
import os

# Set layout configurations
st.set_page_config(layout="centered", initial_sidebar_state="collapsed")

# --- BACKGROUND SYSTEM LOGIC (SILENT CHECK & LOAD) ---
model_filename = "cosmic_objects_rf_pipeline.pkl"

if os.path.exists(model_filename):
    file_size_mb = os.path.getsize(model_filename) / (1024 * 1024)
    # Halt with error if it detects a broken Git LFS pointer text file
    if file_size_mb < 0.01:
        st.error("🚨 BUG DETECTED: Streamlit is reading a broken Git LFS pointer text file instead of the actual model file.")
        st.stop()
else:
    st.error(f"❌ ERROR: File `{model_filename}` cannot be found in this directory.")
    st.stop()

try:
    artifacts = joblib.load(model_filename)
    best_pipeline = artifacts["pipeline"]
    scaler = artifacts["scaler"]
    label_encoder = artifacts["label_encoder"]
except Exception as e:
    st.error(f"💥 LOADING FAILED: {e}")
    st.stop()

# --- WEB DASHBOARD INTERFACE LAYOUT ---
st.title("🌌 Cosmic-Object Classifier Dashboard")
st.write("Enter the photometric and coordinate details below to determine the celestial category.")
st.markdown("---")

st.subheader("Astronomical Feature Inputs")

# Numeric inputs pre-seeded with the exact coordinates/values from your screenshots
alpha = st.number_input("Alpha (Right Ascension)", value=340.99, format="%.2f")
delta = st.number_input("Delta (Declination)", value=20.59, format="%.2f")
u = st.number_input("u (Ultraviolet filter band)", value=23.48, format="%.2f")
g = st.number_input("g (Green filter band)", value=23.34, format="%.2f")
r = st.number_input("r (Red filter band)", value=21.32, format="%.2f")
i = st.number_input("i (Near Infrared filter band)", value=20.25, format="%.2f")
z = st.number_input("z (Infrared filter band)", value=19.54, format="%.2f")
redshift = st.number_input("Redshift (z)", value=1.42, format="%.2f")

st.markdown("---")

# Prediction handling block
if st.button("Classify Celestial Object", type="primary"):
    # Convert active values to 2D numpy structure
    raw_input = np.array([[alpha, delta, u, g, r, i, z, redshift]])
    
    # Standardize data using the model's loaded scaler parameters
    scaled_input = scaler.transform(raw_input)
    
    # Predict encoded target numeric value
    predicted_encoded = best_pipeline.predict(scaled_input)[0]
    
    # Inverse map index label back to original classification string
    original_class_name = label_encoder.inverse_transform([predicted_encoded])[0]
    
    # Output success message container matching target layout
    st.success(f"✨ The predicted celestial identity is: **{original_class_name}**")