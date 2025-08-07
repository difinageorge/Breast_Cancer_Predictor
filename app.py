import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# Load model
model = tf.keras.models.load_model("model.h5")

# Load dataset (to get feature names and use for scaling)
data = load_breast_cancer()
feature_names = data.feature_names

# Initialize scaler and fit it with full data (for consistent scaling)
scaler = StandardScaler()
scaler.fit(data.data)

# Streamlit UI
st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")
st.title("🔬 Breast Cancer Prediction App")
st.markdown("Enter the required medical measurements to predict whether the tumor is **Malignant** or **Benign**.")

# Input fields for all features
user_input = []
cols = st.columns(3)  # Split into 3 columns

for i, feature in enumerate(feature_names):
    val = cols[i % 3].number_input(f"{feature}", value=float(data.data[:, i].mean()), step=0.1)
    user_input.append(val)

# Predict button
if st.button("🧠 Predict"):
    # Preprocess input
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # Make prediction
    prediction = model.predict(input_scaled)[0][0]
    result = "Benign 😊" if prediction > 0.5 else "Malignant ⚠️"

    # Show result
    st.subheader("🩺 Prediction Result:")
    st.success(f"The tumor is likely: **{result}**")

    # Add positive, calming quote
    st.markdown("---")
    st.markdown("💬 *“You are stronger than you think, and you have already survived everything you thought you couldn't.”*")
    st.markdown("❤️ *Stay hopeful. This prediction is just one step – not the full journey.*")
