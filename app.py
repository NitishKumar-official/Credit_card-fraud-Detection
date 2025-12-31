import streamlit as st
import pickle
import numpy as np

# Load model
with open("fraud_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("💳 Credit Card Fraud Detection App")

st.write("Enter transaction details (V1 to V28):")

input_features = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0)
    input_features.append(val)

threshold = st.slider("Select Fraud Threshold", 0.1, 0.9, 0.3)

if st.button("Check Transaction"):
    data = np.array(input_features).reshape(1, -1)
    prob = model.predict_proba(data)[0][1]

    if prob >= threshold:
        st.error(f"⚠️ Fraud Transaction (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Normal Transaction (Probability: {prob:.2f})")
