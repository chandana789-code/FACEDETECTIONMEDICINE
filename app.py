import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Medication Reaction Alert", layout="centered")
st.title("💊 Medication Reaction Alert System")
st.caption("⚠️ Early risk indication only. Not a medical diagnosis.")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("reaction_model.pkl")

# -----------------------------
# Input Section
# -----------------------------
st.subheader("Enter Facial Feature Changes")

lip_ratio = st.slider("Lip swelling ratio", 0.9, 2.0, 1.0, 0.01)
redness = st.slider("Skin redness level", 0.2, 1.0, 0.3, 0.01)
fatigue = st.slider("Eye fatigue level", 0.4, 1.0, 0.8, 0.01)
face_distance = st.slider("Face similarity distance", 0.02, 0.25, 0.05, 0.01)

st.subheader("Symptoms")
itching = st.checkbox("Itching")
nausea = st.checkbox("Nausea")
dizziness = st.checkbox("Dizziness")
breathing = st.checkbox("Breathing Difficulty")

symptom_score = sum([itching, nausea, dizziness, breathing]) * 0.25

# -----------------------------
# Prediction
# -----------------------------
if st.button("Check Reaction Risk"):
    features = [[
        lip_ratio,
        redness,
        fatigue,
        symptom_score,
        face_distance
    ]]

    prediction = model.predict(features)[0]

    if prediction == 0:
        st.success("✅ No medication reaction detected.")
    elif prediction == 1:
        st.warning("⚠️ Mild reaction detected. Monitor symptoms.")
    else:
        st.error("🚨 Severe reaction suspected. Consult a doctor immediately.")
