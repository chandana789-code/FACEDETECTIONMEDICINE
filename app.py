import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Early Medication Reaction Alert", layout="centered")

st.title("Early Medication Reaction Risk Alert System")
st.caption("This system provides early risk indication only and does not diagnose medical conditions.")

# Load trained model
model = joblib.load("reaction_model.pkl")

st.subheader("Facial Feature Assessment")

lip_ratio = st.slider("Lip swelling ratio", 0.9, 2.0, 1.0, 0.01)
redness = st.slider("Skin redness level", 0.2, 1.0, 0.3, 0.01)
fatigue = st.slider("Eye fatigue level", 0.4, 1.0, 0.8, 0.01)
face_distance = st.slider("Face similarity score", 0.02, 0.25, 0.05, 0.01)

st.subheader("Reported Symptoms")
itching = st.checkbox("Itching")
nausea = st.checkbox("Nausea")
dizziness = st.checkbox("Dizziness")
breathing = st.checkbox("Breathing difficulty")

symptom_score = sum([itching, nausea, dizziness, breathing]) * 0.25

if st.button("Assess Risk"):
    features = [[
        lip_ratio,
        redness,
        fatigue,
        symptom_score,
        face_distance
    ]]

    prediction = model.predict(features)[0]

    st.subheader("System Recommendation")

    if prediction == 0:
        st.success("No immediate risk detected. Continue monitoring.")
    elif prediction == 1:
        st.warning("Mild risk detected. Please monitor symptoms and consider consulting a doctor.")
    else:
        st.error("High risk detected. It is strongly recommended to consult a healthcare professional immediately.")

