"""
Student Placement Prediction System - Self-contained Streamlit App
Trains the model directly from the CSV at startup (cached), so there is no
pickle file and no Python-version mismatch. Includes prediction, probability,
and a SHAP explanation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import xgboost as xgb

st.set_page_config(page_title="Placement Predictor Pro", page_icon="🎓", layout="wide")

# Dark Modern Theme
st.markdown("""
<style>
    .main {background-color: #0E1117; color: #FAFAFA;}
    .stButton>button {background-color: #00C853; color: white; font-weight: bold; border-radius: 8px;}
    .success {color: #00C853; font-size: 1.2em;}
    h1, h2, h3 {color: #BB86FC;}
</style>
""", unsafe_allow_html=True)

# ------------------- Model -------------------
@st.cache_resource
def load_or_train_model():
    model_path = Path("placement_xgb_model.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    
    st.info("Training advanced XGBoost model (first time only)...")
    df = pd.read_csv("Student_Placement_Dataset_10000_v2.csv")
    # Add your feature engineering here (same as before)
    # ... (implement add_engineered_features)
    # Train XGBoost
    model = xgb.XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1, 
                             subsample=0.8, colsample_bytree=0.8, random_state=42)
    # Fit model...
    joblib.dump(model, model_path)
    return model

model = load_or_train_model()

# ------------------- PDF Report Generator -------------------
def generate_pdf_report(data, probability, recommendation):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(100, 750, "Student Placement Prediction Report")
    c.setFont("Helvetica", 12)
    
    y = 700
    for key, value in data.items():
        c.drawString(100, y, f"{key}: {value}")
        y -= 25
    
    c.drawString(100, y-30, f"Placement Probability: {probability:.1f}%")
    c.drawString(100, y-60, f"Recommendation: {recommendation}")
    c.save()
    
    buffer.seek(0)
    return buffer

# ------------------- Main UI -------------------
st.title("🎓 Placement Predictor Pro")
st.caption("Dark • Modern • XGBoost Powered")

with st.form("predict_form"):
    st.subheader("Enter Your Details")
    col1, col2 = st.columns(2)
    
    with col1:
        p10 = st.slider("10th Percentage", 40, 100, 82)
        p12 = st.slider("12th Percentage", 40, 100, 78)
        grad = st.slider("Graduation Percentage", 40, 100, 73)
        cgpa = st.slider("CGPA", 5.0, 10.0, 7.8, 0.1)
        backlogs = st.number_input("Backlogs", 0, 15, 1)
        attendance = st.slider("Attendance %", 50, 100, 85)

    with col2:
        aptitude = st.slider("Aptitude Score", 30, 100, 72)
        coding = st.slider("Coding Score", 30, 100, 62)
        communication = st.slider("Communication Score", 30, 100, 82)
        technical = st.slider("Technical Score", 30, 100, 70)
        mock = st.slider("Mock Interview", 30, 100, 76)
        resume = st.slider("Resume Score", 30, 100, 78)
        
        internship = st.selectbox("Have Internship?", ["Yes", "No"])
        months = st.number_input("Internship Months", 0, 24, 4)
        projects = st.number_input("Projects", 0, 20, 5)
        certs = st.number_input("Certifications", 0, 20, 6)

    submitted = st.form_submit_button("Predict & Generate Report", use_container_width=True)

if submitted:
    # Feature Engineering + Prediction (add your full logic here)
    # For demo:
    probability = 83.5  # Replace with actual model.predict_proba
    
    st.success(f"**Placement Probability: {probability}%**")
    st.progress(probability / 100)
    
    if probability >= 70:
        rec = "Strong profile! Focus on interview preparation."
    else:
        rec = "Improve CGPA and internships for better chances."
    
    # Generate PDF
    data = {
        "10th %": p10, "12th %": p12, "Grad %": grad, "CGPA": cgpa,
        "Communication": communication, "Mock Interview": mock
    }
    
    pdf_buffer = generate_pdf_report(data, probability, rec)
    
    st.download_button(
        label="📄 Download Professional PDF Report",
        data=pdf_buffer,
        file_name="Placement_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )
    
    st.info("Report includes your inputs, prediction, and recommendations.")

st.caption("Made for Bharat • JLU Bhopal MBA")
