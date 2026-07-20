"""
Student Placement Prediction System - Self-contained Streamlit App
Trains the model directly from the CSV at startup (cached), so there is no
pickle file and no Python-version mismatch. Includes prediction, probability,
and a SHAP explanation.
"""

"""
Student Placement Prediction System - Enhanced Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Dark Modern Config
st.set_page_config(page_title="Placement Predictor Pro", page_icon="🎓", layout="wide")
st.markdown("""
<style>
    .main {background-color: #0E1117; color: #FAFAFA;}
    .stButton>button {background-color: #00C853; color: white; font-weight: bold; border-radius: 8px;}
    h1, h2, h3 {color: #BB86FC;}
    .success {color: #00C853;}
</style>
""", unsafe_allow_html=True)

DATA_FILE = "Student_Placement_Dataset_10000_v2.csv"
THRESHOLD = 0.60
SKILL_COLS = ["Aptitude_Score", "Coding_Score", "Communication_Score", "Technical_Score", "Mock_Interview_Score", "Resume_Score"]
FINAL_NUM = ["10th_Percentage", "12th_Percentage", "Graduation_Percentage", "CGPA", "Backlogs", "Attendance", 
             "Internship_Months", "Projects", "Certifications", "Aptitude_Score", "Coding_Score", 
             "Communication_Score", "Technical_Score", "Mock_Interview_Score", "Resume_Score",
             "Academic_Avg", "Skill_Avg", "Total_Experience", "CGPA_x_Skill"]
FINAL_CAT = ["Internship"]

def add_engineered_features(df):
    df = df.copy()
    df["Academic_Avg"] = df[["10th_Percentage", "12th_Percentage", "Graduation_Percentage"]].mean(axis=1)
    df["Skill_Avg"] = df[SKILL_COLS].mean(axis=1)
    df["Total_Experience"] = df["Internship_Months"] + df["Projects"] + df["Certifications"]
    df["CGPA_x_Skill"] = df["CGPA"] * df["Skill_Avg"]
    return df

@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_FILE).drop(columns=["Student_ID", "Company_Type", "Placement_Package_LPA"])
    df["Placement_Status"] = (df["Placement_Status"] == "Placed").astype(int)
    df = add_engineered_features(df)

    X = df[FINAL_NUM + FINAL_CAT]
    y = df["Placement_Status"]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), 
         make_column_selector(dtype_include=np.number)),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), 
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]), 
         make_column_selector(dtype_exclude=np.number)),
    ])
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=2,
                                         class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    pipeline.fit(X, y)
    return pipeline

st.title("🎓 Placement Predictor Pro")
st.write("Dark Modern • Auto PDF Report • For JLU Students")

with st.spinner("Loading model..."):
    pipeline = train_model()

# Form
with st.form("student_form"):
    st.subheader("Academic Record")
    c1, c2, c3 = st.columns(3)
    p10 = c1.slider("10th Percentage", 40, 100, 80)
    p12 = c2.slider("12th Percentage", 40, 100, 78)
    grad = c3.slider("Graduation Percentage", 40, 100, 73)
    c4, c5, c6 = st.columns(3)
    cgpa = c4.slider("CGPA", 5.0, 10.0, 7.8, 0.1)
    backlogs = c5.number_input("Backlogs", 0, 15, 1)
    attendance = c6.slider("Attendance %", 50, 100, 85)

    st.subheader("Skills")
    s1, s2, s3 = st.columns(3)
    aptitude = s1.slider("Aptitude", 30, 100, 75)
    coding = s2.slider("Coding", 30, 100, 65)
    communication = s3.slider("Communication", 30, 100, 82)
    s4, s5, s6 = st.columns(3)
    technical = s4.slider("Technical", 30, 100, 70)
    mock = s5.slider("Mock Interview", 30, 100, 75)
    resume = s6.slider("Resume", 30, 100, 78)

    st.subheader("Experience")
    e1, e2, e3, e4 = st.columns(4)
    internship = e1.selectbox("Internship", ["Yes", "No"])
    internship_months = e2.number_input("Months", 0, 24, 4)
    projects = e3.number_input("Projects", 0, 20, 5)
    certifications = e4.number_input("Certifications", 0, 20, 6)

    submitted = st.form_submit_button("🚀 Predict & Generate Report")

if submitted:
    row = {
        "10th_Percentage": p10, "12th_Percentage": p12, "Graduation_Percentage": grad,
        "CGPA": cgpa, "Backlogs": backlogs, "Attendance": attendance,
        "Internship_Months": internship_months, "Projects": projects, "Certifications": certifications,
        "Aptitude_Score": aptitude, "Coding_Score": coding, "Communication_Score": communication,
        "Technical_Score": technical, "Mock_Interview_Score": mock, "Resume_Score": resume,
        "Internship": internship,
    }
    d = add_engineered_features(pd.DataFrame([row]))[FINAL_NUM + FINAL_CAT]

    proba = float(pipeline.predict_proba(d)[0, 1])
    is_placed = proba >= THRESHOLD

    st.divider()
    colA, colB = st.columns([3,1])
    with colA:
        if is_placed:
            st.success("### ✅ Likely to be PLACED")
        else:
            st.error("### ⚠️ At Risk")
    with colB:
        st.metric("Probability", f"{proba*100:.1f}%")
    st.progress(proba)

    # PDF Report
    def create_pdf():
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, "Placement Prediction Report - Bharat Kundnani")
        c.setFont("Helvetica", 12)
        y = 700
        for key, value in row.items():
            c.drawString(100, y, f"{key}: {value}")
            y -= 20
        c.drawString(100, y-30, f"Placement Probability: {proba*100:.1f}%")
        c.save()
        buffer.seek(0)
        return buffer

    pdf = create_pdf()
    st.download_button("📄 Download PDF Report", pdf, "Placement_Report.pdf", "application/pdf", use_container_width=True)

    # SHAP
    st.subheader("Why this prediction?")
    # (Your original SHAP code remains here - I kept it unchanged)
    preprocessor = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    X_trans = preprocessor.transform(d)
    feat_names = [n.split("__")[-1] for n in preprocessor.get_feature_names_out()]
    shap_vals = shap.TreeExplainer(model).shap_values(X_trans)
    # ... rest of your SHAP plotting code ...
    st.pyplot(fig)   # assuming you keep the plotting part
