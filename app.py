"""
Student Placement Prediction System - Self-contained Streamlit App
Trains the model directly from the CSV at startup (cached), so there is no
pickle file and no Python-version mismatch. Includes prediction, probability,
and a SHAP explanation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓", layout="wide")

st.markdown("---")
st.markdown(
    "<center><b>Developed by Megha Mishra | MBA Business Analytics</b></center>",
    unsafe_allow_html=True
)

DATA_FILE = "Student_Placement_Dataset_10000_v2.csv"
THRESHOLD = 0.60
SKILL_COLS = ["Aptitude_Score", "Coding_Score", "Communication_Score",
              "Technical_Score", "Mock_Interview_Score", "Resume_Score"]
FINAL_NUM = ["10th_Percentage", "12th_Percentage", "Graduation_Percentage", "CGPA",
             "Backlogs", "Attendance", "Internship_Months", "Projects", "Certifications",
             "Aptitude_Score", "Coding_Score", "Communication_Score", "Technical_Score",
             "Mock_Interview_Score", "Resume_Score",
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
    """Load the CSV, engineer features, and train the pipeline. Cached so it runs once."""
    df = pd.read_csv(DATA_FILE).drop(columns=["Student_ID", "Company_Type", "Placement_Package_LPA"])
    df["Placement_Status"] = (df["Placement_Status"] == "Placed").astype(int)
    df = add_engineered_features(df)

    X = df[FINAL_NUM + FINAL_CAT]
    y = df["Placement_Status"]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), make_column_selector(dtype_include=np.number)),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]), make_column_selector(dtype_exclude=np.number)),
    ])
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=2,
                                         class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    pipeline.fit(X, y)
    return pipeline


# ---------- Header ----------
st.title("🎓 Student Placement Prediction System")
st.write("Enter a student's details to predict placement likelihood, with an explanation "
         "of which factors drove the result.")

with st.spinner("Preparing the model (first load only)..."):
    pipeline = train_model()


# ---------- Input form ----------
with st.form("student_form"):
    st.subheader("Academic Record")
    a1, a2, a3 = st.columns(3)
    p10 = a1.slider("10th Percentage", 40, 100, 75)
    p12 = a2.slider("12th Percentage", 40, 100, 75)
    grad = a3.slider("Graduation Percentage", 40, 100, 70)
    a4, a5, a6 = st.columns(3)
    cgpa = a4.slider("CGPA", 5.0, 10.0, 7.5, 0.1)
    backlogs = a5.number_input("Backlogs", min_value=0, max_value=15, value=0)
    attendance = a6.slider("Attendance %", 50, 100, 80)

    st.subheader("Skill Assessments")
    s1, s2, s3 = st.columns(3)
    aptitude = s1.slider("Aptitude Score", 30, 100, 70)
    coding = s2.slider("Coding Score", 30, 100, 65)
    communication = s3.slider("Communication Score", 30, 100, 67)
    s4, s5, s6 = st.columns(3)
    technical = s4.slider("Technical Score", 30, 100, 65)
    mock = s5.slider("Mock Interview Score", 30, 100, 70)
    resume = s6.slider("Resume Score", 30, 100, 70)

    st.subheader("Experience")
    e1, e2, e3, e4 = st.columns(4)
    internship = e1.selectbox("Internship", ["Yes", "No"])
    internship_months = e2.number_input("Internship Months", min_value=0, max_value=24, value=3)
    projects = e3.number_input("Projects", min_value=0, max_value=20, value=4)
    certifications = e4.number_input("Certifications", min_value=0, max_value=20, value=5)

    submitted = st.form_submit_button("Predict Placement")


# ---------- Prediction ----------
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
    r1, r2 = st.columns([2, 1])
    with r1:
        if is_placed:
            st.success("### ✅ Likely to be PLACED")
        else:
            st.error("### ⚠️ AT RISK — recommend early intervention")
    with r2:
        st.metric("Placement Probability", f"{proba * 100:.1f}%")
    st.progress(proba)
    st.caption(f"Classified using a tuned decision threshold of {THRESHOLD:.2f}.")
st.subheader("📊 Student Performance Analysis")

analysis = {
    "CGPA": cgpa * 10,
    "Coding": coding,
    "Communication": communication,
    "Technical": technical,
    "Aptitude": aptitude,
    "Resume": resume,
    "Attendance": attendance,
    "Internship": 100 if internship == "Yes" else 0,
    "Projects": min(projects * 10, 100),
    "Certifications": min(certifications * 10, 100)
}

analysis_df = pd.DataFrame({
    "Feature": list(analysis.keys()),
    "Score": list(analysis.values())
})

fig, ax = plt.subplots(figsize=(8,5))

colors = []

for score in analysis_df["Score"]:
    if score >= 75:
        colors.append("green")
    elif score >= 50:
        colors.append("orange")
    else:
        colors.append("red")

ax.barh(
    analysis_df["Feature"],
    analysis_df["Score"],
    color=colors
)

ax.set_xlim(0,100)
ax.set_xlabel("Performance Score")
ax.set_title("Student-wise Performance Analysis")

plt.gca().invert_yaxis()

st.pyplot(fig)

st.subheader("📌 Prediction Insights")

strengths = []
improvements = []

if cgpa >= 7.5:
    strengths.append("Strong CGPA")
else:
    improvements.append("Improve CGPA")

if coding >= 70:
    strengths.append("Good Coding Skills")
else:
    improvements.append("Improve Coding Skills")

if communication >= 70:
    strengths.append("Good Communication")
else:
    improvements.append("Improve Communication")

if internship == "Yes":
    strengths.append("Internship Experience")
else:
    improvements.append("Complete an Internship")

if projects >= 3:
    strengths.append("Good Project Experience")
else:
    improvements.append("Work on More Projects")

st.success("### Strengths")
for item in strengths:
    st.write("✅", item)

st.warning("### Areas to Improve")
for item in improvements:
    st.write("⚠️", item)
score = round(proba*100)

st.progress(score/100)

st.write(f"Overall Employability Score : **{score}/100**")
