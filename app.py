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

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Load dataset once at the start
df = pd.read_csv("C:/Users/lenovo/Downloads/Student_Placement_Dataset_10000_v2.csv")

st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓", layout="wide")

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

# ============================================================
# Visual identity — palette, type, and shared CSS
# ============================================================
NAVY = "#10243E"
SLATE = "#4B5A73"
GOLD = "#B08D57"
SUCCESS = "#1F7A5C"
DANGER = "#B3413C"
BG = "#F5F7FA"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
h1, h2, h3 {{ font-family: 'Sora', sans-serif; color: {NAVY}; }}

.stApp {{ background-color: {BG}; }}

/* ---- Header banner ---- */
.app-header {{
    background: linear-gradient(135deg, {NAVY} 0%, #1B3A5C 100%);
    padding: 2.2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.75rem;
}}
.app-header .eyebrow {{
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.72rem;
    font-weight: 600;
    color: {GOLD};
    margin: 0 0 0.4rem 0;
}}
.app-header h1 {{
    color: #FFFFFF;
    font-size: 1.9rem;
    margin: 0 0 0.4rem 0;
}}
.app-header p.sub {{
    color: #C9D4E3;
    font-size: 0.95rem;
    margin: 0;
    max-width: 640px;
}}

/* ---- Card sections (native bordered containers) ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 10px !important;
    border: 1px solid #E1E6ED !important;
    box-shadow: 0 1px 3px rgba(16, 36, 62, 0.06);
    background-color: #FFFFFF;
}}

.section-title {{
    font-family: 'Sora', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: {NAVY};
    border-left: 4px solid {GOLD};
    padding-left: 0.65rem;
    margin-bottom: 0.9rem;
}}

/* ---- Buttons ---- */
.stFormSubmitButton > button, .stButton > button {{
    background-color: {NAVY};
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 0.6rem 2.2rem;
    transition: 0.15s ease;
}}
.stFormSubmitButton > button:hover, .stButton > button:hover {{
    background-color: {GOLD};
    color: {NAVY};
}}

/* ---- Metric ---- */
[data-testid="stMetricValue"] {{ color: {NAVY}; font-family: 'Sora', sans-serif; }}
[data-testid="stMetricLabel"] {{ color: {SLATE}; }}

/* ---- Result badge ---- */
.badge {{
    display: inline-block;
    padding: 0.55rem 1.3rem;
    border-radius: 6px;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.01em;
}}
.badge-success {{ background-color: #E6F4EF; color: {SUCCESS}; border: 1px solid {SUCCESS}55; }}
.badge-danger  {{ background-color: #FBEAE9; color: {DANGER};  border: 1px solid {DANGER}55; }}

/* ---- Footer ---- */
.app-footer {{
    color: {SLATE};
    font-size: 0.8rem;
    text-align: center;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #E1E6ED;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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


# ============================================================
# Sidebar — branding, context, disclaimer
# ============================================================
with st.sidebar:
    st.markdown(f"<h3 style='color:{NAVY};'>🎓 Placement Analytics</h3>", unsafe_allow_html=True)
    st.caption("Decision-support tool for placement cells and career counselors.")
    st.divider()
    st.markdown("**How it works**")
    st.write(
        "A Random Forest model, trained on historical placement records, "
        "estimates a student's placement probability from academic, skill, "
        "and experience data. A SHAP explanation shows which factors moved "
        "the prediction most."
    )
    st.divider()
    st.caption(f"Decision threshold: **{THRESHOLD:.2f}** placement probability")
    st.caption("For guidance purposes only — not a substitute for individual counseling.")


# ============================================================
# Header
# ============================================================
st.markdown(
    f"""
    <div class="app-header">
        <p class="eyebrow">Placement Analytics</p>
        <h1>Student Placement Prediction System</h1>
        <p class="sub">Enter a student's academic and skill profile to estimate placement
        likelihood, with a breakdown of the leading factors behind the result.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Preparing the model (first load only)..."):
    pipeline = train_model()


# ============================================================
# Input form
# ============================================================
with st.form("student_form"):

    with st.container(border=True):
        st.markdown('<div class="section-title">📘 Academic Record</div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        p10 = a1.slider("10th Percentage", 40, 100, 75)
        p12 = a2.slider("12th Percentage", 40, 100, 75)
        grad = a3.slider("Graduation Percentage", 40, 100, 70)
        a4, a5, a6 = st.columns(3)
        cgpa = a4.slider("CGPA", 5.0, 10.0, 7.5, 0.1)
        backlogs = a5.number_input("Backlogs", min_value=0, max_value=15, value=0)
        attendance = a6.slider("Attendance %", 50, 100, 80)

    st.write("")

    with st.container(border=True):
        st.markdown('<div class="section-title">🧠 Skill Assessments</div>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        aptitude = s1.slider("Aptitude Score", 30, 100, 70)
        coding = s2.slider("Coding Score", 30, 100, 65)
        communication = s3.slider("Communication Score", 30, 100, 67)
        s4, s5, s6 = st.columns(3)
        technical = s4.slider("Technical Score", 30, 100, 65)
        mock = s5.slider("Mock Interview Score", 30, 100, 70)
        resume = s6.slider("Resume Score", 30, 100, 70)

    st.write("")

    with st.container(border=True):
        st.markdown('<div class="section-title">💼 Experience</div>', unsafe_allow_html=True)
        e1, e2, e3, e4 = st.columns(4)
        internship = e1.selectbox("Internship", ["Yes", "No"])
        internship_months = e2.number_input("Internship Months", min_value=0, max_value=24, value=3)
        projects = e3.number_input("Projects", min_value=0, max_value=20, value=4)
        certifications = e4.number_input("Certifications", min_value=0, max_value=20, value=5)

    st.write("")
    submitted = st.form_submit_button("Predict Placement")


# ============================================================
# Prediction
# ============================================================
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

    st.write("")
    with st.container(border=True):
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
        r1, r2 = st.columns([2, 1])
        with r1:
            if is_placed:
                st.markdown('<div class="badge badge-success">✅ LIKELY TO BE PLACED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge badge-danger">⚠️ AT RISK — RECOMMEND EARLY INTERVENTION</div>', unsafe_allow_html=True)
        with r2:
            st.metric("Placement Probability", f"{proba * 100:.1f}%")
        st.write("")
        st.progress(proba)
        st.caption(f"Classified using a tuned decision threshold of {THRESHOLD:.2f}.")

    st.write("")

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] * 100

    if prediction == 1:
        st.success(f"✅ Likely to be placed! Confidence: {probability:.2f}%")
    else:
        st.error(f"❌ May not be placed. Confidence: {100 - probability:.2f}%")


  
    st.metric("Placement Probability", f"{probability:.2f}%")
    st.progress(int(probability))



    # SHAP explanation
    with st.container(border=True):
        st.markdown('<div class="section-title">Why this prediction?</div>', unsafe_allow_html=True)
        preprocessor = pipeline.named_steps["prep"]
        model = pipeline.named_steps["model"]
        X_trans = preprocessor.transform(d)
        feat_names = [n.split("__")[-1] for n in preprocessor.get_feature_names_out()]

        shap_vals = shap.TreeExplainer(model).shap_values(X_trans)
        arr = np.array(shap_vals)
        if isinstance(shap_vals, list):
            contrib = shap_vals[1][0]
        elif arr.ndim == 3:
            contrib = arr[0, :, 1]
        else:
            contrib = arr[0]

        s = pd.Series(contrib, index=feat_names).sort_values(key=abs, ascending=False).head(8).iloc[::-1]
        colors = [SUCCESS if v > 0 else DANGER for v in s.values]

        plt.rcParams["font.family"] = "sans-serif"
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.barh(s.index, s.values, color=colors)
        ax.axvline(0, color=NAVY, linewidth=0.8)
        ax.set_xlabel("Contribution to placement  (green = toward Placed, red = toward Not Placed)", color=SLATE)
        ax.tick_params(colors=SLATE)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        st.pyplot(fig)

st.markdown(
    '<div class="app-footer">Student Placement Prediction System &middot; '
    'Model outputs are estimates for guidance only.</div>',
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🎓 Student Placement Predictor</h1>", unsafe_allow_html=True)
# -------------------------------
# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("📋 Student Search")

# Load your dataset first (make sure df is defined earlier)
# Example: df = pd.read_csv("Student_Placement_Dataset_10000_v2.csv")

# Dropdown with search
student_name = st.sidebar.selectbox(
    "🔍 Select Student",
    options=df['Name'].unique()
)

# Show selected student details
if student_name:
    student = df[df['Name'] == student_name].iloc[0]
    st.write("Student Details:", student)

    # Auto-fill sliders with student data
    cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, float(student['CGPA']))
    iq = st.sidebar.slider("IQ Level", 50, 150, int(student['IQ']))
    comm = st.sidebar.selectbox(
        "Communication Skills",
        ["Poor", "Average", "Good"],
        index=int(student['Comm'])  # assuming Comm is stored as 0/1/2
    )

