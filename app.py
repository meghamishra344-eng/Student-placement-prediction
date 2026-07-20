import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# Load dataset and model
# -------------------------------
df = pd.read_csv("Student_Placement_Dataset_10000_v2.csv")
model = pickle.load(open("model.pkl", "rb"))

st.title("🎓 Student Placement Predictor")

# -------------------------------
# Sidebar: Student Search
# -------------------------------
student_id = st.sidebar.selectbox("Select Student", df['StudentID'].unique())
student = df[df['StudentID'] == student_id].iloc[0]

# Academic Record
cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, float(student['CGPA']))
tenth = st.sidebar.slider("10th Percentage", 0, 100, int(student['10th_Percentage']))
twelfth = st.sidebar.slider("12th Percentage", 0, 100, int(student['12th_Percentage']))
grad = st.sidebar.slider("Graduation Percentage", 0, 100, int(student['Graduation_Percentage']))
backlogs = st.sidebar.slider("Backlogs", 0, 10, int(student['Backlogs']))
attendance = st.sidebar.slider("Attendance %", 0, 100, int(student['Attendance']))

# Skill Assessments
aptitude = st.sidebar.slider("Aptitude Score", 0, 100, int(student['Aptitude_Score']))
coding = st.sidebar.slider("Coding Score", 0, 100, int(student['Coding_Score']))
comm_score = int(student['Communication_Score'])
if comm_score < 40:
    comm_index = 0
elif comm_score < 70:
    comm_index = 1
else:
    comm_index = 2
comm = st.sidebar.selectbox("Communication Skills", ["Poor", "Average", "Good"], index=comm_index)
technical = st.sidebar.slider("Technical Score", 0, 100, int(student['Technical_Score']))
mock = st.sidebar.slider("Mock Interview Score", 0, 100, int(student['Mock_Interview_Score']))
resume = st.sidebar.slider("Resume Score", 0, 100, int(student['Resume_Score']))

# Experience
internship = st.sidebar.selectbox("Internship", ["No", "Yes"], index=int(student['Internship']))
internship_months = st.sidebar.slider("Internship Months", 0, 12, int(student['Internship_Months']))
projects = st.sidebar.slider("Projects", 0, 10, int(student['Projects']))
certifications = st.sidebar.slider("Certifications", 0, 20, int(student['Certifications']))

# -------------------------------
# Build features for prediction
# -------------------------------
features = np.array([[cgpa, aptitude, coding, comm_score, technical, mock]])

# -------------------------------
# Prediction
# -------------------------------
if st.sidebar.button("Predict Placement"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] * 100

    if prediction == 1:
        st.success(f"✅ Likely to be placed! Confidence: {probability:.2f}%")
    else:
        st.error(f"❌ May not be placed. Confidence: {100 - probability:.2f}%")

    st.metric("Placement Probability", f"{probability:.2f}%")
    st.progress(int(probability))

    # -------------------------------
    # Student Profile Graph
    # -------------------------------
    scores = {
        "10th %": tenth,
        "12th %": twelfth,
        "Graduation %": grad,
        "CGPA": cgpa,
        "Aptitude": aptitude,
        "Coding": coding,
        "Communication": comm_score,
        "Technical": technical,
        "Mock Interview": mock,
        "Resume": resume,
        "Backlogs": backlogs,
        "Attendance": attendance,
        "Internship Months": internship_months,
        "Projects": projects,
        "Certifications": certifications
    }

    df_scores = pd.DataFrame(list(scores.items()), columns=["Metric", "Value"]).set_index("Metric")
    st.subheader("📊 Student Profile")
    st.bar_chart(df_scores)
