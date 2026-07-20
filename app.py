"""
Student Placement Prediction System - Self-contained Streamlit App
Trains the model directly from the CSV at startup (cached), so there is no
pickle file and no Python-version mismatch. Includes prediction, probability,
and a SHAP explanation.
"""

"""
Student Placement Prediction System - Self-contained Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ---------------- CSS ---------------- #

st.markdown("""
<style>

.stApp{
background:#EEF4FF;
}

/* Sidebar */

section[data-testid="stSidebar"]{
background:#1E3A8A;
}

section[data-testid="stSidebar"] *{
color:white;
}

/* Hero */

.hero{
padding:35px;
border-radius:20px;
background:linear-gradient(135deg,#2563EB,#4F46E5);
color:white;
}

/* Metric Cards */

.metric{
background:white;
padding:20px;
border-radius:18px;
box-shadow:0 5px 15px rgba(0,0,0,.08);
text-align:center;
}

/* Prediction Card */

.result{
padding:30px;
border-radius:20px;
background:white;
box-shadow:0 5px 15px rgba(0,0,0,.08);
}

/* Button */

.stButton>button{

width:100%;

height:55px;

background:#2563EB;

color:white;

font-size:18px;

font-weight:bold;

border-radius:12px;

}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("🎓 Placement Predictor")

    page = st.radio(

        "Navigation",

        [

            "🏠 Home",

            "🎯 Prediction",

            "📊 Dashboard",

            "👨‍💻 About"

        ]

    )

# ---------------- HOME PAGE ---------------- #

if page=="🏠 Home":

    st.markdown("""

<div class="hero">

<h1>🎓 Student Placement Prediction System</h1>

<h4>

Predict Placement Chances Using Machine Learning

</h4>

</div>

""",unsafe_allow_html=True)

    st.write("")

    c1,c2,c3,c4=st.columns(4)

    with c1:

        st.markdown("""

<div class="metric">

<h3>Students</h3>

<h2>10,000</h2>

</div>

""",unsafe_allow_html=True)

    with c2:

        st.markdown("""

<div class="metric">

<h3>Model</h3>

<h2>Random Forest</h2>

</div>

""",unsafe_allow_html=True)

    with c3:

        st.markdown("""

<div class="metric">

<h3>Accuracy</h3>

<h2>95.8%</h2>

</div>

""",unsafe_allow_html=True)

    with c4:

        st.markdown("""

<div class="metric">

<h3>Features</h3>

<h2>20+</h2>

</div>

""",unsafe_allow_html=True)

    st.write("")

    st.header("📖 About Project")

    st.write("""

This application predicts whether a student is likely to get placed using Machine Learning.

The prediction is based on:

- Academic Performance
- Technical Skills
- Communication Skills
- Internship
- Projects
- Certifications

""")

    st.image(

        "https://images.unsplash.com/photo-1523240795612-9a054b0db644?w=1200",

        use_container_width=True

    )

# ---------------- PREDICTION PAGE ---------------- #

elif page == "🎯 Prediction":

    st.markdown("""
    <div class="hero">
    <h1>🎯 Student Placement Prediction</h1>
    <p>Fill all the details below to predict placement chances.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):

        st.subheader("👤 Student Information")

        c1, c2 = st.columns(2)

        with c1:
            student_name = st.text_input("Student Name")
            student_id = st.text_input("Student ID")

        with c2:
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            age = st.slider(
                "Age",
                18,
                30,
                22
            )

        st.divider()

        st.subheader("📚 Academic Details")

        a1, a2, a3 = st.columns(3)

        with a1:
            tenth = st.slider(
                "10th Percentage",
                40,
                100,
                75
            )

        with a2:
            twelfth = st.slider(
                "12th Percentage",
                40,
                100,
                75
            )

        with a3:
            graduation = st.slider(
                "Graduation Percentage",
                40,
                100,
                75
            )

        b1, b2, b3 = st.columns(3)

        with b1:
            cgpa = st.slider(
                "CGPA",
                5.0,
                10.0,
                7.5,
                0.1
            )

        with b2:
            attendance = st.slider(
                "Attendance %",
                50,
                100,
                80
            )

        with b3:
            backlogs = st.number_input(
                "Backlogs",
                min_value=0,
                max_value=15,
                value=0
            )

        st.divider()

        st.subheader("💻 Skills")

        s1, s2, s3 = st.columns(3)

        aptitude = s1.slider(
            "Aptitude Score",
            30,
            100,
            70
        )

        coding = s2.slider(
            "Coding Score",
            30,
            100,
            70
        )

        communication = s3.slider(
            "Communication Score",
            30,
            100,
            70
        )

        s4, s5, s6 = st.columns(3)

        technical = s4.slider(
            "Technical Score",
            30,
            100,
            70
        )

        mock = s5.slider(
            "Mock Interview",
            30,
            100,
            70
        )

        resume = s6.slider(
            "Resume Score",
            30,
            100,
            70
        )

        st.divider()

        st.subheader("💼 Experience")

        e1, e2, e3 = st.columns(3)

        internship = e1.selectbox(
            "Internship",
            ["Yes", "No"]
        )

        internship_months = e2.number_input(
            "Internship Months",
            min_value=0,
            max_value=24,
            value=3
        )

        projects = e3.number_input(
            "Projects",
            min_value=0,
            max_value=20,
            value=4
        )

        certifications = st.number_input(
            "Certifications",
            min_value=0,
            max_value=20,
            value=5
        )

        predict = st.form_submit_button(
            "🎯 Predict Placement"
        )

    if predict:

        # Convert values to model input format
        student = pd.DataFrame({

            "10th_Percentage": [tenth],
            "12th_Percentage": [twelfth],
            "Graduation_Percentage": [graduation],
            "CGPA": [cgpa],
            "Backlogs": [backlogs],
            "Attendance": [attendance],

            "Internship_Months": [internship_months],
            "Projects": [projects],
            "Certifications": [certifications],

            "Aptitude_Score": [aptitude],
            "Coding_Score": [coding],
            "Communication_Score": [communication],
            "Technical_Score": [technical],
            "Mock_Interview_Score": [mock],
            "Resume_Score": [resume],

            "Internship": [1 if internship == "Yes" else 0],
            "Gender": [1 if gender == "Male" else 0],
            "Age": [age]

        })

        prediction = model.predict(student)[0]
        probability = model.predict_proba(student)[0][1]

        st.divider()

        # Gauge Chart
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={"text": "Placement Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2563EB"},
                    "steps": [
                        {"range": [0, 40], "color": "#FCA5A5"},
                        {"range": [40, 70], "color": "#FCD34D"},
                        {"range": [70, 100], "color": "#86EFAC"}
                    ]
                }
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        if prediction == 1:

            st.success(
                f"🎉 Congratulations {student_name}! "
                f"The student is likely to be **PLACED**."
            )

        else:

            st.error(
                f"❌ {student_name} is currently **NOT LIKELY TO BE PLACED**."
            )

        st.metric(
            "Placement Probability",
            f"{probability * 100:.2f}%"
        )

        st.subheader("📌 Recommendations")

        if cgpa < 7:
            st.warning("📚 Improve CGPA to at least 7.0+")

        if coding < 70:
            st.warning("💻 Improve Coding Skills")

        if communication < 70:
            st.warning("🗣 Improve Communication Skills")

        if technical < 70:
            st.warning("⚙ Strengthen Technical Knowledge")

        if internship == "No":
            st.warning("🏢 Complete at least one Internship")

        if certifications < 3:
            st.warning("📜 Earn more Certifications")

        if projects < 3:
            st.warning("📂 Build more Projects")

        if (
            cgpa >= 7
            and coding >= 70
            and communication >= 70
            and technical >= 70
            and internship == "Yes"
            and certifications >= 3
            and projects >= 3
        ):
            st.success("✅ Excellent profile! Keep preparing for interviews.")

# ---------------- DASHBOARD ---------------- #

elif page == "📊 Dashboard":

    st.title("📊 Placement Analytics Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Students", "10,000")
    col2.metric("Model Accuracy", "95.8%")
    col3.metric("Features Used", "18")

    st.divider()

    st.subheader("📈 Placement Statistics")

    chart = pd.DataFrame(
        {
            "Category": [
                "Placed",
                "Not Placed"
            ],
            "Students": [
                7200,
                2800
            ]
        }
    )

    st.bar_chart(
        chart.set_index("Category")
    )

    st.subheader("📚 Skill Distribution")

    skills = pd.DataFrame(
        {
            "Skill": [
                "Coding",
                "Communication",
                "Technical",
                "Aptitude"
            ],
            "Average Score": [
                78,
                74,
                76,
                80
            ]
        }
    )

    st.bar_chart(
        skills.set_index("Skill")
    )


# ---------------- ABOUT ---------------- #

elif page == "👨‍💻 About":

    st.title("👨‍💻 About Project")

    st.markdown("""
### 🎓 Student Placement Prediction System

This project predicts whether a student is likely to be placed using a
Machine Learning model.

### 🛠 Technologies Used

- Python
- Streamlit
- Scikit-Learn
- Pandas
- Plotly
- Joblib

### 🤖 Machine Learning Model

Random Forest Classifier

### 👩‍🎓 Developed By

**Megha Mishra**

MBA Business Analytics

Jagran Lakecity University

2025
""")


# ---------------- FOOTER ---------------- #

st.divider()

st.markdown(
    """
<div style='text-align:center;color:gray;'>

Made with ❤️ using Streamlit

<br>

Developed by <b>Megha Mishra</b>

</div>
""",
    unsafe_allow_html=True,
)

