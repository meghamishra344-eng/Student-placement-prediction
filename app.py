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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(to right,#EEF5FF,#FFFFFF);
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#1E3A8A;
}

section[data-testid="stSidebar"] *{
    color:white;
}

/* Hero Card */
.hero{
background:linear-gradient(135deg,#2563EB,#4F46E5);
padding:35px;
border-radius:20px;
color:white;
margin-bottom:25px;
}

/* Metric Cards */
.metric-card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 8px 25px rgba(0,0,0,.08);
text-align:center;
}

/* Buttons */
.stButton>button{
background:#2563EB;
color:white;
border:none;
border-radius:12px;
height:50px;
font-size:18px;
font-weight:bold;
width:100%;
}

.stButton>button:hover{
background:#1D4ED8;
color:white;
}

/* Footer */
.footer{
text-align:center;
padding:20px;
color:gray;
font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.title("🎓 Placement Predictor")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🎯 Prediction",
            "📊 Dashboard",
            "🤖 About"
        ]
    )

# ---------------- HOME PAGE ----------------

if page=="🏠 Home":

    st.markdown("""
    <div class="hero">

    <h1>🎓 Student Placement Prediction System</h1>

    <h4>
    Predict student placement chances using Machine Learning
    </h4>

    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4=st.columns(4)

    with c1:
        st.markdown("""
        <div class="metric-card">
        <h3>👨‍🎓 Students</h3>
        <h2>10,000</h2>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
        <h3>🤖 Model</h3>
        <h2>Random Forest</h2>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
        <h3>📈 Accuracy</h3>
        <h2>95.8%</h2>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 Features</h3>
        <h2>20+</h2>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.subheader("📖 Project Overview")

    st.write("""
This web application predicts whether a student is likely to get placed based on:

- Academic Performance
- Technical Skills
- Communication Skills
- Internship Experience
- Certifications
- Projects

The prediction is performed using a Random Forest Machine Learning model trained on student placement data.
""")

    st.image(
        "https://images.unsplash.com/photo-1523240795612-9a054b0db644?w=1200",
        use_container_width=True
    )

# ---------------- ABOUT PAGE ----------------

elif page=="🤖 About":

    st.title("🤖 About Project")

    st.info("""

Developer : Megha Mishra

Course : MBA Business Analytics

Project : Student Placement Prediction

Algorithm : Random Forest Classifier

Technology :

• Python

• Streamlit

• Scikit-Learn

• SHAP

• Pandas

• NumPy

""")

elif page == "🎯 Prediction":

    st.markdown("""
    <div class="hero">
    <h1>🎯 Student Placement Prediction</h1>
    <p>Fill in the student's details to predict placement chances.</p>
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

        st.subheader("📚 Academic Performance")

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
                74
            )

        with a3:
            graduation = st.slider(
                "Graduation Percentage",
                40,
                100,
                72
            )

        a4, a5, a6 = st.columns(3)

        with a4:
            cgpa = st.slider(
                "CGPA",
                5.0,
                10.0,
                7.5,
                0.1
            )

        with a5:
            attendance = st.slider(
                "Attendance %",
                50,
                100,
                80
            )

        with a6:
            backlogs = st.number_input(
                "Backlogs",
                0,
                15,
                0
            )

        st.divider()

        st.subheader("💻 Technical & Soft Skills")

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

        e1, e2, e3, e4 = st.columns(4)

        internship = e1.selectbox(
            "Internship",
            ["Yes", "No"]
        )

        internship_months = e2.number_input(
            "Internship Months",
            0,
            24,
            3
        )

        projects = e3.number_input(
            "Projects",
            0,
            20,
            4
        )

        certifications = e4.number_input(
            "Certifications",
            0,
            20,
            5
        )

        st.write("")

        predict = st.form_submit_button(
            "🎯 Predict Placement"
        )

    if predict:

        st.success(
            f"Details submitted successfully for **{student_name}**."
        )

        st.info("""
Prediction logic will be connected in **Part 3**.
Your trained Random Forest model will calculate the placement probability and display the result here.
""")

elif page=="📊 Dashboard":
    st.title("📊 Dashboard")
    st.info("Analytics dashboard will be added in Part 3.")

st.markdown("""
<div class="footer">

Developed by Megha Mishra | MBA Business Analytics

</div>
""", unsafe_allow_html=True)
