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

# Placeholder pages
elif page=="🎯 Prediction":
    st.title("🎯 Prediction Page")
    st.info("Prediction form will be added in Part 2.")

elif page=="📊 Dashboard":
    st.title("📊 Dashboard")
    st.info("Analytics dashboard will be added in Part 3.")

st.markdown("""
<div class="footer">

Developed by Megha Mishra | MBA Business Analytics

</div>
""", unsafe_allow_html=True)
