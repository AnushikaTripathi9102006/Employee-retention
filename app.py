import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="HR Employee Retention Predictor",
    page_icon="👨‍💼",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#141E30,#243B55);
}

.main-title{
    text-align:center;
    color:white;
    font-size:42px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:#dddddd;
    font-size:18px;
    margin-bottom:30px;
}

div[data-testid="stForm"]{
    background-color:rgba(255,255,255,0.08);
    padding:30px;
    border-radius:20px;
    box-shadow:0px 10px 25px rgba(0,0,0,0.3);
}

.stButton>button{
    width:100%;
    border-radius:10px;
    background:#00C6FF;
    color:white;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#0072FF;
    color:white;
}

.result-box{
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown("<div class='main-title'>👨‍💼 HR Employee Retention Predictor</div>",
            unsafe_allow_html=True)

st.markdown("<div class='sub-title'>Predict whether an employee is likely to leave the company.</div>",
            unsafe_allow_html=True)

# -----------------------------
# Input Form
# -----------------------------
with st.form("prediction_form"):

    satisfaction = st.slider(
        "Satisfaction Level",
        0.0,
        1.0,
        0.50,
        0.01
    )

    hours = st.slider(
        "Average Monthly Hours",
        80,
        320,
        200
    )

    promotion = st.selectbox(
        "Promotion in Last 5 Years",
        ["No","Yes"]
    )

    salary = st.selectbox(
        "Salary Level",
        ["Low","Medium","High"]
    )

    predict = st.form_submit_button("Predict")

# -----------------------------
# Prediction
# -----------------------------
if predict:

    promotion = 1 if promotion=="Yes" else 0

    salary_high = 1 if salary=="High" else 0
    salary_low = 1 if salary=="Low" else 0
    salary_medium = 1 if salary=="Medium" else 0

    X = pd.DataFrame([{
        "satisfaction_level": satisfaction,
        "average_montly_hours": hours,
        "promotion_last_5years": promotion,
        "salary_high": salary_high,
        "salary_low": salary_low,
        "salary_medium": salary_medium
    }])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1] * 100

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.markdown(
            f"""
            <div class="result-box"
            style="background:#ffebee;color:#c62828;">
            ⚠️ Employee is Likely to Leave
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="result-box"
            style="background:#e8f5e9;color:#2e7d32;">
            ✅ Employee is Likely to Stay
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(min(probability/100, 1.0))
    st.metric("Probability of Leaving", f"{probability:.2f}%")
