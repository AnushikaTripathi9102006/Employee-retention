import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="HR Employee Retention Prediction",
    page_icon="👨‍💼",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(to right,#eef2f3,#d9e4f5);
}

h1{
    color:#0B5ED7;
}

div.stButton > button{
    width:100%;
    background:#0B5ED7;
    color:white;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.result{
    padding:20px;
    border-radius:12px;
    text-align:center;
    font-size:26px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_resource
def train_model():

    df = pd.read_csv("HR_comma_sep.csv")

    subdf = df[['satisfaction_level',
                'average_montly_hours',
                'promotion_last_5years',
                'salary']]

    salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")

    X = pd.concat([subdf, salary_dummies], axis=1)
    X.drop("salary", axis=1, inplace=True)

    y = df["left"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, X.columns

model, columns = train_model()

# -----------------------------
# Title
# -----------------------------
st.title("👨‍💼 HR Employee Retention Prediction")
st.write("Predict whether an employee is likely to leave the company.")

st.divider()

col1, col2 = st.columns(2)

with col1:

    satisfaction = st.slider(
        "😊 Satisfaction Level",
        0.00,
        1.00,
        0.50
    )

    monthly_hours = st.slider(
        "⏰ Average Monthly Hours",
        90,
        320,
        200
    )

with col2:

    promotion = st.selectbox(
        "🏆 Promotion in Last 5 Years",
        [0,1]
    )

    salary = st.selectbox(
        "💰 Salary",
        ["low","medium","high"]
    )

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    data = {
        'satisfaction_level': satisfaction,
        'average_montly_hours': monthly_hours,
        'promotion_last_5years': promotion,
        'salary_high':0,
        'salary_low':0,
        'salary_medium':0
    }

    data[f"salary_{salary}"] = 1

    input_df = pd.DataFrame([data])

    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.markdown("""
        <div class="result" style="background:#ffd6d6;color:#c1121f;">
        ⚠️ Employee is likely to LEAVE the company.
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result" style="background:#d8f3dc;color:#1b4332;">
        ✅ Employee is likely to STAY in the company.
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.progress(float(max(probability)))

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Stay Probability",
            f"{probability[0]*100:.2f}%"
        )

    with c2:
        st.metric(
            "Leave Probability",
            f"{probability[1]*100:.2f}%"
        )

st.divider()

st.subheader("📊 Features Used")

st.write("""
- Satisfaction Level
- Average Monthly Hours
- Promotion in Last 5 Years
- Salary Level
""")

st.info("Model: Logistic Regression")
