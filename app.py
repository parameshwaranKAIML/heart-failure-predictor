import streamlit as st
import joblib
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="CardioCare AI | Clinical Portal", 
    page_icon="❤️", 
    layout="wide"
)

# Custom CSS Styling for a Modern, Premium Medical Dashboard
st.markdown("""
    <style>
    /* Main Background & Font Styling */
    .main {
        background-color: #f8fafc;
    }
    
    /* Custom Card Containers */
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }
    
    /* Style Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        color: #ffffff;
    }
    [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('heart_failure_model.pkl')

model = load_model()

# --- SIDEBAR INPUTS ---
st.sidebar.markdown("<h2 style='color: #38bdf8;'>📋 Patient Vitals</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #94a3b8; font-size: 0.85rem;'>Input clinical diagnostic parameters below:</p>", unsafe_allow_html=True)

with st.sidebar.expander("👤 Demographics & Habits", expanded=True):
    age = st.slider("Age", 40, 95, 60)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with st.sidebar.expander("🩺 Clinical Biomarkers", expanded=True):
    ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 35)
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.5, 10.0, 1.2, step=0.1)
    serum_sodium = st.number_input("Serum Sodium (mEq/L)", 110, 150, 135)
    platelets = st.number_input("Platelets Count", 25000, 850000, 263000)

with st.sidebar.expander("📋 Medical History & Timeline", expanded=True):
    time = st.slider("Follow-up period (days)", 1, 300, 100)
    high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    anaemia = st.selectbox("Anaemia", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# --- MAIN DASHBOARD ---
st.title("❤️ CardioCare AI Decision Support System")
st.markdown("A professional machine learning interface designed to assist clinicians in evaluating 30-day heart failure readmission risk.")

st.markdown("---")

# Quick Overview Cards using native containers
st.subheader("📌 Current Patient Profile Summary")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(label="Age", value=f"{age} Years")
with c2:
    st.metric(label="Ejection Fraction", value=f"{ejection_fraction}%", delta="-Low" if ejection_fraction < 40 else "Normal")
with c3:
    st.metric(label="Serum Creatinine", value=f"{serum_creatinine} mg/dL")
with c4:
    st.metric(label="Follow-up Window", value=f"{time} Days")

st.markdown("<br>", unsafe_allow_html=True)

# Prediction Button & Results
if st.button("🚀 Analyze Clinical Risk Profile", use_container_width=True):
    input_data = pd.DataFrame([[
        age, anaemia, 160, diabetes, ejection_fraction, 
        high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time
    ]], columns=['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'])
    
    prediction_proba = model.predict_proba(input_data)
    high_risk_prob = prediction_proba[0][1] * 100
    
    st.markdown("### 📊 Diagnostic Evaluation Results")
    
    if high_risk_prob >= 60:
        st.error(f"⚠️ **HIGH RISK ALERT** — Model Confidence: **{high_risk_prob:.2f}%**\n\n*Clinical Recommendation:* Immediate priority for early follow-up, tight fluid balance regulation, and medication review.")
    elif 30 <= high_risk_prob < 60:
        st.warning(f"⚠️ **MODERATE / BORDERLINE RISK** — Model Confidence: **{high_risk_prob:.2f}%**\n\n*Clinical Recommendation:* Exercise caution, monitor borderline lab markers closely, and schedule a check-in within 7–14 days.")
    else:
        st.success(f"✅ **LOW RISK STABLE** — Model Confidence: **{100 - high_risk_prob:.2f}%**\n\n*Clinical Recommendation:* Patient maintains stable biometric indicators suitable for standard care routine.")