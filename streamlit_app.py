import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

model = pickle.load(open('fraud_model.pkl', 'rb'))

st.title("💳 Credit Card Fraud Detection")
st.caption("Random Forest model trained on 284,807 real transactions — 99% precision, 76% recall on fraud.")

st.subheader("Try a sample transaction")
col1, col2, col3 = st.columns(3)

if "amount" not in st.session_state:
    st.session_state.amount = 50.0
    st.session_state.time = 40000.0
    st.session_state.v14 = 0.0
    st.session_state.v4 = 0.0
    st.session_state.v11 = 0.0

def load_preset(amount, time, v14, v4, v11):
    st.session_state.amount = amount
    st.session_state.time = time
    st.session_state.v14 = v14
    st.session_state.v4 = v4
    st.session_state.v11 = v11

with col1:
    if st.button("🟢 Typical legit purchase"):
        load_preset(amount=45.0, time=50000.0, v14=0.3, v4=-0.1, v11=-0.2)

with col2:
    if st.button("🔴 Likely fraud pattern"):
        load_preset(amount=9.0, time=5000.0, v14=-6.5, v4=3.2, v11=4.1)

with col3:
    if st.button("🔄 Reset to blank"):
        load_preset(amount=0.0, time=0.0, v14=0.0, v4=0.0, v11=0.0)

st.divider()

left, right = st.columns([1, 1.3])

with left:
    st.subheader("Transaction details")
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=st.session_state.amount, step=1.0)
    time = st.number_input("Transaction Time (seconds since first transaction)", min_value=0.0,
                            value=st.session_state.time, step=100.0)

    st.subheader("Top risk signals (from SHAP analysis)")
    st.caption("These 3 features had the strongest influence on the model's predictions.")

    v14 = st.slider("V14 — low values push toward fraud", -10.0, 10.0, float(st.session_state.v14), 0.1)
    v4 = st.slider("V4 — high values push toward fraud", -10.0, 10.0, float(st.session_state.v4), 0.1)
    v11 = st.slider("V11 — high values push toward fraud", -10.0, 10.0, float(st.session_state.v11), 0.1)

    predict_clicked = st.button("🔍 Predict", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Prediction + explanation
# ---------------------------------------------------------
with right:
    st.subheader("Result")

    if predict_clicked:
        input_data = np.zeros(30)
        input_data[0] = time      
        input_data[3] = v4        
        input_data[10] = v11      
        input_data[13] = v14      
        input_data[29] = amount   

        prediction = model.predict([input_data])[0]
        probability = model.predict_proba([input_data])[0][1]

        if prediction == 1:
            st.error(f"🚨 FRAUD DETECTED — {probability*100:.1f}% confidence")
        else:
            st.success(f"✅ Legitimate Transaction — {probability*100:.1f}% fraud probability")

        st.progress(min(probability, 1.0))
        st.metric("Fraud probability", f"{probability*100:.1f}%")

        st.markdown("**Why the model decided this:**")
        reasons = []
        if v14 < -2:
            reasons.append("🔻 V14 is abnormally low — strong fraud signal, model recommends blocking + customer verification.")
        if v4 > 2:
            reasons.append("🔺 V4 is abnormally high — associated with fraud in this model.")
        if v11 > 2:
            reasons.append("🔺 V11 is abnormally high — associated with fraud in this model.")
        if amount < 20:
            reasons.append("💵 Small transaction amount — fraud cases skew smaller, not larger, in this dataset.")
        if not reasons:
            reasons.append("No individual signal was strongly abnormal — decision is based on the combined feature pattern.")

        for r in reasons:
            st.write(r)
    else:
        st.info("Adjust the values on the left, or load a sample transaction, then click **Predict**.")

st.divider()
st.info("Note: This demo exposes Amount, Time, and the 3 most predictive SHAP features (V14, V4, V11) as controls. "
        "The remaining V1–V28 features are set to 0 for demo purposes. In production, all 30 features would be "
        "captured automatically by the bank's transaction system.")
