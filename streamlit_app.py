import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('fraud_model.pkl', 'rb'))

st.title('Credit Card Fraud Detection')
st.write('Enter transaction details below.')

amount = st.number_input('Transaction Amount ($)', min_value=0.0)
time = st.number_input('Transaction Time (seconds)', min_value=0.0)

if st.button('Predict'):
    input_data = np.zeros(30)
    input_data[0] = time
    input_data[29] = amount
    prediction = model.predict([input_data])
    probability = model.predict_proba([input_data])[0][1]
    
    if prediction[0] == 1:
        st.error(f'FRAUD DETECTED — {probability*100:.1f}% confidence')
    else:
        st.success(f'Legitimate Transaction — {probability*100:.1f}% fraud probability')