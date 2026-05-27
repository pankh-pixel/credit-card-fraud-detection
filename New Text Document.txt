import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('fraud_model.pkl', 'rb'))

st.title('Credit Card Fraud Detection')
st.write('Enter transaction details to check if it is fraud or legitimate.')

amount = st.number_input('Transaction Amount ($)', min_value=0.0)
time = st.number_input('Transaction Time (seconds)', min_value=0.0)

if st.button('Predict'):
    input_data = np.zeros(30)
    input_data[0] = time
    input_data[29] = amount
    prediction = model.predict([input_data])
    if prediction[0] == 1:
        st.error('⚠️ FRAUD DETECTED')
    else:
        st.success('✅ Legitimate Transaction')