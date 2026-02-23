import streamlit as st
import joblib
import pandas as pd

# Load full trained pipeline (Scaler + Model together)
model = joblib.load('insurance_pipeline.joblib')

st.set_page_config(page_title="Insurance Charges Predictor", layout="wide")

st.title("💰 Insurance Charges Prediction")
st.markdown("### Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    claim_amount = st.number_input('Claim Amount (USD)', min_value=0.0, value=1000.0, step=100.0)
    past_consultations = st.number_input('Past Consultations', min_value=0, value=2, step=1)

with col2:
    hospital_expenditure = st.number_input('Hospital Expenditure (USD)', min_value=0.0, value=500.0, step=50.0)
    annual_salary = st.number_input('Annual Salary (USD)', min_value=0.0, value=50000.0, step=1000.0)

with col3:
    children = st.number_input('Number of Children', min_value=0, value=1, step=1)
    smoker_option = st.selectbox('Smoker?', ('No', 'Yes'))
    smoker = 1 if smoker_option == 'Yes' else 0

if st.button('Predict Charges'):

    input_df = pd.DataFrame([{
        'claim_amount': claim_amount,
        'past_consultations': past_consultations,
        'hospital_expenditure': hospital_expenditure,
        'annual_salary': annual_salary,
        'children': children,
        'smoker': smoker
    }])

    predicted_charges_usd = model.predict(input_df)[0]
    predicted_charges_inr = predicted_charges_usd * 83

    st.success(f'Predicted Insurance Charges (USD): **${predicted_charges_usd:,.2f}**')
    st.success(f'Predicted Insurance Charges (INR): **₹{predicted_charges_inr:,.2f}**')


    
