import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model and scaler
model = joblib.load('linear_regression_model.joblib')
scaler = joblib.load('scaler.pkl')

st.set_page_config(layout="wide")
st.title('Insurance Charge Prediction')

st.markdown("### Enter Patient Details for Charges Prediction")

# Create input fields for the features
# Use st.columns for better layout
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

# Create a button to predict
if st.button('Predict Charges'):
    # Define ALL numerical columns the scaler was fitted on, including target and derived duplicates
    # This list must EXACTLY match the `numerical_cols` from the notebook's scaling step.
    all_cols_scaler_fitted_on = ['claim_amount', 'past_consultations', 'hospital_expenditure',
                                 'annual_salary', 'children', 'smoker', 'charges',
                                 'children.1', 'smoker.1', 'charges.1']

    # Define the columns the model expects as input features (i.e., X columns)
    model_feature_cols = ['claim_amount', 'past_consultations', 'hospital_expenditure',
                          'annual_salary', 'children', 'smoker', 'children.1',
                          'smoker.1', 'charges.1']

    # Prepare the input data for the scaler.
    # Populate with user inputs and dummy values for target/derived features not from input.
    input_values_for_scaler = [
        claim_amount,
        past_consultations,
        hospital_expenditure,
        annual_salary,
        children,
        smoker,
        0.0, # Placeholder for 'charges' (target)
        children, # Assuming 'children.1' is a duplicate of 'children'
        smoker,   # Assuming 'smoker.1' is a duplicate of 'smoker'
        0.0       # Placeholder for 'charges.1' (assuming it's a duplicate of charges or a dummy)
    ]

    input_data_for_scaler_df = pd.DataFrame([input_values_for_scaler],
                                            columns=all_cols_scaler_fitted_on)

    # Scale the full input data
    scaled_full_data = scaler.transform(input_data_for_scaler_df)

    # Convert the scaled array back to a DataFrame to easily select model features
    scaled_full_df_for_selection = pd.DataFrame(scaled_full_data, columns=all_cols_scaler_fitted_on)

    # Extract only the scaled feature columns that the model was trained on
    scaled_input_features_for_model = scaled_full_df_for_selection[model_feature_cols]

    # Make prediction using the model
    prediction_scaled = model.predict(scaled_input_features_for_model)

    # Find the index of the 'charges' column in the list the scaler was fitted on
    charges_idx = all_cols_scaler_fitted_on.index('charges')

    # Create a dummy array for inverse transformation, matching the shape and columns
    # of the data the scaler was fitted on.
    dummy_input_for_inverse = np.zeros((1, len(all_cols_scaler_fitted_on)))
    dummy_input_for_inverse[0, charges_idx] = prediction_scaled[0] # Place the scaled prediction at the correct index

    # Inverse transform the dummy array to get the prediction in original scale (USD)
    original_scale_values = scaler.inverse_transform(dummy_input_for_inverse)
    predicted_charges_usd = original_scale_values[0, charges_idx]

    # Convert to INR (assuming 1 USD = 83 INR)
    predicted_charges_inr = predicted_charges_usd * 83

    st.success(f'Predicted Insurance Charges (USD): **${predicted_charges_usd:,.2f}**')
    st.success(f'Predicted Insurance Charges (INR): **₹{predicted_charges_inr:,.2f}**')
