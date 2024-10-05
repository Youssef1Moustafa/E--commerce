import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier  # or whichever model you're using

# Load your trained model and label encoders
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('delivered_days.pkl', 'rb') as f:
    label_encoder_customer_state = pickle.load(f)

# Define a function to preprocess the input data
def preprocess_input_data(input_data):
    # Apply log transformation
    input_data['product_volume_log'] = np.log1p(input_data['product_volume'])
    input_data['Duration_approved_log'] = np.log1p(input_data['Duration_approved'])
    input_data['product_weight_g_log'] = np.log1p(input_data['product_weight_g'])
    input_data['REV_Tax_percent_log'] = np.log1p(input_data['REV_Tax_percent'])
    input_data['Revenue_log'] = np.log1p(input_data['Revenue'])
    input_data['REV_gift_log'] = np.log1p(input_data['REV_gift'])
    input_data['REV_Tax_log'] = np.log1p(input_data['REV_Tax'])
    input_data['REV_Bank_fees_log'] = np.log1p(input_data['REV_Bank_fees'])
    input_data['price_log'] = np.log1p(input_data['price'])
    input_data['payment_value_log'] = np.log1p(input_data['payment_value'])

    # Encode categorical variables
    input_data['customer_state'] = label_encoder_customer_state.transform(input_data['customer_state'])
    # Add other label encodings as necessary...

    return input_data

# Streamlit user inputs
st.title("Order Status Prediction")
product_volume = st.number_input("Product Volume")
duration_approved = st.number_input("Duration Approved")
product_weight_g = st.number_input("Product Weight (g)")
rev_tax_percent = st.number_input("Revenue Tax Percentage")
revenue = st.number_input("Revenue")
rev_gift = st.number_input("Revenue Gift")
rev_tax = st.number_input("Revenue Tax")
rev_bank_fees = st.number_input("Revenue Bank Fees")
price = st.number_input("Price")
payment_value = st.number_input("Payment Value")
customer_state = st.selectbox("Customer State", options=["State1", "State2"])  # Replace with actual states

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'product_volume': [product_volume],
    'Duration_approved': [duration_approved],
    'product_weight_g': [product_weight_g],
    'REV_Tax_percent': [rev_tax_percent],
    'Revenue': [revenue],
    'REV_gift': [rev_gift],
    'REV_Tax': [rev_tax],
    'REV_Bank_fees': [rev_bank_fees],
    'price': [price],
    'payment_value': [payment_value],
    'customer_state': [customer_state]
})

# Preprocess the input data
processed_input = preprocess_input_data(input_data)

# Make predictions
predictions = model.predict(processed_input)

# Map numerical predictions to class labels
label_mapping = {0: 'L', 1: 'M', 2: 'S'}
predicted_labels = [label_mapping[pred] for pred in predictions]

# Display the predictions
st.write("Predicted Order Status:")
st.write(predicted_labels)
