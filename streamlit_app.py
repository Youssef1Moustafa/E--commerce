import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Load your DataFrame
df = pd.read_csv('your_data_file.csv')  # Replace with your actual data file path

# Load the model and features
@st.cache(allow_output_mutation=True)
def load_model():
    with open('delivered_days.pkl', 'rb') as model_file:
        loaded_model_data = pickle.load(model_file)
    return loaded_model_data['model'], loaded_model_data['features']

# Load model and features
loaded_model, loaded_features = load_model()

st.title("Order Status Prediction")

# Create a dictionary to hold input data
input_data = {}

# Feature 1: 'customer_state'
st.subheader('Customer Information')
customer_state = st.selectbox("Customer State", df['customer_state'].unique().tolist())
input_data['customer_state'] = customer_state

# Continue creating input fields for other features
for feature in loaded_features:
    if feature != 'customer_state':  # Skip if it's already added
        if feature in ['product_category_name']:  # Example for categorical features
            value = st.selectbox(feature, options=["Option 1", "Option 2"])  # Replace with actual options
        else:  # Numerical features
            value = st.number_input(feature, min_value=0.0, value=1.0)  # Adjust min/max values as needed
        input_data[feature] = value

# Convert input_data to a DataFrame to ensure correct types
input_df = pd.DataFrame([input_data])

# Make prediction using the loaded model
if st.button('Predict'):
    try:
        # Ensure categorical features are encoded correctly if needed
        input_df['customer_state'] = input_df['customer_state'].astype(str)  # Convert to string if needed
        # Add more encoding if necessary

        prediction = loaded_model.predict(input_df)  # Use the DataFrame directly for prediction
        st.write("Prediction:", "Delivered" if prediction[0] == 1 else "Not Delivered")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
