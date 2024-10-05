import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the label encoder
def load_label_encoder(encoder_path):
    try:
        with open(encoder_path, 'rb') as f:
            return pickle.load(f)  # Ensure this returns a LabelEncoder instance
    except FileNotFoundError:
        st.error("Label encoder file not found. Please check the path.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading label encoder: {str(e)}")
        st.stop()

# Preprocess input data
def preprocess_input_data(input_data, label_encoder_customer_state):
    # Check if the required column is present
    if 'customer_state' not in input_data.columns:
        st.error("Input data does not contain the 'customer_state' column.")
        return None

    # Transform the customer_state with error handling
    try:
        input_data['customer_state'] = label_encoder_customer_state.transform(input_data['customer_state'])
    except Exception as e:
        st.error(f"Error during label encoding: {str(e)}")
        return None

    # Add any additional preprocessing here if needed
    return input_data

# Specify the path for the label encoder
encoder_path = 'delivered_days.pkl'  # Update this path as necessary
label_encoder_customer_state = load_label_encoder(encoder_path)

# Ensure the loaded object is an instance of LabelEncoder
if not isinstance(label_encoder_customer_state, LabelEncoder):
    st.error("The loaded object is not a LabelEncoder.")
    st.stop()

# Streamlit UI elements
st.title("E-commerce Prediction App")

# Example input data (replace this with your actual input method)
input_data = pd.DataFrame({
    'customer_state': ['State1', 'State2'],  # Example states
    # Add other features needed for prediction
})

# Preprocess the input data
processed_input = preprocess_input_data(input_data, label_encoder_customer_state)

if processed_input is not None:
    # Here, add the code to perform predictions using your model
    # model.predict(processed_input)
    st.write("Processed Input Data:")
    st.dataframe(processed_input)
