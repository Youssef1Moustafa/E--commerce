import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model and label encoder
# Note: Replace 'model.pkl' with your actual model file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the label encoders for any categorical variables you may need
with open('delivered_days.pkl', 'rb') as f:
    label_encoder_customer_state = pickle.load(f)

# Function to make predictions and convert to labels
def make_predictions(input_data):
    # Assuming input_data is a DataFrame structured the same as your training data
    predictions = model.predict(input_data)  # Get predictions
    label_mapping = {0: 'L', 1: 'M', 2: 'S'}  # Mapping from numerical to class labels
    predicted_labels = [label_mapping[pred] for pred in predictions]  # Convert to labels
    return predicted_labels

# Streamlit UI
st.title("Order Status Prediction")

# User Input
product_volume = st.number_input("Enter product volume:", min_value=0.0)
payment_value = st.number_input("Enter payment value:", min_value=0.0)
# Add additional inputs as needed based on your model's requirements

# Convert user inputs to a DataFrame for prediction
input_data = pd.DataFrame({
    'product_volume': [product_volume],
    'payment_value': [payment_value],
    # Include any other input fields your model requires
})

# Button to make predictions
if st.button("Predict"):
    # Make predictions
    predicted_labels = make_predictions(input_data)
    # Display predictions
    st.write("Predicted Class Labels:")
    st.write(predicted_labels)

    # Optionally, display in a structured format
    df_predictions = pd.DataFrame({
        'Prediction Index': range(len(predicted_labels)),
        'Predicted Class': predicted_labels
    })

    st.table(df_predictions)
