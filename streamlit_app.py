import streamlit as st
import pandas as pd
import pickle

# Define file path for the model
file_path = 'oreder(1).pkl'  # Replace with your actual model filename

# Load the trained model
trained_model = None
try:
    with open(file_path, 'rb') as f:
        trained_model = pickle.load(f)
    if isinstance(trained_model, dict):
        trained_model = trained_model.get('model')
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("The model file was not found.")

# Load the encoder
encoder = None
try:
    with open('encoder.pkl', 'rb') as f:  # Keep this if you're using an encoder
        encoder = pickle.load(f)
    st.success("Encoder loaded successfully!")
except FileNotFoundError:
    st.error("The encoder file was not found.")

# Create the Streamlit app layout
st.title("Order Status Prediction")

# User input fields
dif_days_delivered = st.number_input("Days Delivered:", min_value=0)
shipping_charges = st.number_input("Shipping Charges:", min_value=0.0)
product_category_name = st.selectbox("Product Category:", options=['Electronics', 'Books', 'Clothing'])  # Adjust categories
payment_installments = st.number_input("Payment Installments:", min_value=1)
REV_Tax_percent_log = st.number_input("Revenue Tax Percent (Log):", value=0.1)
REV_Tax_log = st.number_input("Revenue Tax (Log):", value=5.0)
order_estimated_delivery_day = st.number_input("Estimated Delivery Day:", min_value=1)
product_height_cm = st.number_input("Product Height (cm):", min_value=0.0)
payment_value_log = st.number_input("Payment Value (Log):", value=100.0)
Duration_approved_log = st.number_input("Duration Approved (Log):", value=2.5)
product_weight_g_log = st.number_input("Product Weight (g, Log):", value=0.5)
product_volume_log = st.number_input("Product Volume (Log):", value=1.0)
REV_Bank_fees_percent = st.number_input("Revenue Bank Fees Percent:", value=2.0)

# Create a button to make the prediction
if st.button("Predict"):
    # Check if the model and encoder were loaded successfully
    if trained_model is not None and encoder is not None:
        # Create a DataFrame from user input
        sample_input = {
            'dif_days_delivered': dif_days_delivered,
            'shipping_charges': shipping_charges,
            'product_category_name': product_category_name,
            'payment_installments': payment_installments,
            'REV_Tax_percent_log': REV_Tax_percent_log,
            'REV_Tax_log': REV_Tax_log,
            'order_estimated_delivery_day': order_estimated_delivery_day,
            'product_height_cm': product_height_cm,
            'payment_value_log': payment_value_log,
            'Duration_approved_log': Duration_approved_log,
            'product_weight_g_log': product_weight_g_log,
            'product_volume_log': product_volume_log,
            'REV_Bank_fees_percent': REV_Bank_fees_percent
        }

        input_data = pd.DataFrame([sample_input])

        # Encode categorical features
        encoded_features = encoder.transform(input_data[['product_category_name']])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())
        input_data_encoded = pd.concat([input_data.drop('product_category_name', axis=1), encoded_df], axis=1)

        # Make predictions using the trained model
        try:
            predictions = trained_model.predict(input_data_encoded)
            st.success(f"Predicted Order Status: {predictions[0]}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("Model and encoder must be loaded before making predictions.")
