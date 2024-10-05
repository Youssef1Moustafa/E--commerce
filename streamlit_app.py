import os
import streamlit as st
import pickle
import numpy as np

# File path to your model and features
file_path = 'model_features.pkl'

# Check if the file exists
if os.path.exists(file_path):
    with open(file_path, 'rb') as file:
        model_data = pickle.load(file)
else:
    st.error(f"File {file_path} not found. Please ensure the file is in the correct location.")
    st.stop()

# Extract the model and feature names
model = model_data['model']
features = model_data['features']

# Title of the page
st.title("Order Status Prediction")

# Create a sidebar navigation for feature input
st.sidebar.title("Navigation")

# A dictionary to store user input for each feature
input_data = {}

# Define the form for taking feature inputs
with st.form("feature_input_form"):
    
    # Feature 1: 'customer_state'
    st.subheader('Customer Information')
    customer_state = st.selectbox("Customer State", ['NY', 'CA', 'TX', 'FL', 'Other'])
    input_data['customer_state'] = customer_state

    customer_city = st.text_input("Customer City", "Enter the city")
    input_data['customer_city'] = customer_city

    # Feature 2: 'order_purchase_year', 'order_purchase_month', 'order_purchase_day'
    st.subheader('Order Details')
    order_purchase_year = st.number_input("Order Purchase Year", min_value=2000, max_value=2023, value=2022)
    order_purchase_month = st.number_input("Order Purchase Month", min_value=1, max_value=12, value=1)
    order_purchase_day = st.number_input("Order Purchase Day", min_value=1, max_value=31, value=1)

    input_data['order_purchase_year'] = order_purchase_year
    input_data['order_purchase_month'] = order_purchase_month
    input_data['order_purchase_day'] = order_purchase_day

    # Feature 3: 'order_estimated_delivery_month', 'order_estimated_delivery_day'
    st.subheader('Estimated Delivery')
    order_estimated_delivery_month = st.number_input("Estimated Delivery Month", min_value=1, max_value=12, value=1)
    order_estimated_delivery_day = st.number_input("Estimated Delivery Day", min_value=1, max_value=31, value=1)

    input_data['order_estimated_delivery_month'] = order_estimated_delivery_month
    input_data['order_estimated_delivery_day'] = order_estimated_delivery_day

    # Feature 4: 'product_category', 'product_category_name'
    st.subheader('Product Information')
    product_category = st.selectbox("Product Category", ['Electronics', 'Apparel', 'Home', 'Books', 'Other'])
    product_category_name = st.text_input("Product Category Name", "Enter the product category name")

    input_data['product_category'] = product_category
    input_data['product_category_name'] = product_category_name

    # Feature 5: 'discount', 'REV_gift_log', 'REV_gift_percent', 'price_log'
    st.subheader('Price & Discount Information')
    discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=10.0)
    rev_gift_log = st.number_input("Revenue Gift Log", min_value=0.0, max_value=10.0, value=1.0)
    rev_gift_percent = st.number_input("Revenue Gift Percent (%)", min_value=0.0, max_value=100.0, value=5.0)
    price_log = st.number_input("Price Log", min_value=0.0, max_value=10.0, value=3.0)

    input_data['discount'] = discount
    input_data['REV_gift_log'] = rev_gift_log
    input_data['REV_gift_percent'] = rev_gift_percent
    input_data['price_log'] = price_log

    # Submit button to make predictions
    submitted = st.form_submit_button("Predict")

# Only proceed with prediction if the form is submitted
if submitted:
    try:
        # Convert the input dictionary to a 2D numpy array (required for prediction)
        input_array = np.array([list(input_data.values())])

        # Make prediction
        prediction = model.predict(input_array)

        # Output prediction results
        st.success(f"Prediction: {'Order Delivered' if prediction[0] == 'delivered' else 'Other Status'}")

    except Exception as e:
        st.error(f"Error making prediction: {e}")
