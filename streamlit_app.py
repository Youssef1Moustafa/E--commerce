import os
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the model
file_path = 'delivered_days.pkl'

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
st.title("Delivered Days Prediction")

# Create a sidebar navigation for feature input
st.sidebar.title("Navigation")

# Create a dictionary mapping states to cities
city_mapping = {
    'SP': ['São Paulo', 'Campinas', 'Santo André'],
    'RJ': ['Rio de Janeiro', 'Niterói', 'Petropolis'],
    'S': ['cool_stuff', 'garden_tools', 'furniture_decor'],
}

# Create a dictionary mapping product categories to category names
product_mapping = {
    'L': ['toys', 'watches_gifts', 'construction_tools_garden'],
    'M': ['bed_bath_table', 'auto', 'health_beauty'],
    # Add more categories as needed
}

# A dictionary to store user input for each feature
input_data = {}

# Define the form for taking feature inputs
with st.form("feature_input_form"):
    # Customer Information
    st.subheader('Customer Information')
    customer_state = st.selectbox("Customer State", list(city_mapping.keys()))
    input_data['customer_state'] = customer_state

    # Dynamically select cities based on the selected state
    customer_city = st.selectbox("Customer City", city_mapping[customer_state])
    input_data['customer_city'] = customer_city

    # Order Details
    st.subheader('Order Details')
    order_purchase_year = st.number_input("Order Purchase Year", min_value=2017, max_value=2024, value=2017)
    order_purchase_month = st.number_input("Order Purchase Month", min_value=1, max_value=12, value=1)
    order_purchase_day = st.number_input("Order Purchase Day", min_value=1, max_value=31, value=1)

    input_data['order_purchase_year'] = order_purchase_year
    input_data['order_purchase_month'] = order_purchase_month
    input_data['order_purchase_day'] = order_purchase_day

    # Estimated Delivery
    st.subheader('Estimated Delivery')
    order_estimated_delivery_month = st.number_input("Estimated Delivery Month", min_value=1, max_value=12, value=1)
    order_estimated_delivery_day = st.number_input("Estimated Delivery Day", min_value=1, max_value=31, value=1)

    input_data['order_estimated_delivery_month'] = order_estimated_delivery_month
    input_data['order_estimated_delivery_day'] = order_estimated_delivery_day

    # Product Information
    st.subheader('Product Information')
    product_category = st.selectbox("Product Category", list(product_mapping.keys()))
    product_category_name = st.selectbox("Product Category Name", product_mapping[product_category])

    input_data['product_category'] = product_category
    input_data['product_category_name'] = product_category_name

    # Price & Discount Information
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
        # Prepare input for prediction
        input_values = list(input_data.values())

        # Apply Label Encoding on categorical inputs if needed
        label_encoder = LabelEncoder()

        # Transform categorical variables
        input_values[features.index('customer_state')] = label_encoder.fit_transform([input_data['customer_state']])[0]
        input_values[features.index('customer_city')] = label_encoder.fit_transform([input_data['customer_city']])[0]
        input_values[features.index('product_category')] = label_encoder.fit_transform([input_data['product_category']])[0]
        input_values[features.index('product_category_name')] = label_encoder.fit_transform([input_data['product_category_name']])[0]

        # Convert input to a numpy array with correct dtype
        input_array = np.array([input_values], dtype=float)

        # Make prediction
        prediction = model.predict(input_array)

        # Output prediction results
        st.success(f"Prediction: {'Order Delivered' if prediction[0] == 1 else 'Other Status'}")

    except Exception as e:
        st.error(f"Error making prediction: {e}")
