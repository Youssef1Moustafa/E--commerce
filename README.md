# Order Status Prediction App

This is a Streamlit-based web application that predicts the order status (e.g., delivered, shipped) using a machine learning model. The model takes several features such as customer state, city, product category, and purchase details to provide predictions.

## Features

The application uses the following features for prediction:

- `customer_state`: State where the customer resides (e.g., NY, CA)
- `customer_city`: City where the customer resides
- `order_purchase_year`: Year when the order was purchased
- `order_purchase_month`: Month of the purchase
- `order_purchase_day`: Day of the purchase
- `order_estimated_delivery_month`: Estimated month for delivery
- `order_estimated_delivery_day`: Estimated day for delivery
- `product_category`: Category of the product (e.g., Electronics, Apparel)
- `product_category_name`: Name of the product category
- `discount`: Discount applied to the order (in percentage)
- `REV_gift_log`: Log of gift revenue for the order
- `REV_gift_percent`: Percentage of revenue from gifts
- `price_log`: Logarithmic value of the product's price

## How to Run the App

### Prerequisites

Make sure you have Python 3.x installed on your machine. You’ll also need the following Python libraries:

- `streamlit`
- `numpy`
- `pickle`
- `sklearn`
- `pandas`

You can install the necessary libraries by running:

```bash
pip install streamlit numpy pickle5 scikit-learn pandas
```

### Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/order-status-prediction-app.git
cd order-status-prediction-app
```

### Running the App

Make sure your machine learning model is saved as `model_features.pkl` and is located in the same directory as `app.py`. Then run the following command to launch the app:

```bash
streamlit run app.py
```

This will start a local web server and open the app in your browser.

### Usage

1. **Customer Information**: Input the customer’s state and city.
2. **Order Details**: Input the purchase year, month, and day for the order.
3. **Estimated Delivery**: Provide the estimated delivery month and day.
4. **Product Information**: Choose the product category and provide the product category name.
5. **Price & Discount**: Input the discount applied to the order, revenue from gifts (log), and the price (logarithmic scale).
6. **Predict**: Click the `Predict` button to get the predicted order status.

## File Structure

- `app.py`: The main application file for Streamlit.
- `model_features.pkl`: This file contains the trained machine learning model and the list of features.
- `README.md`: Documentation for the project.

## Model Training

The machine learning model used in this app is trained using features like customer state, city, order dates, product categories, discount, and price details. The model is saved in a `pickle` file (`model_features.pkl`), which is loaded at runtime.

## Example

Below is an example of the interface, where you can input feature values and predict whether an order will be delivered or not:

![Screenshot of the app](example_screenshot.png)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
