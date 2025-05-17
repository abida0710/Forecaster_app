import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import load_model
import pickle
import os
import xgboost as xgb
#import gdown


# Load the trained XGBoost model
@st.cache_resource
def load_model():
    with open('./saved_models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Load the dataset and apply train-test split
@st.cache_data
def load_data():
    data = pd.read_csv('./data/train_final.csv')
    data['date'] = pd.to_datetime(data['date'])  # Ensure 'date' is in datetime format
    split_date = '2014-01-01'
    #train = data[data['date'] < split_date]  # Training data (not used in the app)
    test = data[data['date'] >= split_date]  # Test data for predictions
    return test

# Load model and test data
model = load_model()
test_data = load_data()

# Sidebar for user inputs
st.sidebar.title("Sales Forecaster")
st.sidebar.markdown("---")
st.sidebar.write("### How it works:")
st.sidebar.write("1. Select a store.")
st.sidebar.write("2. Select an item (filtered based on the selected store).")
st.sidebar.write("3. Choose a month to view its prediction data.")
st.sidebar.markdown("---")

# User inputs
store_nbr = st.sidebar.selectbox("Select Store Number", test_data['store_nbr'].unique())

# Dynamically filter items based on the selected store
filtered_items = test_data[test_data['store_nbr'] == store_nbr]['item_nbr'].unique()
item_nbr = st.sidebar.selectbox("Select Item Number", filtered_items)

# Dynamically determine available months
month_dict = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
              7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
available_months = [1, 2, 3]  # Only three months for prediction
available_month_names = [month_dict[m] for m in available_months]
selected_month = st.sidebar.selectbox("Select Month to View Prediction Data:", available_month_names)

# Map selected month back to numerical value
selected_month_number = {v: k for k, v in month_dict.items()}[selected_month]

# Filter data for the selected store and item
filtered_data = test_data[(test_data['store_nbr'] == store_nbr) & (test_data['item_nbr'] == item_nbr)]

# Ensure filtered data is not empty
if filtered_data.empty:
    st.warning("No data available for the selected store and item.")
else:
    # Prepare data for forecasting
    filtered_data = filtered_data[filtered_data['date'].dt.month.isin(available_months)]
    if filtered_data.empty:
        st.warning("No data available for the selected months.")
    else:
        # Prepare features for prediction
        required_features = [
            'store_nbr', 'item_nbr', 'onpromotion', 'year', 'month', 'day', 'day_of_week',
            'unit_sales_7d_avg', 'perishable', 'oil_price', 'transactions', 'is_holiday',
            'week', 'is_weekend', 'lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_std_7'
        ]
        X = filtered_data[required_features]

        # Make predictions
        filtered_data['Predicted Sales'] = model.predict(X)

        # Display predicted sales for the selected month in the sidebar
        st.sidebar.markdown("### Predicted Sales:")
        for month in available_months:
            if month <= selected_month_number:
                month_data = filtered_data[filtered_data['date'].dt.month == month]
                predicted_sales = month_data['Predicted Sales'].sum()
                st.sidebar.write(f"{month_dict[month]}'s predicted sales = {predicted_sales:.0f}")

        # Line Chart: Predicted Sales for Three Months
        st.title("Sales Forecast")
        st.markdown(f"### Predictions for Store {store_nbr} and Item {item_nbr}")
        st.markdown("### Predicted Sales for Three Months")
        plt.figure(figsize=(10, 6))
        for month in available_months:
            month_data = filtered_data[filtered_data['date'].dt.month == month]
            plt.plot(month_data['date'], month_data['Predicted Sales'], marker='o', label=f"{month_dict[month]}")
        plt.xlabel("Date")
        plt.ylabel("Predicted Sales")
        plt.title("Predicted Sales for Three Months")
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)

        # Table: Display prediction data for the selected month
        st.markdown(f"### Prediction Data for {selected_month}")
        selected_month_data = filtered_data[filtered_data['date'].dt.month == selected_month_number]
        if selected_month_data.empty:
            st.warning(f"No data available for {selected_month}.")
        else:
            table_data = selected_month_data[['date', 'month', 'Predicted Sales']].copy()
            table_data['date'] = table_data['date'].dt.strftime('%Y-%m-%d')  # Format the date
            table_data['month'] = table_data['month'].map(month_dict)
            table_data.rename(columns={'month': 'Month Name'}, inplace=True)
            st.dataframe(table_data)