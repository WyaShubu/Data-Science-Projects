import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import base64

# Function to load data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Function to preprocess data
def preprocess_data(data, timestamp_col, consumption_col):
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data.set_index(timestamp_col, inplace=True)
    data['day_of_week'] = data.index.dayofweek
    data['hour'] = data.index.hour
    # Additional features
    data['month'] = data.index.month
    data['quarter'] = data.index.quarter
    
    # Handle missing values
    data.fillna(method='ffill', inplace=True)
    
    X = data[['day_of_week', 'hour', 'month', 'quarter']]
    y = data[consumption_col]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to perform grid search for XGB Regressor
def train_xgb(X_train, y_train):
    model = XGBRegressor(random_state=42)
    params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Function to perform grid search for LightGBM model
def train_lgbm(X_train, y_train):
    model = LGBMRegressor(random_state=42)
    params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [-1, 5, 7]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Function to forecast
def forecast(model, X_test):
    return model.predict(X_test)

# Streamlit App
st.title("Energy Consumption Forecasting")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("Data Overview")
    st.write(data.head())
    
    timestamp_col = st.sidebar.selectbox("Select Timestamp Column", data.columns)
    consumption_col = st.sidebar.selectbox("Select Consumption Column", data.columns)
    
    # Initialize session state variables
    if 'model' not in st.session_state:
        st.session_state.model = None

    model_type = st.sidebar.selectbox("Select Model", ["XGB Regressor", "LightGBM"])
    
    if st.sidebar.button("Train Model"):
        X_train, X_test, y_train, y_test = preprocess_data(data, timestamp_col, consumption_col)
        if model_type == "XGB Regressor":
            st.session_state.model = train_xgb(X_train, y_train)
        elif model_type == "LightGBM":
            st.session_state.model = train_lgbm(X_train, y_train)
        y_pred = forecast(st.session_state.model, X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        st.write(f"Model Performance:\nMean Squared Error: {mse}\nMean Absolute Error: {mae}\nMean Absolute Percentage Error: {mape}")

        fig, ax = plt.subplots()
        ax.plot(y_test.values, label='Actual')
        ax.plot(y_pred, label='Forecast')
        ax.legend()
        st.pyplot(fig)

    if st.session_state.model is not None:
        forecast_hours = st.sidebar.number_input("Number of hours to forecast", min_value=1, max_value=24, value=1)
        
        future_dates = pd.date_range(start=data.index[-1], periods=forecast_hours + 1, freq='H')[1:]
        future_data = pd.DataFrame(index=future_dates)
        future_data['day_of_week'] = future_data.index.dayofweek
        future_data['hour'] = future_data.index.hour
        future_data['month'] = future_data.index.month
        future_data['quarter'] = future_data.index.quarter
        
        scaler = StandardScaler()
        future_data_scaled = scaler.fit_transform(future_data)
        future_forecast = forecast(st.session_state.model, future_data_scaled)
        
        st.write("Future Forecast")
        st.write(future_forecast)

        fig, ax = plt.subplots()
        ax.plot(future_dates, future_forecast, label='Forecast')
        ax.legend()
        st.pyplot(fig)
