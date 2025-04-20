
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime

# Load models
@st.cache_resource
def load_model(sector):
    if sector == "Commercial":
        return joblib.load("commercial_model.pkl")
    else:
        return joblib.load("industrial_model.pkl")

# Title
st.title("⚡ Energy Consumption Forecast")

# Sidebar Inputs
sector = st.selectbox("Select Sector", ["Commercial", "Industrial"])
months_to_forecast = st.selectbox("Forecast Period (months)", [3, 6, 9, 12])

# Forecast Button
if st.button("Forecast"):
    model = load_model(sector)
    series = pd.Series(model.data.endog)
    series.index = model.data.dates
    last_date = series.index[-1]

    # Extend the series up to the current month if needed
    today = pd.to_datetime(datetime.today().strftime('%Y-%m-01'))
    months_gap = (today.year - last_date.year) * 12 + (today.month - last_date.month)

    extended_series = series.copy()
    for i in range(1, months_gap + 1):
        gap_month = last_date + pd.DateOffset(months=i)
        extended_series[gap_month] = np.nan

    final_last_date = extended_series.index[-1]

    # Get model order info
    order = model.model.order
    seasonal_order = model.model.seasonal_order

    # Refit model with extended series
    new_model = SARIMAX(
        extended_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    new_result = new_model.filter(model.params)

    # Forecast with confidence intervals
    forecast_obj = new_result.get_forecast(steps=months_to_forecast)
    forecast_index = [final_last_date + pd.DateOffset(months=i+1) for i in range(months_to_forecast)]
    forecast = pd.Series(forecast_obj.predicted_mean, index=forecast_index)
    conf_int = forecast_obj.conf_int()
    conf_int.index = forecast_index

    # Display forecast table
    st.subheader("Forecasted Consumption")
    forecast_df = pd.DataFrame({
        "Forecast": forecast.round().astype(int),
        "Lower CI": conf_int.iloc[:, 0].round().astype(int),
        "Upper CI": conf_int.iloc[:, 1].round().astype(int),
    })
    st.dataframe(forecast_df)

    # Plot
    recent_actual = extended_series.dropna().iloc[-9:].round().astype(int)   
    fig = go.Figure()    
    fig.add_trace(go.Scatter(x=recent_actual.index, y=recent_actual, mode='lines+markers', name='Last 12 Months Actual'))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines+markers', name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast.index, y=conf_int.iloc[:, 0], mode='lines', name='Lower CI', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=forecast.index, y=conf_int.iloc[:, 1], mode='lines', name='Upper CI', line=dict(dash='dot')))
    fig.update_layout(title="Forecast from Current Month Onward", xaxis_title="Date", yaxis_title="Consumption")
    st.plotly_chart(fig)
