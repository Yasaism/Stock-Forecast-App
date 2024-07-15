import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from math import sqrt

# Constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Streamlit App Title
st.title('Stock Forecast App')

# Select stock for prediction
stocks = ('BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'BTC-USD')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Select number of years for prediction
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Load data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Display raw data
st.subheader('Raw data')
st.write(data.tail())

# Moving average window size slider
ma_window = st.slider('Moving Average Window Size:', 1, 100, 20)

# Calculate moving average
data['Moving_Avg'] = data['Close'].rolling(window=ma_window).mean()

# Plot raw data with moving average
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Moving_Avg'], name="moving_avg", line=dict(color='orange')))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Select forecasting method
method = st.selectbox('Select forecasting method', ('Prophet', 'ARIMA', 'SMA', 'EMA'))

# Prepare data for forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Function to test stationarity
def test_stationarity(timeseries):
    result = adfuller(timeseries)
    st.write('ADF Statistic: %f' % result[0])
    st.write('p-value: %f' % result[1])
    for key, value in result[4].items():
        st.write('Critical Values:')
        st.write(f'   {key}, {value}')

# Forecasting with Prophet
if method == 'Prophet':
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    
    st.subheader('Forecast data (Prophet)')
    st.write(forecast.tail())
    
    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

# Forecasting with ARIMA
elif method == 'ARIMA':
    df_train.set_index('ds', inplace=True)
    st.subheader('ADF Test for Stationarity')
    test_stationarity(df_train['y'])
    
    # Differencing to make series stationary
    df_train_diff = df_train['y'].diff().dropna()
    st.subheader('ADF Test for Differenced Series')
    test_stationarity(df_train_diff)
    
    # ACF and PACF plots for differenced series
    st.subheader('ACF and PACF plots for Differenced Series')
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    plot_acf(df_train_diff, ax=axes[0])
    plot_pacf(df_train_diff, ax=axes[1])
    st.pyplot(fig)
    
    # Model fitting
    p = st.slider('AR (p)', 0, 10, 5)
    d = st.slider('Difference (d)', 0, 2, 1)
    q = st.slider('MA (q)', 0, 10, 0)
    
    model = sm.tsa.ARIMA(df_train['y'], order=(p, d, q))
    model_fit = model.fit()
    
    # Model diagnostics
    st.subheader('Model Diagnostics')
    residuals = model_fit.resid
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    ax[0].plot(residuals)
    ax[0].set_title('Residuals')
    plot_acf(residuals, ax=ax[1])
    ax[1].set_title('Residual ACF')
    st.pyplot(fig)
    
    forecast = model_fit.forecast(steps=period)
    forecast_dates = pd.date_range(start=df_train.index[-1] + timedelta(days=1), periods=period)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
    
    st.subheader('Forecast data (ARIMA)')
    st.write(forecast_df.tail())
    
    st.write(f'Forecast plot for {n_years} years')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train.index, y=df_train['y'], name='Observed'))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], name='Forecast', line=dict(color='red')))
    fig.layout.update(title_text='ARIMA Forecast', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


# Forecasting with SMA
elif method == 'SMA':
    sma_forecast = data['Close'].rolling(window=ma_window).mean().iloc[-1]
    sma_dates = pd.date_range(start=data['Date'].iloc[-1] + timedelta(days=1), periods=period)
    sma_forecast_values = pd.Series([sma_forecast] * len(sma_dates), index=sma_dates)

    st.subheader('Forecast data (SMA)')
    st.write(sma_forecast_values.tail())

    st.write(f'SMA forecast plot for {n_years} years')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Observed'))
    fig.add_trace(go.Scatter(x=sma_forecast_values.index, y=sma_forecast_values, name='SMA Forecast', line=dict(color='green')))
    fig.layout.update(title_text='SMA Forecast', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Forecasting with EMA
elif method == 'EMA':
    ema = data['Close'].ewm(span=ma_window, adjust=False).mean()
    ema_forecast = ema.iloc[-1]
    ema_dates = pd.date_range(start=data['Date'].iloc[-1] + timedelta(days=1), periods=period)
    ema_forecast_values = pd.Series([ema_forecast] * len(ema_dates), index=ema_dates)

    st.subheader('Forecast data (EMA)')
    st.write(ema_forecast_values.tail())

    st.write(f'EMA forecast plot for {n_years} years')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Observed'))
    fig.add_trace(go.Scatter(x=ema_forecast_values.index, y=ema_forecast_values, name='EMA Forecast', line=dict(color='purple')))
    fig.layout.update(title_text='EMA Forecast', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)