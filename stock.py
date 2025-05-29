import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# Your Alpha Vantage API key here
API_KEY = "LQM2XB6YQ8U6YOQY"

# Define Forex pairs supported by Alpha Vantage
FOREX_PAIRS = {
    "EUR/USD": ("EUR", "USD"),
    "GBP/USD": ("GBP", "USD"),
    "USD/JPY": ("USD", "JPY"),
    "AUD/USD": ("AUD", "USD"),
    "USD/CAD": ("USD", "CAD"),
    "USD/CHF": ("USD", "CHF"),
    "NZD/USD": ("NZD", "USD"),
}

TIMEFRAMES = {
    "15min": "15min",
    "1h": "60min",
    "4h": "60min",  # Alpha Vantage does not support 4h directly, we'll fetch 60min and resample
}

def fetch_forex_data(from_symbol, to_symbol, interval="15min"):
    url = (
        f"https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={from_symbol}"
        f"&to_symbol={to_symbol}&interval={interval}&apikey={API_KEY}&outputsize=full"
    )
    response = requests.get(url)
    data = response.json()

    # Key in response
    key = f"Time Series FX ({interval})"
    if key not in data:
        st.error("Error fetching data: " + str(data.get("Error Message", "API limit exceeded or bad request")))
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(data[key], orient='index')
    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close"
    })

    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def resample_4h(df):
    df_4h = df.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()
    return df_4h

def EMA(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    RS = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + RS))
    return rsi

def generate_signals(df):
    df['EMA20'] = EMA(df['Close'], 20)
    df['EMA50'] = EMA(df['Close'], 50)
    df['MACD'], df['Signal'] = MACD(df['Close'])
    df['RSI'] = RSI(df['Close'], 14)

    df['Buy'] = (
        (df['EMA20'] > df['EMA50']) &
        (df['MACD'] > df['Signal']) &
        (df['RSI'] < 70)
    )
    df['Sell'] = (
        (df['EMA20'] < df['EMA50']) &
        (df['MACD'] < df['Signal']) &
        (df['RSI'] > 30)
    )
    return df

def plot_chart(df):
    buys = df[df['Buy']]
    sells = df[df['Sell']]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['EMA20'], mode='lines', name='EMA20', line=dict(color='orange')))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['EMA50'], mode='lines', name='EMA50', line=dict(color='purple')))

    fig.add_trace(go.Scatter(
        x=buys.index, y=buys['Close'],
        mode='markers', name='Buy Signal',
        marker=dict(symbol='triangle-up', color='green', size=12)))
    fig.add_trace(go.Scatter(
        x=sells.index, y=sells['Close'],
        mode='markers', name='Sell Signal',
        marker=dict(symbol='triangle-down', color='red', size=12)))

    fig.update_layout(
        title="Forex Price with Buy/Sell Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=700,
        width=1100,
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

# Streamlit UI

st.title("Forex Price Action Strategy with Buy/Sell Signals")

pair = st.selectbox("Select Forex Pair", list(FOREX_PAIRS.keys()))
timeframe = st.selectbox("Select Timeframe", ["15min", "1h", "4h"])

from_symbol, to_symbol = FOREX_PAIRS[pair]

with st.spinner("Fetching data..."):
    df = fetch_forex_data(from_symbol, to_symbol, interval=TIMEFRAMES[timeframe])
    if df.empty:
        st.stop()
    if timeframe == "4h":
        df = resample_4h(df)

df = generate_signals(df)
plot_chart(df)

st.subheader("Recent Buy/Sell Signals")
st.write(df[['Open', 'High', 'Low', 'Close', 'EMA20', 'EMA50', 'MACD', 'Signal', 'RSI', 'Buy', 'Sell']].tail(10))
