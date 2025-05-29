import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from io import StringIO
import yfinance as yf
from datetime import datetime, timedelta

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

def fetch_coingecko_btc(days=30):
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        prices = data['prices']  # list of [timestamp, price]
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        df = df.drop('timestamp', axis=1)
        # CoinGecko prices are close-like values; we'll assume OHLC = price for simplicity
        df['Open'] = df['price']
        df['High'] = df['price']
        df['Low'] = df['price']
        df['Close'] = df['price']
        return df
    except Exception as e:
        st.error(f"Failed to fetch BTC data from CoinGecko: {e}")
        return pd.DataFrame()

st.title("Gold and BTC Trading Strategy with CoinGecko & Yahoo Finance")

option = st.selectbox("Select Asset", ["Bitcoin (BTC)", "Gold (GC=F)"])

if option == "Bitcoin (BTC)":
    days = st.slider("Select number of days to fetch", 10, 90, 30)
    data = fetch_coingecko_btc(days)
    if data.empty:
        st.stop()
else:
    # Gold futures data using yfinance
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=90))
    end_date = st.date_input("End Date", datetime.today())
    if start_date > end_date:
        st.error("Start date must be before end date")
        st.stop()
    ticker = yf.Ticker("GC=F")
    data = ticker.history(start=start_date, end=end_date)
    if data.empty:
        st.error("No data fetched for Gold")
        st.stop()

# Calculate indicators
data['EMA'] = EMA(data['Close'], 20)
data['MACD'], data['Signal'] = MACD(data['Close'])
data['RSI'] = RSI(data['Close'], 14)

data_clean = data.dropna(subset=['EMA', 'MACD', 'Signal', 'RSI'])

data_clean['Buy'] = (
    (data_clean['Close'] > data_clean['EMA']) &
    (data_clean['MACD'] > data_clean['Signal']) &
    (data_clean['MACD'].shift(1) < data_clean['Signal'].shift(1)) &
    (data_clean['RSI'] < 60)
)
data_clean['Sell'] = (
    (data_clean['Close'] < data_clean['EMA']) &
    (data_clean['MACD'] < data_clean['Signal']) &
    (data_clean['MACD'].shift(1) > data_clean['Signal'].shift(1)) &
    (data_clean['RSI'] > 60)
)

st.subheader(f"{option} Price Chart with Buy/Sell Signals")

buys = data_clean[data_clean['Buy']]
sells = data_clean[data_clean['Sell']]

fig = go.Figure()

fig.add_trace(go.Scatter(x=data_clean.index, y=data_clean['Close'],
                         mode='lines', name='Close Price', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=data_clean.index, y=data_clean['EMA'],
                         mode='lines', name='EMA20', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'],
                         mode='markers', name='Buy Signal',
                         marker=dict(symbol='triangle-up', color='green', size=12)))
fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'],
                         mode='markers', name='Sell Signal',
                         marker=dict(symbol='triangle-down', color='red', size=12)))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    legend=dict(x=0, y=1),
    hovermode='x unified',
    template='plotly_white',
    height=700,
    width=1100,
    xaxis_rangeslider_visible=True
)

st.plotly_chart(fig, use_container_width=True)
st.write("Recent Signal Data:")
st.dataframe(data_clean.tail(10))
