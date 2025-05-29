import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# EMA, MACD, RSI functions (same as before)
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

# CoinGecko API for crypto OHLC (unchanged)
@st.cache_data(ttl=300)
def fetch_ohlc_coin_gecko(coin_id, vs_currency='usd', days=1, interval='hourly'):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency={vs_currency}&days={days}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    else:
        st.error(f"Failed to fetch data from CoinGecko: {response.status_code}")
        return pd.DataFrame()

# Update: use your API key here
API_KEY = "LQM2XB6YQ8U6YOQY"

@st.cache_data(ttl=300)
def fetch_forex_data(pair='EURUSD', interval='15min', days=5):
    function_map = {
        '15min': 'FX_INTRADAY',
        '60min': 'FX_INTRADAY',
        '240min': 'FX_INTRADAY',
        '1day': 'FX_DAILY'
    }
    
    if interval not in function_map:
        st.error("Unsupported interval for Forex data.")
        return pd.DataFrame()
    
    base_currency = pair[:3]
    quote_currency = pair[3:]
    
    # Alpha Vantage expects intervals exactly as 15min, 60min, 240min etc. for intraday
    # For daily data interval param is not needed
    interval_param = interval if interval != '1day' else ''
    
    url = f"https://www.alphavantage.co/query?function={function_map[interval]}&from_symbol={base_currency}&to_symbol={quote_currency}"
    if interval_param:
        url += f"&interval={interval_param}"
    url += f"&outputsize=full&apikey={API_KEY}"
    
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Error fetching Forex data from Alpha Vantage.")
        return pd.DataFrame()
    
    data_json = response.json()
    if 'Error Message' in data_json:
        st.error(f"API Error: {data_json['Error Message']}")
        return pd.DataFrame()
    if 'Note' in data_json:
        st.warning(f"API Note: {data_json['Note']}")
        return pd.DataFrame()
    
    time_series_key = None
    for key in data_json.keys():
        if 'Time Series' in key:
            time_series_key = key
            break
    if not time_series_key:
        st.error("Unexpected API response format.")
        return pd.DataFrame()
    
    ts = data_json[time_series_key]
    df = pd.DataFrame.from_dict(ts, orient='index')
    df.columns = ['open', 'high', 'low', 'close']
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

# Signals and plotting code remain the same

# ... rest of your code ...

