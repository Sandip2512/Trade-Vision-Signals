import streamlit as st
import pandas as pd
import requests
import numpy as np

# --- Indicator Functions ---

def EMA(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    RS = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + RS))
    return rsi

# --- Fetch Gold Intraday from Twelve Data ---

@st.cache_data(ttl=300)
def fetch_gold_intraday(api_key, interval='15min'):
    symbol = "XAU/USD"
    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol={symbol}&interval={interval}&apikey={api_key}&format=json"
    )
    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        st.error(f"Gold API error: {data.get('message', 'No data returned')}")
        return pd.DataFrame()

    df = pd.DataFrame(data["values"])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = df[col].astype(float)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    return df

# --- Fetch BTC Intraday from Binance ---

@st.cache_data(ttl=300)
def fetch_btc_binance(interval='15m', limit=500):
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()

    if not isinstance(data, list):
        st.error(f"Binance API error: {data.get('msg', 'No data returned')}")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df.sort_index(inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']]

# --- Generate Buy/Sell Signals ---

def generate_signals(df):
    df['EMA_fast'] = EMA(df['close'], period=9)
    df['EMA_slow'] = EMA(df['close'], period=21)
    df['RSI'] = RSI(df['close'], period=14)

    df['Buy'] = (df['EMA_fast'] > df['EMA_slow']) & (df['EMA_fast'].shift(1) <= df['EMA_slow'].shift(1)) & (df['RSI'] < 70)
    df['Sell'] = (df['EMA_fast'] < df['EMA_slow']) & (df['EMA_fast'].shift(1) >= df['EMA_slow'].shift(1)) & (df['RSI'] > 30)
    return df

# --- Streamlit UI ---

st.title("Gold & BTC Intraday with Buy/Sell Signals")

api_key = st.text_input("Enter your Twelve Data API Key for Gold", type="password")

asset = st.selectbox("Select Asset", ["Gold (XAU/USD)", "Bitcoin (BTCUSDT)"])
interval = st.selectbox("Select Interval", ["15min", "30min", "1h"])

if asset == "Gold (XAU/USD)":
    if api_key:
        df = fetch_gold_intraday(api_key, interval=interval)
        if not df.empty:
            df = generate_signals(df)
            st.write(f"Showing {asset} intraday data with signals")
            st.dataframe(df.tail(20))
        else:
            st.warning("No data fetched for Gold. Check API key or usage limits.")
    else:
        st.info("Please enter your API key to fetch Gold data.")
else:
    df = fetch_btc_binance(interval=interval.replace('min', 'm'))
    if not df.empty:
        df = generate_signals(df)
        st.write(f"Showing {asset} intraday data with signals")
        st.dataframe(df.tail(20))
    else:
        st.warning("No data fetched for BTC.")

# Optional: You can add plotly chart visualization with buy/sell markers too if needed.
