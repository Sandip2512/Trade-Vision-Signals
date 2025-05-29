import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go

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
        f"symbol={symbol}&interval={interval}&apikey={api_key}&format=json&outputsize=500"
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

# --- Generate Buy/Sell Signals ---
def generate_signals(df):
    df['EMA_fast'] = EMA(df['close'], period=9)
    df['EMA_slow'] = EMA(df['close'], period=21)
    df['RSI'] = RSI(df['close'], period=14)

    df['Buy'] = (df['EMA_fast'] > df['EMA_slow']) & (df['EMA_fast'].shift(1) <= df['EMA_slow'].shift(1)) & (df['RSI'] < 70)
    df['Sell'] = (df['EMA_fast'] < df['EMA_slow']) & (df['EMA_fast'].shift(1) >= df['EMA_slow'].shift(1)) & (df['RSI'] > 30)
    return df

# --- Plot candlestick + buy/sell signals ---
def plot_candlestick_with_signals(df):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Gold Price'
    )])

    # Plot Buy signals
    buys = df[df['Buy']]
    fig.add_trace(go.Scatter(
        x=buys.index,
        y=buys['close'],
        mode='markers',
        marker=dict(symbol='triangle-up', color='green', size=12),
        name='Buy Signal'
    ))

    # Plot Sell signals
    sells = df[df['Sell']]
    fig.add_trace(go.Scatter(
        x=sells.index,
        y=sells['close'],
        mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=12),
        name='Sell Signal'
    ))

    fig.update_layout(
        title='Gold (XAU/USD) Intraday Price with Buy/Sell Signals',
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )

    return fig

# --- Streamlit UI ---
st.title("Gold (XAU/USD) Intraday with Buy/Sell Signals and Chart")

api_key = st.text_input("Enter your Twelve Data API Key for Gold", type="password")
interval = st.selectbox("Select Interval", ["15min", "30min", "1h"])

if api_key:
    df = fetch_gold_intraday(api_key, interval=interval)
    if not df.empty:
        df = generate_signals(df)

        st.subheader("Latest Data with Signals")
        st.dataframe(df.tail(20))

        st.subheader("Price Chart with Buy/Sell Signals")
        fig = plot_candlestick_with_signals(df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Failed to fetch Gold data. Please check your API key or interval.")
else:
    st.info("Please enter your Twelve Data API Key to fetch Gold data.")
