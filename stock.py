import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

# EMA, MACD, RSI functions
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

@st.cache_data(ttl=300)
def fetch_gold_intraday(api_key, symbol="XAU/USD", interval="15min"):
    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol={symbol}&interval={interval}&apikey={api_key}&format=JSON&outputsize=500"
    )
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Error fetching Gold data from Twelve Data.")
        return pd.DataFrame()
    data_json = response.json()
    if "status" in data_json and data_json["status"] == "error":
        st.error(f"API Error: {data_json.get('message', 'Unknown error')}")
        return pd.DataFrame()
    if "values" not in data_json:
        st.error("Unexpected API response format. No 'values' found.")
        return pd.DataFrame()

    df = pd.DataFrame(data_json["values"])
    df = df.rename(columns={
        "datetime": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume"
    })
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    df.set_index('date', inplace=True)
    return df

def generate_signals(df):
    df['EMA'] = EMA(df['close'])
    df['MACD'], df['Signal'] = MACD(df['close'])
    df['RSI'] = RSI(df['close'])
    df.dropna(inplace=True)

    # Simplified buy/sell signals to ensure signals appear:
    df['Buy'] = (df['MACD'] > df['Signal']) & (df['RSI'] < 70)
    df['Sell'] = (df['MACD'] < df['Signal']) & (df['RSI'] > 30)
    return df

def plot_signals(df, title):
    buys = df[df['Buy']]
    sells = df[df['Sell']]

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    )])

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EMA'],
        mode='lines',
        name='EMA',
        line=dict(color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=buys.index,
        y=buys['high'] * 1.01,
        mode='markers',
        name='Buy Signal',
        marker=dict(symbol='triangle-up', color='green', size=12)
    ))

    fig.add_trace(go.Scatter(
        x=sells.index,
        y=sells['low'] * 0.99,
        mode='markers',
        name='Sell Signal',
        marker=dict(symbol='triangle-down', color='red', size=12)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        template='plotly_white',
        height=700,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

# Streamlit UI
st.title("Gold Intraday Trading Signals")

api_key = st.text_input("Enter your Twelve Data API Key for Gold", type="password")
if not api_key:
    st.warning("Please enter your API key to fetch data.")
else:
    gold_df = fetch_gold_intraday(api_key)
    if not gold_df.empty:
        gold_df = generate_signals(gold_df)

        # Debug info - show counts of signals
        st.write(f"Buy signals count: {gold_df['Buy'].sum()}")
        st.write(f"Sell signals count: {gold_df['Sell'].sum()}")

        plot_signals(gold_df, title="Gold Price with Buy/Sell Signals (15min)")
        st.dataframe(gold_df.tail(10))
    else:
        st.warning("No data available or API limit reached.")
