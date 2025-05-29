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
def fetch_forex_data(pair='EURUSD', interval='15min'):
    API_KEY = st.secrets["ALPHAVANTAGE_API_KEY"]

    url = (
        f"https://www.alphavantage.co/query?"
        f"function=FX_INTRADAY&from_symbol={pair[:3]}&to_symbol={pair[3:]}&interval={interval}"
        f"&outputsize=full&apikey={API_KEY}"
    )
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Error fetching Forex data from Alpha Vantage.")
        return pd.DataFrame()
    
    data_json = response.json()
    st.write(data_json)  # Debug output to check raw API response

    if 'Error Message' in data_json:
        st.error(f"API Error: {data_json['Error Message']}")
        return pd.DataFrame()
    
    if 'Note' in data_json:
        st.error(f"API Note: {data_json['Note']}")
        return pd.DataFrame()

    time_series_key = next((k for k in data_json if 'Time Series' in k), None)
    if not time_series_key:
        st.error("Unexpected API response format. No 'Time Series' data found.")
        return pd.DataFrame()
    
    ts = data_json[time_series_key]
    df = pd.DataFrame.from_dict(ts, orient='index')
    df.columns = ['open', 'high', 'low', 'close']
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def generate_signals(df):
    df['EMA'] = EMA(df['close'])
    df['MACD'], df['Signal'] = MACD(df['close'])
    df['RSI'] = RSI(df['close'])
    df.dropna(inplace=True)

    df['Buy'] = (
        (df['close'] > df['EMA']) &
        (df['MACD'] > df['Signal']) &
        (df['MACD'].shift(1) < df['Signal'].shift(1)) &
        (df['RSI'] < 60)
    )
    df['Sell'] = (
        (df['close'] < df['EMA']) &
        (df['MACD'] < df['Signal']) &
        (df['MACD'].shift(1) > df['Signal'].shift(1)) &
        (df['RSI'] > 60)
    )
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
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

# Streamlit UI
st.title("Forex Trading Signals")

forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']
pair = st.selectbox("Select Forex Pair", forex_pairs)
timeframe = st.selectbox("Select Timeframe", ['15min'])  # Simplified to 15min for now

df = fetch_forex_data(pair=pair, interval=timeframe)
if not df.empty:
    df = generate_signals(df)
    plot_signals(df, title=f"{pair} Price with Buy/Sell Signals ({timeframe})")
    st.dataframe(df.tail(10))
else:
    st.warning("No data available or API limit reached.")
