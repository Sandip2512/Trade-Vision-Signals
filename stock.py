import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

# --- Indicators ---
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
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Fetch GOLD data from TwelveData ---
@st.cache_data(ttl=300)
def fetch_gold_intraday(api_key, interval):
    symbol = "XAU/USD"
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=500&apikey={api_key}"
    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        st.error("API Error or Invalid API Key.")
        return pd.DataFrame()

    df = pd.DataFrame(data["values"])
    df.columns = [col.lower() for col in df.columns]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df.astype(float)
    df.sort_index(inplace=True)
    return df

# --- Generate Buy/Sell Signals ---
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

# --- Signal Accuracy Estimator ---
def calculate_signal_accuracy(df):
    correct = 0
    total = 0
    for i in range(len(df) - 3):
        if df.iloc[i]['Buy']:
            total += 1
            if df['close'].iloc[i+3] > df['close'].iloc[i]:
                correct += 1
        elif df.iloc[i]['Sell']:
            total += 1
            if df['close'].iloc[i+3] < df['close'].iloc[i]:
                correct += 1
    return (correct / total) * 100 if total > 0 else 0

# --- Plotting ---
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
        marker=dict(symbol='triangle-up', color='green', size=10)
    ))

    fig.add_trace(go.Scatter(
        x=sells.index,
        y=sells['low'] * 0.99,
        mode='markers',
        name='Sell Signal',
        marker=dict(symbol='triangle-down', color='red', size=10)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        template='plotly_white',
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Streamlit UI ---
st.title("📈 Gold Buy/Sell Signal Generator (Intraday & Daily)")

api_key = st.text_input("🔑 Enter your Twelve Data API Key for Gold", type="password")

timeframe_map = {
    "5 min": "5min",
    "15 min": "15min",
    "30 min": "30min",
    "1 Hour": "1h",
    "4 Hour": "4h",
    "1 Day": "1day"
}
selected_tf = st.selectbox("⏱️ Select Timeframe", list(timeframe_map.keys()))
interval = timeframe_map[selected_tf]

if api_key:
    df = fetch_gold_intraday(api_key, interval)
    if not df.empty:
        df = generate_signals(df)
        plot_signals(df, title=f"XAU/USD ({interval}) with Buy/Sell Signals")
        accuracy = calculate_signal_accuracy(df)
        st.success(f"✅ Signal Accuracy Estimate: {accuracy:.2f}%")
        st.dataframe(df.tail(10))
else:
    st.warning("Please enter a valid Twelve Data API Key.")
