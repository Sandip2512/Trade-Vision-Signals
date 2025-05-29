import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

# Utility functions
def EMA(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def generate_signals(df):
    df['EMA20'] = EMA(df['close'], 20)
    df['EMA50'] = EMA(df['close'], 50)
    
    df['Buy'] = (df['EMA20'] > df['EMA50']) & (df['EMA20'].shift(1) < df['EMA50'].shift(1))
    df['Sell'] = (df['EMA20'] < df['EMA50']) & (df['EMA20'].shift(1) > df['EMA50'].shift(1))
    return df

@st.cache_data(ttl=300)
def fetch_gold_data(api_key, interval):
    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol=XAU/USD&interval={interval}&outputsize=500&apikey={api_key}"
    )
    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        st.error("Error fetching data or API limit reached.")
        return pd.DataFrame()
    
    df = pd.DataFrame(data["values"])
    df.columns = [col.lower() for col in df.columns]
    df = df[['datetime', 'open', 'high', 'low', 'close']]
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    return df

def plot_signals(df):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name="Price"
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df['EMA20'],
        line=dict(color='blue'), name="EMA20"
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df['EMA50'],
        line=dict(color='orange'), name="EMA50"
    ))

    # Plot Buy signals
    fig.add_trace(go.Scatter(
        x=df[df['Buy']].index,
        y=df[df['Buy']]['low'] * 0.99,
        mode='markers',
        marker=dict(symbol='triangle-up', color='green', size=12),
        name='Buy'
    ))

    # Plot Sell signals
    fig.add_trace(go.Scatter(
        x=df[df['Sell']].index,
        y=df[df['Sell']]['high'] * 1.01,
        mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=12),
        name='Sell'
    ))

    fig.update_layout(
        title="Gold (XAU/USD) Buy & Sell Signals",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        height=700,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

# App UI
st.title("📈 Gold Trading Signals (XAU/USD)")
api_key = st.text_input("🔑 Enter your Twelve Data API Key", type="password")

timeframes = ['5min', '15min', '30min', '1h', '4h', '1day']
selected_tf = st.selectbox("⏱️ Select Timeframe", timeframes)

if api_key:
    df = fetch_gold_data(api_key, selected_tf)
    if not df.empty:
        df = generate_signals(df)
        plot_signals(df)

        # Show recent signal data
        signals = df[df['Buy'] | df['Sell']].copy()
        signals['Signal'] = signals.apply(lambda row: 'Buy' if row['Buy'] else 'Sell', axis=1)
        signals = signals[['open', 'high', 'low', 'close', 'Signal']]
        signals = signals.sort_index(ascending=False).head(10)

        st.subheader("📊 Last 10 Signals")
        st.dataframe(signals)
