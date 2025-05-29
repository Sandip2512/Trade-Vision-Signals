import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

# ====== Indicators ======
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
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ====== Data Fetching ======
@st.cache_data(ttl=300)
def fetch_gold_data(api_key, interval):
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval={interval}&outputsize=500&apikey={api_key}"
    r = requests.get(url)
    data = r.json()
    
    if "values" not in data:
        st.error(f"API Error: {data.get('message', 'Unexpected error')}")
        return pd.DataFrame()

    df = pd.DataFrame(data['values'])
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    df.set_index('datetime', inplace=True)

    # Convert only numeric columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.sort_index(inplace=True)
    return df

# ====== Signal Generation ======
def generate_signals(df):
    df['EMA'] = EMA(df['close'], period=20)
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

# ====== Chart Plotting ======
def plot_signals(df, title):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Gold Price'
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EMA'],
        mode='lines',
        name='EMA',
        line=dict(color='orange')
    ))

    # Add Buy signals slightly below low price for visibility
    fig.add_trace(go.Scatter(
        x=df[df['Buy']].index,
        y=df[df['Buy']]['low'] * 0.995,
        mode='markers',
        name='Buy Signal',
        marker=dict(symbol='triangle-up', size=14, color='green')
    ))

    # Add Sell signals slightly above high price for visibility
    fig.add_trace(go.Scatter(
        x=df[df['Sell']].index,
        y=df[df['Sell']]['high'] * 1.005,
        mode='markers',
        name='Sell Signal',
        marker=dict(symbol='triangle-down', size=14, color='red')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date (IST)',
        yaxis_title='Price (USD)',
        template='plotly_white',
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

# ====== Streamlit UI ======
st.title("Gold Buy/Sell Signals (XAU/USD)")
api_key = st.text_input("🔐 Enter Your Twelve Data API Key", type='password')

timeframes = ['5min', '15min', '30min', '1h', '4h', '1day']
selected_tf = st.selectbox("⏱️ Select Timeframe", timeframes)

if api_key:
    df = fetch_gold_data(api_key, selected_tf)

    if not df.empty:
        df = generate_signals(df)
        plot_signals(df, f"Gold (XAU/USD) - Buy/Sell Signals [{selected_tf}]")

        latest_signal = df[df['Buy'] | df['Sell']]
        if not latest_signal.empty:
            last_signal = latest_signal.iloc[-1]
            signal_time = last_signal.name.strftime('%Y-%m-%d %H:%M')
            if last_signal['Buy']:
                st.success(f"✅ BUY signal at {signal_time} IST")
            elif last_signal['Sell']:
                st.error(f"❌ SELL signal at {signal_time} IST")

        st.subheader("📊 Latest Candles with Signals")
        st.dataframe(df.tail(10)[['open', 'high', 'low', 'close', 'EMA', 'RSI', 'MACD', 'Signal', 'Buy', 'Sell']])
    else:
        st.warning("No data available or API limit reached.")
else:
    st.info("Please enter your Twelve Data API key to continue.")
