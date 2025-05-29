import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Load NSE and BSE ticker lists ---
@st.cache_data
def load_tickers():
    try:
        nse_df = pd.read_csv('nse_stock_list.csv')  # Replace with your local file path
        bse_df = pd.read_csv('bse_stock_list.csv')  # Replace with your local file path

        # Extract tickers and add yfinance suffixes
        nse_tickers = nse_df['Symbol'].astype(str).str.strip() + '.NS'
        bse_tickers = bse_df['SC_CODE'].astype(str).str.strip() + '.BO'

        # Combine and return as list
        return sorted(list(nse_tickers) + list(bse_tickers))
    except Exception as e:
        st.error(f"Error loading ticker lists: {e}")
        return []

all_tickers = load_tickers()

# --- Technical Indicator Functions ---
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
    RS = avg_gain / avg_loss
    return 100 - (100 / (1 + RS))

# --- App Header ---
st.title("📈 Advanced Swing & Intraday Trading Strategy for NSE & BSE Stocks")
st.image(
    "https://raw.githubusercontent.com/Sandip2512/Trade-Vision-Signals/main/Image%201.jpg",
    caption="Market Trends 📊",
    use_container_width=True
)
st.markdown("### Real-time Stock Market Signals with Buy/Sell Indicators 🚦")

# --- User Inputs ---
if not all_tickers:
    st.warning("Ticker lists not loaded. Please ensure CSV files are present.")
    st.stop()

selected_ticker = st.selectbox("Choose Stock", all_tickers)
mode = st.radio("Select Mode", ['Daily', 'Intraday'])

if mode == 'Intraday':
    interval = '5m'
    start = pd.to_datetime("today") - pd.Timedelta(days=5)
    end = pd.to_datetime("today")
else:
    interval = '1d'
    start = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end = st.date_input("End Date", pd.to_datetime("today"))

# --- Fetch Data using yfinance ---
df = yf.download(selected_ticker, start=start, end=end, interval=interval)
if df.empty:
    st.error("No data retrieved. Try changing ticker or date range.")
    st.stop()
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# --- Indicators ---
ema_period = 20 if mode == 'Intraday' else 50
rsi_period = 7 if mode == 'Intraday' else 14

df['EMA'] = EMA(df['Close'], ema_period)
df['MACD'], df['Signal'] = MACD(df['Close'])
df['RSI'] = RSI(df['Close'], rsi_period)

df.dropna(subset=['EMA', 'MACD', 'Signal', 'RSI'], inplace=True)
if df.empty:
    st.warning("No valid data after indicator calculation.")
    st.stop()

df['Buy'] = (df['Close'] > df['EMA']) & (df['MACD'] > df['Signal']) & (df['MACD'].shift(1) < df['Signal'].shift(1))
df['Sell'] = (df['MACD'] < df['Signal']) | (df['RSI'] > 70)

# --- Insights ---
st.subheader("Insights 📊")
latest = df.iloc[-1]
trend = "🚀 Bullish (Close > EMA)" if latest['Close'] > latest['EMA'] else "🐻 Bearish (Close ≤ EMA)"
rsi_status = "🔴 Overbought" if latest['RSI'] > 70 else "⚪ Neutral"

st.markdown(f"- **Buy Signals:** {df['Buy'].sum()} ✅")
st.markdown(f"- **Sell Signals:** {df['Sell'].sum()} ❌")
st.markdown(f"- **Latest RSI:** {latest['RSI']:.2f} {rsi_status}")
st.markdown(f"- **Latest Close Price:** ₹{latest['Close']:.2f} 💰")
st.markdown(f"- **Trend:** {trend}")

# --- Plotting ---
st.subheader(f"{selected_ticker} Chart ({mode} Mode)")

fig = go.Figure()

if mode == 'Intraday':
    df.index = df.index.tz_convert('Asia/Kolkata')
    df_today = df.between_time("09:15", "15:30")
    buys = df_today[df_today['Buy']]
    sells = df_today[df_today['Sell']]

    fig.add_trace(go.Candlestick(
        x=df_today.index,
        open=df_today['Open'],
        high=df_today['High'],
        low=df_today['Low'],
        close=df_today['Close'],
        name='Price'
    ))
else:
    buys = df[df['Buy']]
    sells = df[df['Sell']]

    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='blue')))

fig.add_trace(go.Scatter(x=df.index, y=df['EMA'], mode='lines', name=f'EMA{ema_period}', line=dict(color='orange')))

fig.add_trace(go.Scatter(x=buys.index, y=buys['High'] * 1.01, mode='markers', name='Buy Signal',
                         marker=dict(symbol='triangle-up', color='green', size=14)))
fig.add_trace(go.Scatter(x=sells.index, y=sells['Low'] * 0.998, mode='markers', name='Sell Signal',
                         marker=dict(symbol='triangle-down', color='red', size=12)))

fig.update_layout(
    xaxis_title="Date" if mode == 'Daily' else "Time",
    yaxis_title="Price (₹)",
    legend=dict(x=0, y=1),
    hovermode='x unified',
    template='plotly_white',
    height=700,
    width=1100,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# --- Data Table ---
st.write("Recent Signal Data:")
st.dataframe(df.tail(10))
