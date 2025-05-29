import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import websockets
import json
from datetime import datetime
import pytz
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
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ====== Aggregate Trades ======
def aggregate_trades(trades, interval_seconds):
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['candle_time'] = df['timestamp'].dt.floor(f'{interval_seconds}s')
    ohlc = df.groupby('candle_time').agg(
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('quantity', 'sum')
    )
    return ohlc

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

# ====== Plot Signals ======
def plot_signals(df, title):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index.tz_convert('Asia/Kolkata'),
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))

    fig.add_trace(go.Scatter(
        x=df.index.tz_convert('Asia/Kolkata'),
        y=df['EMA'],
        mode='lines',
        name='EMA',
        line=dict(color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=df[df['Buy']].index.tz_convert('Asia/Kolkata'),
        y=df[df['Buy']]['low'] * 0.995,
        mode='markers',
        name='Buy',
        marker=dict(symbol='triangle-up', size=12, color='green')
    ))

    fig.add_trace(go.Scatter(
        x=df[df['Sell']].index.tz_convert('Asia/Kolkata'),
        y=df[df['Sell']]['high'] * 1.005,
        mode='markers',
        name='Sell',
        marker=dict(symbol='triangle-down', size=12, color='red')
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title='Date (IST)',
        yaxis_title='Price',
        template='plotly_white',
        height=700
    )

    # ⚠️ Important: Use theme=None for stable display
    st.plotly_chart(fig, use_container_width=True, theme=None)

# ====== Streamlit UI ======
st.title("Gold (XAU/USDT) Real-Time Buy/Sell Signals")

interval_map = {
    '5min': 300,
    '15min': 900,
    '30min': 1800,
    '1h': 3600,
    '4h': 14400,
    '1day': 86400
}

selected_tf = st.selectbox("Select Timeframe:", list(interval_map.keys()))

chart_placeholder = st.empty()
signal_placeholder = st.empty()
data_placeholder = st.empty()

trade_buffer = []

# ====== Binance WebSocket Listener ======
async def binance_ws_listener():
    uri = "wss://stream.binance.com:9443/ws/xauusdt@trade"
    async with websockets.connect(uri) as websocket:
        while True:
            msg = await websocket.recv()
            data = json.loads(msg)
            trade = {
                'price': float(data['p']),
                'quantity': float(data['q']),
                'timestamp': int(data['T'])
            }
            trade_buffer.append(trade)

            df_candles = aggregate_trades(trade_buffer, interval_map[selected_tf])

            if len(df_candles) > 20:
                df_signals = generate_signals(df_candles)

                chart_placeholder.empty()
                plot_signals(df_signals, f"Gold (XAU/USDT) Signals [{selected_tf}]")

                latest_signal = df_signals[(df_signals['Buy'] | df_signals['Sell'])].iloc[-1:]
                if not latest_signal.empty:
                    signal_time = latest_signal.index[0].tz_convert('Asia/Kolkata').strftime('%Y-%m-%d %H:%M')
                    if latest_signal['Buy'].iloc[0]:
                        signal_placeholder.success(f"✅ BUY signal at {signal_time} IST")
                    elif latest_signal['Sell'].iloc[0]:
                        signal_placeholder.error(f"❌ SELL signal at {signal_time} IST")
                else:
                    signal_placeholder.info("No recent signal")

                data_placeholder.dataframe(df_signals.tail(10)[[
                    'open', 'high', 'low', 'close', 'EMA', 'RSI', 'MACD', 'Signal', 'Buy', 'Sell'
                ]])

            await asyncio.sleep(1)

# ====== Main Execution ======
def main():
    try:
        asyncio.run(binance_ws_listener())
    except RuntimeError:
        pass  # streamlit rerun

if st.button("Start Real-Time Stream"):
    main()
else:
    st.info("Click 'Start Real-Time Stream' to begin live updates.")
