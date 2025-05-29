import streamlit as st
import pandas as pd
import threading
import time
import json
import websocket  # pip install websocket-client
import plotly.graph_objects as go

# Initialize session state
if 'trades' not in st.session_state:
    st.session_state['trades'] = []
if 'running' not in st.session_state:
    st.session_state['running'] = False

# WebSocket callbacks
def on_message(ws, message):
    data = json.loads(message)
    trade = {
        'price': float(data['p']),
        'quantity': float(data['q']),
        'timestamp': int(data['T'])
    }
    st.session_state['trades'].append(trade)

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    print("WebSocket connection opened")

# WebSocket thread function
def ws_thread():
    ws = websocket.WebSocketApp(
        "wss://stream.binance.com:9443/ws/xauusdt@trade",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

# Aggregate trades into OHLC candles
def aggregate_trades(trades, interval_seconds=300):
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['candle_time'] = df['timestamp'].dt.floor(f'{interval_seconds}s')
    ohlc = df.groupby('candle_time').agg(
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('quantity', 'sum')
    )
    return ohlc

# Plot OHLC candles with Plotly
def plot_candles(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='XAU/USDT'
    ))
    fig.update_layout(
        title='XAU/USDT Real-Time Candlestick Chart',
        xaxis_title='Time',
        yaxis_title='Price (USDT)',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# Streamlit UI
st.title("Gold (XAU/USDT) Real-Time Streaming Chart")

interval_sec = st.selectbox("Select Candle Interval (seconds):", [300, 900, 1800, 3600], index=0)

if not st.session_state['running']:
    if st.button("Start Streaming"):
        st.session_state['running'] = True
        threading.Thread(target=ws_thread, daemon=True).start()
        st.experimental_rerun()
else:
    st.write("Streaming live data...")

    # Refresh every 5 seconds
    while True:
        if len(st.session_state['trades']) < 10:
            st.write("Waiting for trade data...")
        else:
            df_candles = aggregate_trades(st.session_state['trades'], interval_seconds=interval_sec)
            if not df_candles.empty:
                plot_candles(df_candles.tail(50))
        time.sleep(5)
        st.experimental_rerun()
