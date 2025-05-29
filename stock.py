import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

API_KEY = 'YOUR_FINNHUB_API_KEY'

def fetch_finnhub_candle(symbol, asset_type, resolution='60', days=5):
    """
    Fetch candle data from Finnhub API
    asset_type: 'crypto' or 'forex'
    resolution: '1', '5', '15', '30', '60', 'D', 'W', 'M'
    days: number of past days to fetch
    """
    end_time = int(datetime.utcnow().timestamp())
    start_time = end_time - days * 24 * 60 * 60
    
    url = f'https://finnhub.io/api/v1/{asset_type}/candle'
    params = {
        'symbol': symbol,
        'resolution': resolution,
        'from': start_time,
        'to': end_time,
        'token': API_KEY
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if data.get('s') != 'ok':
        st.error(f"Failed to fetch data for {symbol}: {data.get('error', 'Unknown error')}")
        return pd.DataFrame()
    
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(data['t'], unit='s'),
        'open': data['o'],
        'high': data['h'],
        'low': data['l'],
        'close': data['c'],
        'volume': data['v']
    })
    df.set_index('timestamp', inplace=True)
    return df

st.title("Finnhub Forex & Crypto Live Prices")

# Fetch BTC data from Crypto endpoint
btc_data = fetch_finnhub_candle('COINBASE:BTC-USD', 'crypto', resolution='60', days=5)

# Fetch Gold data from Forex endpoint
gold_data = fetch_finnhub_candle('OANDA:XAU_USD', 'forex', resolution='60', days=5)

if not btc_data.empty:
    st.subheader("Bitcoin (BTC/USD) - Last 5 days")
    fig_btc = go.Figure(data=[go.Candlestick(
        x=btc_data.index,
        open=btc_data['open'],
        high=btc_data['high'],
        low=btc_data['low'],
        close=btc_data['close'],
        name='BTC'
    )])
    fig_btc.update_layout(height=500, title="BTC/USD Price (Hourly)")
    st.plotly_chart(fig_btc)

if not gold_data.empty:
    st.subheader("Gold (XAU/USD) - Last 5 days")
    fig_gold = go.Figure(data=[go.Candlestick(
        x=gold_data.index,
        open=gold_data['open'],
        high=gold_data['high'],
        low=gold_data['low'],
        close=gold_data['close'],
        name='Gold'
    )])
    fig_gold.update_layout(height=500, title="Gold (XAU/USD) Price (Hourly)")
    st.plotly_chart(fig_gold)















































































































"""import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from io import StringIO

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

@st.cache_data(show_spinner=False)
def load_nse_tickers():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = StringIO(response.text)
        df = pd.read_csv(data)
        tickers = df['SYMBOL'].tolist()
        tickers_ns = [ticker + ".NS" for ticker in tickers if ticker.isalpha()]
        return tickers_ns
    except Exception as e:
        st.error(f"Error fetching NSE ticker list: {e}")
        return []

def filter_signals(buy_indices, sell_indices):
    filtered = []
    i, j = 0, 0
    last_signal = None  # 'buy' or 'sell'

    buy_indices_sorted = sorted(buy_indices)
    sell_indices_sorted = sorted(sell_indices)

    while i < len(buy_indices_sorted) or j < len(sell_indices_sorted):
        next_buy = buy_indices_sorted[i] if i < len(buy_indices_sorted) else None
        next_sell = sell_indices_sorted[j] if j < len(sell_indices_sorted) else None

        if next_buy is not None and (next_sell is None or next_buy < next_sell):
            if last_signal != 'buy':
                filtered.append(next_buy)
                last_signal = 'buy'
            i += 1
        elif next_sell is not None:
            if last_signal != 'sell':
                filtered.append(next_sell)
                last_signal = 'sell'
            j += 1
        else:
            break

    return filtered

st.title("Advanced Swing & Intraday Trading Strategy for NSE Stocks")

st.image(
    r"C:\Users\hp\Downloads\Stock bot\image\Image 1.jpg",
    caption="Market Trends 📈",
    use_container_width=True
)

st.markdown("### 📊 Real-time Stock Market Signals with Buy/Sell Indicators 🚦")

all_nse_tickers = load_nse_tickers()

if not all_nse_tickers:
    st.warning("NSE ticker list not loaded. Please try refreshing or check your connection.")
    st.stop()

selected_ticker = st.selectbox("Select NSE Stock", sorted(all_nse_tickers))
mode = st.radio("Select Mode", ['Daily', 'Intraday'])

if mode == 'Intraday':
    interval = '5m'
    max_days = 5
    start_date = pd.to_datetime("today") - pd.Timedelta(days=max_days)
    end_date = pd.to_datetime("today")
else:
    interval = '1d'
    start_date = st.date_input("Start date", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End date", pd.to_datetime("today"))

if selected_ticker:
    data_raw = yf.download(selected_ticker, start=start_date, end=end_date, interval=interval)
    if isinstance(data_raw.columns, pd.MultiIndex):
        data_raw.columns = data_raw.columns.get_level_values(0)

    data = data_raw.copy()

    if data.empty:
        st.error("No data fetched. Try different dates or ticker.")
    else:
        ema_period = 20 if mode == 'Intraday' else 50
        rsi_period = 7 if mode == 'Intraday' else 14

        data['EMA'] = EMA(data['Close'], ema_period)
        data['MACD'], data['Signal'] = MACD(data['Close'])
        data['RSI'] = RSI(data['Close'], rsi_period)

        indicators = ['EMA', 'MACD', 'Signal', 'RSI']
        data_clean = data.dropna(subset=indicators)

        if data_clean.empty:
            st.error("No data after indicator calculation. Try different range.")
        else:
            # Updated Buy and Sell Signals with Close < EMA in Sell condition
            data_clean['Buy'] = (
                (data_clean['Close'] > data_clean['EMA']) &
                (data_clean['MACD'] > data_clean['Signal']) &
                (data_clean['MACD'].shift(1) < data_clean['Signal'].shift(1)) &
                (data_clean['RSI'] < 60)
            )
            data_clean['Sell'] = (
                (data_clean['Close'] < data_clean['EMA']) &  # Added this condition here
                (data_clean['MACD'] < data_clean['Signal']) &
                (data_clean['MACD'].shift(1) > data_clean['Signal'].shift(1)) &
                (data_clean['RSI'] > 60)
            )

            buy_indices = data_clean.index[data_clean['Buy']].tolist()
            sell_indices = data_clean.index[data_clean['Sell']].tolist()

            filtered_indices = filter_signals(buy_indices, sell_indices)

            data_clean['Filtered_Buy'] = False
            data_clean['Filtered_Sell'] = False
            for idx in filtered_indices:
                if idx in buy_indices:
                    data_clean.at[idx, 'Filtered_Buy'] = True
                else:
                    data_clean.at[idx, 'Filtered_Sell'] = True

            st.subheader("Insights 📈📉")
            total_buys = data_clean['Filtered_Buy'].sum()
            total_sells = data_clean['Filtered_Sell'].sum()
            latest_rsi = data_clean['RSI'].iloc[-1]
            latest_close = data_clean['Close'].iloc[-1]
            latest_ema = data_clean['EMA'].iloc[-1]
            trend = "Bullish (Close > EMA) 🚀" if latest_close > latest_ema else "Bearish (Close <= EMA) 🐻"

            st.markdown(f"- Total Buy Signals: **{total_buys}** ✅")
            st.markdown(f"- Total Sell Signals: **{total_sells}** ❌")
            st.markdown(f"- Latest RSI: **{latest_rsi:.2f}** {'(Overbought 🔴)' if latest_rsi > 70 else '(Neutral ⚪)'}")
            st.markdown(f"- Latest Close Price: **₹{latest_close:.2f}** 💰")
            st.markdown(f"- EMA{ema_period} Trend: **{trend}**")

            st.subheader(f"{selected_ticker} Chart with Buy/Sell Signals ({mode} mode)")

            if mode == 'Intraday':
                data_clean.index = data_clean.index.tz_convert('Asia/Kolkata')
                today_ist = pd.Timestamp.now(tz='Asia/Kolkata').normalize()

                intraday_data = data_clean[
                    (data_clean.index >= today_ist + pd.Timedelta(hours=9, minutes=15)) &
                    (data_clean.index <= today_ist + pd.Timedelta(hours=15, minutes=30))
                ]

                buys = intraday_data[intraday_data['Filtered_Buy']]
                sells = intraday_data[intraday_data['Filtered_Sell']]

                fig = go.Figure(data=[go.Candlestick(
                    x=intraday_data.index,
                    open=intraday_data['Open'],
                    high=intraday_data['High'],
                    low=intraday_data['Low'],
                    close=intraday_data['Close'],
                    name='Candlestick'
                )])

                fig.add_trace(go.Scatter(
                    x=intraday_data.index,
                    y=intraday_data['EMA'],
                    mode='lines',
                    name=f'EMA{ema_period}',
                    line=dict(color='orange')
                ))

                fig.add_trace(go.Scatter(
                    x=buys.index,
                    y=buys['High'] * 1.01,
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(symbol='triangle-up', color='#00FF00', size=16)
                ))

                fig.add_trace(go.Scatter(
                    x=sells.index,
                    y=sells['Low'] * 0.998,
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(symbol='triangle-down', color='#B22222', size=12)
                ))

                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Price (₹)",
                    legend=dict(x=0, y=1),
                    hovermode='x unified',
                    template='plotly_white',
                    height=700,
                    width=1100,
                    xaxis_rangeslider_visible=False
                )

                st.plotly_chart(fig, use_container_width=True)
                st.write("Recent Signal Data:")
                st.dataframe(intraday_data.tail(10))

            else:
                buys = data_clean[data_clean['Filtered_Buy']]
                sells = data_clean[data_clean['Filtered_Sell']]

                fig = go.Figure()

                fig.add_trace(go.Scatter(x=data_clean.index, y=data_clean['Close'],
                                         mode='lines', name='Close Price', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=data_clean.index, y=data_clean['EMA'],
                                         mode='lines', name=f'EMA{ema_period}', line=dict(color='orange')))

                fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'],
                                         mode='markers', name='Buy Signal',
                                         marker=dict(symbol='triangle-up', color='green', size=12)))

                fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'],
                                         mode='markers', name='Sell Signal',
                                         marker=dict(symbol='triangle-down', color='red', size=12)))

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    legend=dict(x=0, y=1),
                    hovermode='x unified',
                    template='plotly_white',
                    height=700,
                    width=1100,
                    xaxis_rangeslider_visible=True
                )
"""
                st.plotly_chart(fig, use_container_width=True)
                st.write("Recent Signal Data:")
                st.dataframe(data_clean.tail(10))
