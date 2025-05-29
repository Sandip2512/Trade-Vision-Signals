import streamlit as st
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
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = StringIO(response.text)
        df = pd.read_csv(data)
        tickers = df['SYMBOL'].tolist()
        tickers_ns = [ticker + ".NS" for ticker in tickers if ticker.isalpha()]
        return tickers_ns
    except Exception as e:
        st.error(f"Error fetching NSE ticker list: {e}")
        return []

st.title("Advanced Swing & Intraday Trading Strategy for NSE Stocks")

# ✅ Updated Image Section
st.image(
    r"https://github.com/Sandip2512/Trade-Vision-Signals/blob/main/Image%201.jpg",
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
            data_clean['Buy'] = (
                (data_clean['Close'] > data_clean['EMA']) &
                (data_clean['MACD'] > data_clean['Signal']) &
                (data_clean['MACD'].shift(1) < data_clean['Signal'].shift(1))
            )
            data_clean['Sell'] = (
                (data_clean['MACD'] < data_clean['Signal']) |
                (data_clean['RSI'] > 70)
            )

            # Insights with emojis
            st.subheader("Insights 📈📉")
            total_buys = data_clean['Buy'].sum()
            total_sells = data_clean['Sell'].sum()
            latest_rsi = data_clean['RSI'].iloc[-1]
            latest_close = data_clean['Close'].iloc[-1]
            latest_ema = data_clean['EMA'].iloc[-1]
            trend = "Bullish (Close > EMA) 🚀" if latest_close > latest_ema else "Bearish (Close <= EMA) 🐻"

            st.markdown(f"- Total Buy Signals: **{total_buys}** ✅")
            st.markdown(f"- Total Sell Signals: **{total_sells}** ❌")
            st.markdown(f"- Latest RSI: **{latest_rsi:.2f}** {'(Overbought 🔴)' if latest_rsi > 70 else '(Neutral ⚪)'}")
            st.markdown(f"- Latest Close Price: **₹{latest_close:.2f}** 💰")
            st.markdown(f"- EMA{ema_period} Trend: **{trend}**")

            # Plotting based on mode
            st.subheader(f"{selected_ticker} Chart with Buy/Sell Signals ({mode} mode)")

            if mode == 'Intraday':
                # Convert index to IST timezone
                data_clean.index = data_clean.index.tz_convert('Asia/Kolkata')

                today_ist = pd.Timestamp.now(tz='Asia/Kolkata').normalize()

                # Filter today's data between 9:15 and 15:30 IST
                intraday_data = data_clean[
                    (data_clean.index >= today_ist + pd.Timedelta(hours=9, minutes=15)) & 
                    (data_clean.index <= today_ist + pd.Timedelta(hours=15, minutes=30))
                ]

                buys = intraday_data[intraday_data['Buy']]
                sells = intraday_data[intraday_data['Sell']]

                st.write(f"Intraday Buy signals found: {len(buys)}")

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
                    y=buys['High'] + (buys['High'] * 0.01),
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
                buys = data_clean[data_clean['Buy']]
                sells = data_clean[data_clean['Sell']]

                st.write(f"Daily Buy signals found: {len(buys)}")

                fig = go.Figure()

                fig.add_trace(go.Scatter(x=data_clean.index, y=data_clean['Close'],
                                         mode='lines', name='Close Price', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=data_clean.index, y=data_clean['EMA'],
                                         mode='lines', name=f'EMA{ema_period}', line=dict(color='orange')))

                fig.add_trace(go.Scatter(x=buys.index, y=buys['High'] + (buys['High'] * 0.01),
                                         mode='markers', name='Buy Signal', marker_symbol='triangle-up',
                                         marker=dict(color='#00FF00', size=16)))

                fig.add_trace(go.Scatter(x=sells.index, y=sells['Low'] * 0.998,
                                         mode='markers', name='Sell Signal', marker_symbol='triangle-down',
                                         marker=dict(color='#B22222', size=12)))

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    legend=dict(x=0, y=1),
                    hovermode='x unified',
                    template='plotly_white',
                    height=700,
                    width=1100
                )

                st.plotly_chart(fig, use_container_width=True)
                st.write("Recent Signal Data:")
                st.dataframe(data_clean.tail(10))
