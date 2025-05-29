import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

# Fetch Binance crypto data (1m candlesticks)
@st.cache_data(ttl=60)
def fetch_binance_klines(symbol='BTCUSDT', interval='1m', limit=100):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
        'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
        'Taker Buy Quote Asset Volume', 'Ignore'
    ])
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
    return df[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Fetch Gold (XAU/USD) intraday data from Twelve Data
@st.cache_data(ttl=60)
def fetch_gold_intraday(api_key, interval='1min', symbol='XAU/USD', outputsize=100):
    url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={api_key}&outputsize={outputsize}'
    response = requests.get(url)
    data = response.json()
    if "values" not in data:
        st.error(f"Error fetching Gold data: {data.get('message', 'Unknown error')}")
        return pd.DataFrame()
    df = pd.DataFrame(data["values"])
    df['datetime'] = pd.to_datetime(df['datetime'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df = df.rename(columns={'datetime':'Open Time', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'})
    df = df[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.sort_values('Open Time')
    return df

# Plot function
def plot_candles(df, title):
    fig = go.Figure(data=[go.Candlestick(
        x=df['Open Time'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Price', template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

# Streamlit UI
st.title("Intraday Crypto & Gold Price Viewer")

option = st.selectbox("Select Asset", ['BTCUSDT (Crypto)', 'ETHUSDT (Crypto)', 'XAU/USD (Gold)'])

if option == 'XAU/USD (Gold)':
    api_key = st.text_input("Enter your Twelve Data API Key for Gold")
    if api_key:
        gold_df = fetch_gold_intraday(api_key)
        if not gold_df.empty:
            st.dataframe(gold_df.tail(10))
            plot_candles(gold_df, 'Gold (XAU/USD) Intraday Prices')
else:
    # Binance symbol for crypto
    symbol = option.split()[0]
    df = fetch_binance_klines(symbol=symbol)
    st.dataframe(df.tail(10))
    plot_candles(df, f'{symbol} Intraday Prices (1m Interval)')
