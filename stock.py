import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

API_KEY = 'd0sa91pr01qkkplu0drgd0sa91pr01qkkplu0ds0'

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
