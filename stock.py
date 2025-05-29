import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

# EMA, MACD, RSI functions
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

@st.cache_data(ttl=300)
def fetch_gold_intraday(api_key, interval='15min'):
    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol=XAU/USD&interval={interval}&apikey={api_key}&format=JSON"
    )
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Error fetching Gold data from Twelve Data.")
        return pd.DataFrame()
    data_json = response.json()
    if "status" in data_json and data_json["status"] == "error":
        st.error(f"API Error: {data_json.get('message', 'Unknown error')}")
        return pd.DataFrame()
    if 'values' not in data_json:
        st.error("Unexpected API response format.")
        return pd.DataFrame()
    values = data_json['values']
    df = pd.DataFrame(values)
    df = df.rename(columns={
        'datetime': 'date',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })
    # Convert columns to correct types
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df

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

def plot_candlestick_with_signals(df):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Gold Price'
    )])

    # Plot Buy signals
    buys = df[df['Buy']]
    fig.add_trace(go.Scatter(
        x=buys.index,
        y=buys['close'],
        mode='markers',
        marker=dict(symbol='triangle-up', color='green', size=15, opacity=0.9),
        name='Buy Signal'
    ))

    # Plot Sell signals
    sells = df[df['Sell']]
    fig.add_trace(go.Scatter(
        x=sells.index,
        y=sells['close'],
        mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=15, opacity=0.9),
        name='Sell Signal'
    ))

    fig.update_layout(
        title='Gold (XAU/USD) Intraday Price with Buy/Sell Signals',
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            showgrid=True,
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            showgrid=True,
            tickfont=dict(size=12)
        ),
        font=dict(size=14),
        template='plotly_white',
        width=1000,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

# Streamlit UI
st.title("Gold (XAU/USD) Intraday Trading Signals")

api_key = st.text_input("Enter your Twelve Data API Key for Gold", type="password")

if api_key:
    gold_df = fetch_gold_intraday(api_key, interval='15min')
    if not gold_df.empty:
        gold_df = generate_signals(gold_df)
        plot_candlestick_with_signals(gold_df)
        st.dataframe(gold_df.tail(10))
    else:
        st.warning("No data available or API limit reached.")
else:
    st.info("Please enter your Twelve Data API Key to fetch Gold intraday data.")
