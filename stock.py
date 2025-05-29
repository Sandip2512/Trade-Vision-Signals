import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

# Indicators
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

# Fetch Forex data from Alpha Vantage
@st.cache_data(ttl=300)
def fetch_forex_data(pair='EURUSD', interval='15min'):
    API_KEY = st.secrets["ALPHAVANTAGE_API_KEY"]
    function_map = {
        '15min': 'FX_INTRADAY',
        '60min': 'FX_INTRADAY',
        '240min': 'FX_INTRADAY',
        '1day': 'FX_DAILY'
    }
    if interval not in function_map:
        st.error("Unsupported interval")
        return pd.DataFrame()

    base = pair[:3]
    quote = pair[3:]
    func = function_map[interval]
    intv = interval if interval != '1day' else ''
    url = f"https://www.alphavantage.co/query?function={func}&from_symbol={base}&to_symbol={quote}&interval={intv}&outputsize=compact&apikey={API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Failed to fetch data from Alpha Vantage")
        return pd.DataFrame()

    data = response.json()
    ts_key = None
    for key in data.keys():
        if 'Time Series' in key:
            ts_key = key
            break
    if not ts_key:
        st.error("No time series data found")
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(data[ts_key], orient='index')
    df.columns = ['open', 'high', 'low', 'close']
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

# Generate buy/sell signals
def generate_signals(df):
    df['EMA'] = EMA(df['close'])
    df['MACD'], df['Signal'] = MACD(df['close'])
    df['RSI'] = RSI(df['close'])
    df.dropna(inplace=True)

    df['Buy'] = ((df['close'] > df['EMA']) &
                 (df['MACD'] > df['Signal']) &
                 (df['MACD'].shift(1) < df['Signal'].shift(1)) &
                 (df['RSI'] < 60))

    df['Sell'] = ((df['close'] < df['EMA']) &
                  (df['MACD'] < df['Signal']) &
                  (df['MACD'].shift(1) > df['Signal'].shift(1)) &
                  (df['RSI'] > 60))
    return df

# Plot chart
def plot_signals(df, title):
    buys = df[df['Buy']]
    sells = df[df['Sell']]

    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'
    )])
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA'], mode='lines', name='EMA', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=buys.index, y=buys['high'] * 1.01, mode='markers', name='Buy Signal',
                             marker=dict(symbol='triangle-up', color='green', size=12)))
    fig.add_trace(go.Scatter(x=sells.index, y=sells['low'] * 0.99, mode='markers', name='Sell Signal',
                             marker=dict(symbol='triangle-down', color='red', size=12)))

    fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Price', template='plotly_white', height=700)
    st.plotly_chart(fig, use_container_width=True)

# App UI
st.title("Forex Trading Signals")

forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']
pair = st.selectbox("Select Forex Pair", forex_pairs)
timeframe = st.selectbox("Select Timeframe", ['15min', '60min', '240min'])

data_load_state = st.text('Loading data...')
data = fetch_forex_data(pair=pair, interval=timeframe)
data_load_state.text('')

if data.empty:
    st.warning("No data available or API limit reached.")
else:
    data = generate_signals(data)
    plot_signals(data, f"{pair} Price with Buy/Sell Signals ({timeframe} timeframe)")
    st.dataframe(data.tail(10))
