import streamlit as st
import requests
import pandas as pd

@st.cache_data(ttl=300)
def fetch_gold_intraday(api_key, interval='15min'):
    symbol = "XAU/USD"
    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol={symbol}&interval={interval}&apikey={api_key}&format=json"
    )
    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        st.error(f"API error or unexpected response: {data.get('message', 'No data returned')}")
        return pd.DataFrame()

    df = pd.DataFrame(data["values"])

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = df[col].astype(float)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    return df

st.title("Gold (XAU/USD) Intraday Data")

api_key = st.text_input("Enter your Twelve Data API Key for Gold", type="password")

if api_key:
    gold_df = fetch_gold_intraday(api_key)
    if not gold_df.empty:
        st.write("Showing recent Gold intraday data:")
        st.dataframe(gold_df.head(10))
    else:
        st.warning("No data fetched. Check your API key or usage limits.")
else:
    st.info("Please enter your API key above to fetch Gold data.")
