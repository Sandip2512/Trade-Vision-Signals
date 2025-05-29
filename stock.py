import streamlit as st
import pandas as pd
import os
import yfinance as yf

# Fetch NSE tickers from Wikipedia (NIFTY 50 companies)
def fetch_nse_tickers_wiki():
    url = "https://en.wikipedia.org/wiki/List_of_NIFTY_50_companies"
    tables = pd.read_html(url)
    df = tables[0]
    df['Symbol'] = df['Symbol'].astype(str) + ".NS"  # yfinance NSE ticker format
    df[['Symbol']].to_csv("nse_stock_list.csv", index=False)
    return df['Symbol'].tolist()

# Check if ticker list exists locally, else fetch it
if not os.path.exists("nse_stock_list.csv"):
    try:
        st.info("Fetching NSE tickers from Wikipedia...")
        nse_tickers = fetch_nse_tickers_wiki()
        st.success(f"NSE tickers saved locally. Total: {len(nse_tickers)}")
    except Exception as e:
        st.error(f"Failed to fetch NSE tickers: {e}")
        st.stop()

# Load NSE tickers from CSV
try:
    nse_df = pd.read_csv("nse_stock_list.csv")
except Exception as e:
    st.error(f"Error loading NSE ticker list: {e}")
    st.stop()

nse_tickers = list(nse_df['Symbol'])

# User selects ticker
selected_ticker = st.selectbox("Choose NSE Stock", sorted(nse_tickers))

# Download last 30 days daily data for selected ticker
df = yf.download(selected_ticker, period="30d", interval="1d")

if df.empty:
    st.error("No data available for the selected ticker.")
    st.stop()

# Show line chart of Close price
st.line_chart(df['Close'])
