import streamlit as st
import pandas as pd
import os
import yfinance as yf
import plotly.graph_objects as go

# Function to fetch NSE tickers from Wikipedia
def fetch_nse_tickers_wiki():
    url = "https://en.wikipedia.org/wiki/List_of_NIFTY_50_companies"
    tables = pd.read_html(url)
    df = tables[0]
    # Symbol column has the ticker, add .NS suffix for yfinance NSE tickers
    df['Symbol'] = df['Symbol'].astype(str) + ".NS"
    df[['Symbol']].to_csv("nse_stock_list.csv", index=False)
    return df['Symbol'].tolist()

# Function to fetch BSE tickers from Wikipedia
def fetch_bse_tickers_wiki():
    url = "https://en.wikipedia.org/wiki/List_of_BSE_SENSEX_companies"
    tables = pd.read_html(url)
    df = tables[0]
    # BSE codes need .BO suffix for yfinance
    df['Code'] = df['Code'].astype(str) + ".BO"
    df[['Code']].to_csv("bse_stock_list.csv", index=False)
    return df['Code'].tolist()

# Ensure ticker CSV files exist, else fetch them
if not os.path.exists("nse_stock_list.csv"):
    try:
        st.info("Fetching NSE tickers from Wikipedia...")
        nse_tickers = fetch_nse_tickers_wiki()
        st.success(f"NSE tickers saved to CSV. Total: {len(nse_tickers)}")
    except Exception as e:
        st.error(f"Failed to fetch NSE tickers: {e}")

if not os.path.exists("bse_stock_list.csv"):
    try:
        st.info("Fetching BSE tickers from Wikipedia...")
        bse_tickers = fetch_bse_tickers_wiki()
        st.success(f"BSE tickers saved to CSV. Total: {len(bse_tickers)}")
    except Exception as e:
        st.error(f"Failed to fetch BSE tickers: {e}")

# Load ticker lists
try:
    nse_df = pd.read_csv("nse_stock_list.csv")
    bse_df = pd.read_csv("bse_stock_list.csv")
except Exception as e:
    st.error(f"Error loading ticker lists: {e}")
    st.stop()

all_tickers = list(nse_df['Symbol']) + list(bse_df.iloc[:,0])  # BSE df has 'Code' col but safer to use first column

# Your existing Streamlit app below...

selected_ticker = st.selectbox("Choose Stock (NSE or BSE)", sorted(all_tickers))

# For demo, fetch last 30 days daily data:
df = yf.download(selected_ticker, period="30d", interval="1d")

if df.empty:
    st.error("No data for selected ticker.")
    st.stop()

st.line_chart(df['Close'])
