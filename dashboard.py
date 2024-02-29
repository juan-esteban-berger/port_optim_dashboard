import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import polars as pl

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import yfinance as yf

from pymongo import MongoClient

###########################################################################
# Set the Page Config
st.set_page_config(layout="wide")

###########################################################################
# Title Using HTML Header 1
st.markdown("<h1 style='text-align: center;'>Hadoop Portfolio Optimization</h1>", unsafe_allow_html=True)

###########################################################################
# Function to get the Portfolio Data
@st.cache_resource  # Use the correct decorator here
def get_portfolio_data(_collection):
    records = _collection.find({})  # Use the argument with an underscore
    df = pl.DataFrame(list(records))
    return df

# Function to get the Weights Data
@st.cache_resource  # Use the correct decorator here
def get_weights_data(_collection):
    records = _collection.find({})  # Use the argument with an underscore
    df = pl.DataFrame(list(records))
    return df

# Function to get the Stock Returns Data
@st.cache_resource  # Use the correct decorator here
def get_stock_returns_data(_collection):
    records = _collection.find({})
    df = pl.DataFrame(list(records))
    return df

# Function to get prices from Yahoo Finance
@st.cache_resource
def get_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    return data

###########################################################################
# Get Data from MongoDB
mongo_uri = st.secrets["mongo_uri"]

client = MongoClient(mongo_uri)
db = client['port_optim']

# Get Portfolio Data
collection = db['finalResult']
df = get_portfolio_data(collection)

# Get Weights Data
collection = db['portfolios']
df_weights = get_weights_data(collection)

# Get Stock Returngs Data
collection = db['stockReturns']
df_stock_returns = get_stock_returns_data(collection)

client.close()

###########################################################################
# Clean the Data from the Portflios 
df = df.drop('_id')

df = df.rename({'field0': 'PortfolioID', 'field1': 'Expected Return', 'field2': 'Portfolio Risk'})

# Convert to Float64
df = df.select([
    pl.col('PortfolioID').cast(pl.Int64),
    pl.col('Expected Return').cast(pl.Float64),
    pl.col('Portfolio Risk').cast(pl.Float64)
])

df = df.sort("PortfolioID")

###########################################################################
# Clean the Weights Data
df_weights = df_weights.drop('_id')

df_weights = df_weights.rename({'field0': 'PortfolioID', 'field1': 'Stock', 'field2': 'Weight'})

# Convert to Float64
df_weights = df_weights.select([
    pl.col('PortfolioID').cast(pl.Int64),
    pl.col('Stock'),
    pl.col('Weight').cast(pl.Float64)
])
df_filt = df_weights.clone()

###########################################################################
# Clean the Stock Returns Data
df_stock_returns = df_stock_returns.drop('_id')

df_stock_returns = df_stock_returns.rename({'field0': 'Date', 'field1': 'Stock', 'field2': 'Return'})

# Convert to Float64
df_stock_returns = df_stock_returns.select([
    pl.col('Date').str.strptime(pl.Date, format='%Y-%m-%d'),
    pl.col('Stock'),
    pl.col('Return').cast(pl.Float64)
])

###########################################################################
# Get the Stock Prices from Yahoo Finance
earliest_date = df_stock_returns['Date'].min()
latest_date = df_stock_returns['Date'].max()

# Get the list of unique stocks
stocks = df_stock_returns['Stock'].unique().to_list()

# Get the Stock Prices
prices = get_prices(stocks, earliest_date, latest_date)
# Only Select Adjusted Close
prices = prices['Adj Close']

###########################################################################
# Calculate the Efficient Frontier
lowest_risk = df['Portfolio Risk'].min()
# Highest Risk for Porfolios above 0.7 Expected Return
highest_risk = df.filter(pl.col('Expected Return') > 0.7)['Portfolio Risk'].max()

risk_range = np.linspace(lowest_risk, highest_risk, 100)

frontier = []
for risk in risk_range:
    # Get the Portfolio with the Highest Expected Return
    highest_return = df.filter(pl.col('Portfolio Risk') < risk)['Expected Return'].max()
    frontier.append([risk, highest_return])

frontier = pl.DataFrame(frontier)
frontier = frontier.transpose()
frontier.columns = ['Portfolio Risk', 'Expected Return']

###########################################################################
# Create Efficient Frontier Plot
fig = go.Figure()

fig.add_trace(go.Scatter(x=df['Portfolio Risk'], y=df['Expected Return'],
                         mode='markers', name='Portfolios'))

fig.add_trace(go.Scatter(x=frontier['Portfolio Risk'], y=frontier['Expected Return'],
                         mode='lines', name='Efficient Frontier',
                         line=dict(color='red', width=4)))

fig.update_layout(title='Portfolio Optimization with Efficient Frontier',
                  xaxis_title='Portfolio Risk',
                  yaxis_title='Expected Return',
                  showlegend=True)

risk_level = st.slider("Risk Level", min_value=1.64, max_value=3.8, value=2.25, step=0.01)

fig.add_vline(x=risk_level, line_width=3, line_color="green")

###########################################################################
# Create the Treemap Plot
pid = df.filter(pl.col('Portfolio Risk') < risk_level)['Expected Return'].arg_max()

df_filt = df_filt.filter(pl.col('PortfolioID') == pid)

df2 = df.clone()
df2 = df2.filter(pl.col('PortfolioID') == pid)

df_filt = df_filt.join(df2, on='PortfolioID', how='left')

###########################################################################
# Display the plots
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.treemap(df_filt.to_pandas(), path=['Stock'], values='Weight')
    st.plotly_chart(fig, use_container_width=True)

###########################################################################
# Plot the Prices using Plotly Express Line Charts
col1, col2, col3, col4 = st.columns(4)

with col1:
    fig = px.line(prices, x=prices.index, y=prices.columns[2], title=prices.columns[2])
    fig.update_traces(line_color='lightcoral')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.line(prices, x=prices.index, y=prices.columns[0], title=prices.columns[0])
    st.plotly_chart(fig, use_container_width=True)


with col3:
    fig = px.line(prices, x=prices.index, y=prices.columns[3], title=prices.columns[3])
    fig.update_traces(line_color='blue')
    st.plotly_chart(fig, use_container_width=True)

with col4:
    fig = px.line(prices, x=prices.index, y=prices.columns[1], title=prices.columns[1])
    fig.update_traces(line_color='red')
    st.plotly_chart(fig, use_container_width=True)

###########################################################################
# Explanation
with st.expander("Explanation", expanded=True):
    st.write("Hello World")
