import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import polars as pl

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import altair
from sklearn.preprocessing import StandardScaler


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
@st.cache_resource
def get_portfolio_data(_collection):
    records = _collection.find({})
    df = pl.DataFrame(list(records))
    return df

# Function to get the Weights Data
@st.cache_resource
def get_weights_data(_collection):
    records = _collection.find({})
    df = pl.DataFrame(list(records))
    return df

@st.cache_resource
def get_stock_returns_data(_collection):
    records = _collection.find({})
    df = pl.DataFrame(list(records))
    return df

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

df_stock_returns = df_stock_returns.select([
    pl.col('Date').str.strptime(pl.Date, format='%Y-%m-%d'),
    pl.col('Stock'),
    pl.col('Return').cast(pl.Float64)
])

###########################################################################
# Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Number of Portfolios", value=df.shape[0])

with col2:
    st.metric(label="Number of Stocks", value=df_weights['Stock'].n_unique())

df_stock_returns2 = df_stock_returns.to_pandas()
df_stock_returns2['Date'] = pd.to_datetime(df_stock_returns2['Date'])
earliest_date = df_stock_returns2['Date'].min().strftime('%Y-%m-%d')
latest_date = df_stock_returns2['Date'].max().strftime('%Y-%m-%d')

with col3:
    st.metric(label="Earliest Date", value=earliest_date)

with col4:
    st.metric(label="Latest Date", value=latest_date)

###########################################################################
# Get the Stock Prices from Yahoo Finance
earliest_date = df_stock_returns['Date'].min()
latest_date = df_stock_returns['Date'].max()

# Get the list of unique stocks
stocks = df_stock_returns['Stock'].unique().to_list()

# Get the Stock Prices
prices = get_prices(stocks, earliest_date, latest_date)
prices = prices['Adj Close']

###########################################################################
# Calculate the Efficient Frontier
lowest_risk = 3.6
highest_risk = 6.5

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
# Randomly select 10% of rows
df_mini = df.sample(n=5000, seed=0)
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_mini['Portfolio Risk'], y=df_mini['Expected Return'],
                         mode='markers', name='Portfolios'))

fig.add_trace(go.Scatter(x=frontier['Portfolio Risk'], y=frontier['Expected Return'],
                         mode='lines', name='Efficient Frontier',
                         line=dict(color='red', width=4)))

fig.update_layout(xaxis_title='Portfolio Risk',
                  yaxis_title='Expected Return',
                  showlegend=True)

risk_level = st.slider("Risk Level", min_value=3.8, max_value=6.5, value=4.2, step=0.01)

fig.add_vline(x=risk_level, line_width=3, line_color="green")

fig.update_traces(hovertemplate='Risk: %{x:.2f}<br>Return: %{y:.2f}')

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
    # Add a title to the plot
    fig.update_layout(title='Efficient Frontier')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Add a title to the plot
    fig = px.treemap(df_filt.to_pandas(), path=['Stock'], values='Weight')
    fig.update_traces(textinfo='label+percent entry')
    fig.update_traces(hovertemplate='Stock: %{label}<br>Weight: %{value:.2f}')
    # Ensure that MSFT is colored lightcoral
    # Ensure that AAPL is colored blue
    # Ensure that NVDA is colored green
    # Ensure that CHPT is colored red
    fig.update_traces(marker=dict(colors=['blue',
                                          'orange',
                                          'red',
                                          'violet',
                                          'green',
                                          'lightblue']))
    fig.update_layout(title='Portfolio Weights')
    st.plotly_chart(fig, use_container_width=True)

###########################################################################
# Define Color Dictionary
# A dictionary to store line colors for different stocks for differentiation
color_dict = {'AAPL':'blue', 'BB':'orange', 'GPRO':'red', 'MSFT':'violet','NVDA':'green', 'T':'lightblue'}

###########################################################################
col1, col2, col3 = st.columns(3)

prices = prices.melt(ignore_index=False)

stocks = prices['Ticker'].unique()

# Multi Select Filter for 'Ticker'
with col1:
    tickers = st.multiselect('Select Ticker', stocks)

# Start Date Filter
with col2:
    start_date = st.date_input('Start Date', earliest_date)

# End Date Filter
with col3:
    end_date = st.date_input('End Date', latest_date)

# Download the prices from yfinance
prices_data = yf.download("AAPL BB GPRO MSFT NVDA T", start=start_date, end=end_date)
# Only Select Adj Close
prices_data = prices_data['Adj Close']

temp_cols = prices_data.columns

# Standardize the Data
scaler = StandardScaler()
prices_data = scaler.fit_transform(prices_data)

prices_data = pd.DataFrame(prices_data, columns=temp_cols)

# Apply Filters
if tickers != []:
    # Filter the Data
    prices_data = prices_data[tickers]
    df_stock_returns = df_stock_returns.filter(pl.col('Stock').is_in(tickers))

# Plot the Prices
fig = go.Figure()

color_dict = {'AAPL':'blue', 'BB':'orange', 'GPRO':'red', 'MSFT':'violet','NVDA':'green', 'T':'lightblue'}
fig = px.line(prices_data,
              x=prices_data.index,
              y=prices_data.columns,
              color_discrete_map=color_dict)

fig.update_layout(xaxis_title='Date',
                    yaxis_title='Price',
                    showlegend=True)

# Add a Title to the Plot
fig.update_layout(title='Standardized Stock Prices')

# Initialize New Columns
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig, use_container_width=True)

###########################################################################
# Filter the Stock Returns Data by Date
df_stock_returns = df_stock_returns.filter(pl.col('Date') >= start_date)
df_stock_returns = df_stock_returns.filter(pl.col('Date') <= end_date)

# Plot the Histogram of Returns colored by Stock
fig = px.histogram(df_stock_returns.to_pandas(),
                   x='Return',
                   color='Stock',
                   color_discrete_map=color_dict,
                   marginal='box',
                   nbins=100,
                   histnorm='probability density')

fig.update_layout(xaxis_title='Return',
                    yaxis_title='Density',
                    showlegend=True)

# Add a Title to the Plot
fig.update_layout(title='Stock Returns')

with col2:
    st.plotly_chart(fig, use_container_width=True)
