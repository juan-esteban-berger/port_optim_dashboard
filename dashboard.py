import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from pymongo import MongoClient

# File containing the MongoDB URI
uri_file = 'mongo_uri.txt'

# Read the MongoDB URI from the file
with open(uri_file, 'r') as file:
    mongo_uri = file.read().strip()

mongo_uri = st.secrets["mongo_uri"]

# Connect to MongoDB
client = MongoClient(mongo_uri)

# Specify the database and collection
db = client['port_optim']  # Adjust the database name if needed
collection = db['finalResult']  # Adjust the collection name if needed

# Fetch all records from the collection
records = collection.find({})

# Convert the records to a pandas DataFrame
df = pd.DataFrame(list(records))

# Display the DataFrame

# Close the MongoDB connection
client.close()

# Drop the _id column
df.drop('_id', axis=1, inplace=True)

# Rename field0 to PortfolioID
# Rename field1 to Expected Return
# Rename field2 to Portfolio Risk (Standard Deviation)
df.rename(columns={'field0': 'PortfolioID', 'field1': 'Expected Return', 'field2': 'Portfolio Risk'}, inplace=True)

# Convert all columns to double
df = df.astype('double')

# Sort by PortfolioID
df.sort_values('PortfolioID', inplace=True)


# Create violion plot
# Using plotly.express

# Scatter plot
fig = px.scatter(df, x="Portfolio Risk", y="Expected Return")

st.plotly_chart(fig)
