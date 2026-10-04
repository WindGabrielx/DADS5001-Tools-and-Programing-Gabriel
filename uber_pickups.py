import streamlit as st
import pandas as pd
import numpy as np

st.title("Uber Pickups in NYC 5")

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
# data_load_state.text('Loading data...done!')
data_load_state.text('Done ! (using st.cache_data)')

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
    # st.dataframe(data)

# st.write(type(data[DATE_COLUMN].dt.hour))

st.subheader('Number of pickups by hour')
hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
# st.write(hist_values)

st.bar_chart(hist_values)

hour_to_filter = st.sidebar.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)

st.subheader('Rest API')
import requests

@st.cache_data
def api_call():
    response = requests.get('https://jsonplaceholder.typicode.com/posts/1')
    return response.json()
ans = api_call()
st.write(ans)

st.subheader('Transformers')
import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("text-classification", model = "Flu")

model = load_model()
query = st.text_input("Your query", value="I love Streamlit!")
if query:
    result = model(query)[0]
    st.write(result)