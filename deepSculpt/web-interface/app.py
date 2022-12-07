from urllib import response
import streamlit as st
import pandas as pd
import datetime

"""
# TaxiFareModel front
"""

st.markdown(
    """
Remember that there are several ways to output content into your web page...

Either as with the title by just creating a string (or an f-string).
Or as with this paragraph using the `st.` functions
"""
)
"""
## Here we would like to add some controllers in order to ask the user to select
# the parameters of the ride

1. Let's ask for:
- date and time
- pickup longitude
- pickup latitude
- dropoff longitude
- dropoff latitude
- passenger count
"""

pickup_date = st.date_input("pickup_date", datetime.date(2019, 7, 6))
pickup_time = st.time_input("pick_time", datetime.time(8, 45))

pickup_longitude = st.number_input("Insert pickup lon")
pickup_latitude = st.number_input("Insert a pickup lat")
dropoff_longitude = st.number_input("Insert dropoff lon")
dropoff_latitude = st.number_input("Insert dropoff lat")
passenger_count = st.number_input("Insert a number")

df = pd.DataFrame(
    {
        "pickup_datetime": str(pickup_date) + " " + str(pickup_time),
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": pickup_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_latitude": dropoff_latitude,
        "passenger_count": passenger_count,
    },
    index=[0],
)

import requests

url = "https://taxifare.lewagon.ai/predict"

if url == "https://taxifare.lewagon.ai/predict":

    st.markdown(
        "Maybe you want to use your own API for the prediction, not the one provided by Le Wagon..."
    )

params = {
    "pickup_datetime": datetime.datetime.strptime(
        (str(pickup_date) + " " + str(pickup_time)), "%Y-%m-%d %H:%M:%S"
    ),
    "pickup_longitude": pickup_longitude,
    "pickup_latitude": pickup_latitude,
    "dropoff_longitude": dropoff_longitude,
    "dropoff_latitude": dropoff_latitude,
    "passenger_count": int(passenger_count),
}
params

response = requests.get(url, params=params).json()

response
