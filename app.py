import streamlit as st
import numpy as np
import pandas as pd
import joblib
import random
import hashlib
from datetime import datetime
from meteostat import Point, Hourly
from tensorflow.keras.models import load_model
import holidays

# =============================
# ✅ Streamlit Configuration
# =============================
st.set_page_config("Flight Delay Predictor", page_icon="✈️", layout="centered")

# =============================
# ✅ Sidebar Navigation
# =============================
page = st.sidebar.radio("Navigation", ["🏠 Home", "📈 Predict Delay", "🛠 Behind The Scenes"])

# =============================
# ✅ Load model and scaler
# =============================
@st.cache_resource
def load_resources():
    model = load_model("flight_delay_lstm_model2.h5", compile=False)
    scaler = joblib.load("lstm_scaler2.pkl")
    return model, scaler

model, scaler = load_resources()
us_holidays = holidays.US()

# =============================
# ✅ Airport & Carrier Mappings
# =============================
airport_data = {
    'ATL': (33.6407, -84.4277), 'CLT': (35.2140, -80.9431), 'DEN': (39.8561, -104.6737),
    'DFW': (32.8998, -97.0403), 'EWR': (40.6895, -74.1745), 'IAH': (29.9902, -95.3368),
    'JFK': (40.6413, -73.7781), 'LAS': (36.0840, -115.1537), 'LAX': (33.9416, -118.4085),
    'MCO': (28.4312, -81.3081), 'MIA': (25.7959, -80.2870), 'ORD': (41.9742, -87.9073),
    'PHX': (33.4373, -112.0078), 'SEA': (47.4502, -122.3088), 'SFO': (37.6213, -122.3790)
}
airport_mapping = {code: idx for idx, code in enumerate(airport_data.keys())}
airport_fullname_to_code = {
    "Hartsfield–Jackson Atlanta Intl (ATL)": "ATL", "Charlotte Douglas Intl (CLT)": "CLT",
    "Denver Intl (DEN)": "DEN", "Dallas/Fort Worth Intl (DFW)": "DFW", "Newark Liberty Intl (EWR)": "EWR",
    "George Bush Intercontinental (IAH)": "IAH", "John F. Kennedy Intl (JFK)": "JFK",
    "Harry Reid Intl (LAS)": "LAS", "Los Angeles Intl (LAX)": "LAX", "Orlando Intl (MCO)": "MCO",
    "Miami Intl (MIA)": "MIA", "O'Hare Intl (ORD)": "ORD", "Phoenix Sky Harbor Intl (PHX)": "PHX",
    "Seattle–Tacoma Intl (SEA)": "SEA", "San Francisco Intl (SFO)": "SFO"
}
carrier_fullname_to_code = {
    "American Airlines (AA)": "AA", "Alaska Airlines (AS)": "AS", "JetBlue Airways (B6)": "B6",
    "Delta Air Lines (DL)": "DL", "Frontier Airlines (F9)": "F9", "Allegiant Air (G4)": "G4",
    "Envoy Air (MQ)": "MQ", "Spirit Airlines (NK)": "NK", "PSA Airlines (OH)": "OH",
    "SkyWest Airlines (OO)": "OO", "United Airlines (UA)": "UA", "Southwest Airlines (WN)": "WN",
    "Republic Airways (YX)": "YX"
}
carrier_mapping = {code: idx for idx, code in enumerate(carrier_fullname_to_code.values())}

# =============================
# ✅ Pages
# =============================
if page == "🏠 Home":
    st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🛬 Welcome to the Flight Delay Predictor!</h1>", unsafe_allow_html=True)
    st.markdown("""
    ### ✈️ How to Use:
    - Select the **Departure Date** and **Scheduled Departure Time**.
    - Choose the **Origin** and **Destination** airports.
    - Pick your **Airline Carrier**.
    - Click **Predict Arrival Delay** to get real-time insights.

    🌦️ The app pulls live weather data automatically for your selected departure airport!
    """)

elif page == "📈 Predict Delay":
    st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🛫 Smart Flight Delay Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Real-Time Delay Insights Using AI + Live Weather</h4>", unsafe_allow_html=True)
    # (Prediction code continues here...)

elif page == "🛠 Behind The Scenes":
    st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🔍 Behind The Scenes</h1>", unsafe_allow_html=True)
    st.markdown("""
    ### How It Works:
    - **Input Variables Collected:**
      - Origin Airport, Destination Airport, Airline Carrier
      - Scheduled Departure Time and Date
      - Real-Time Weather Conditions (Temperature, Humidity, Wind Speed, Wind Direction, Pressure)
    - **Feature Engineering:**
      - Date fields (Month, Week, Day of Week) are transformed using **cyclical encoding** (sine and cosine transformations) to capture periodic patterns.
      - Weather features are directly fed into the model.
      - Holiday flags are generated to capture special days.
    - **Preprocessing:**
      - Airports and carriers are **label-encoded** to numerical values.
      - All features are **standardized** using a **scaler** to optimize LSTM model convergence.
    - **Prediction:**
      - Inputs are reshaped appropriately for the LSTM model.
      - The model predicts the **arrival delay in minutes** based on historical patterns.
    - **Deployment:**
      - A seamless Streamlit web application integrates all steps for real-time user predictions.
    """)

# =============================
# ✅ Footer
# =============================
st.markdown("---")
st.markdown("<small><i>Created by Kelvin Ndambu Kilonzo | Strathmore University | v1.0</i></small>", unsafe_allow_html=True)
