import requests
import numpy as np
import pandas as pd
import pickle
import re
from datetime import datetime
from xgboost import XGBClassifier
import plotly.express as px
import plotly.graph_objects as go

# Define daily weather variables used in the model
daily_weather_variables = [
    "weather_code", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "daylight_duration", "sunshine_duration", "precipitation_sum", "rain_sum",
    "snowfall_sum", "precipitation_hours", "wind_speed_10m_max", "wind_gusts_10m_max",
    "wind_direction_10m_dominant", "et0_fao_evapotranspiration"
]
# Load the XGBoost model from the file
file_path = 'SAAW-Wildfire-Predictor/xgboost_model_Canada.pkl'
with open(file_path, 'rb') as file:
    xgb_classifier_Canada = pickle.load(file)
file_path = 'SAAW-Wildfire-Predictor/xgboost_model_California.pkl'
with open(file_path, 'rb') as file:
    xgb_classifier_California = pickle.load(file)

file_path = 'SAAW-Wildfire-Predictor/xgboost_model_Oregon.pkl'
with open(file_path, 'rb') as file:
    xgb_classifier_Oregon = pickle.load(file)
    
print("Model loaded successfully.")


def get_lat_lon(location_input,api_key):
    if re.match(r'^[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?),\s*[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)$', location_input.strip()):
        lat, lon = map(float, location_input.split(','))
        return lat, lon
    else:
        geocoding_api_url = 'https://geocode.maps.co/search'
        parameters = {'q': location_input, 'apikey': api_key}
        response = requests.get(geocoding_api_url, params=parameters)
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
        # print('Location not supported. Maybe you typed it wrong')
        return None, None

def getlocation(lat,lon):  #return the location, approximate lattitude and approximate longitude
  if 48.5<=lat<54.5 and -126.9<=lon<-114.9:
    return 'Canada', round(lat,1),round(lon/2,1)*2
  elif 37.96<=lat<42.76 and -124.84<=lon<-120.04:
    return 'California', round(lat*100/8)*8/100, round(lon*100/8)*8/100
  elif 42.76<=lat<47.56 and -124.84<=lon<-120.04:
    return 'Oregon', round(lat*100/8)*8/100, round(lon*100/8)*8/100
  else:
    return None,None,None

class RiskScorer:
    def __init__(self, api_key, xgb_classifier):
        self.api_key = api_key
        self.xgb_classifier = xgb_classifier

    def fetch_current_weather(self, lat, lon):
        """Fetches current weather data from the API based on latitude and longitude."""
        from datetime import datetime
        weather_api_url = "https://api.open-meteo.com/v1/forecast"
        params = {
        'latitude': lat,
        'longitude': lon,
        "daily": ['temperature_2m_max','temperature_2m_mean','precipitation_sum','rain_sum,snowfall_sum','wind_speed_10m_max','wind_gusts_10m_max','wind_direction_10m_dominant','shortwave_radiation_sum','et0_fao_evapotranspiration'],
        "forecast_days": 1
        }
        response = requests.get(weather_api_url, params=params)
        df=pd.DataFrame(response.json()['daily']).drop('time',axis=1)
        df['lat']=lat
        df['lon']=lon
        return df

    def predict_fire_risk(self, weather_data):
        """Uses the loaded XGBoost model to predict fire risk from weather data."""
        predictions = self.xgb_classifier.predict_proba(weather_data)[:, 1]
        return predictions[0]

    def categorize(self, predictions):
        """Categorizes risk based on the model's prediction output."""
        thresholds=[0.001,0.007,0.015,0.05,0.8]
        categories = ['Very Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Extreme Risk']
        category = categories[np.searchsorted(thresholds, predictions, side='right')]
        return category

# Most important:fire prediction execution, return a string of output
def predictfire(user_input):
    api_key = '65e507374795f403018645aqm64a121'
    lat1,lon1=get_lat_lon(user_input,api_key)
    if pd.isnull(lat1):
      return "Location not supported. Maybe you typed it wrong.",'no category'
    else:
      place,lat,lon=getlocation(lat1,lon1)
      valid=1
      if place=='Canada':
        scorer = RiskScorer(api_key, xgb_classifier_Canada)
      elif place=='California':
        scorer= RiskScorer(api_key, xgb_classifier_California)
      elif place=='Oregon':
        scorer= RiskScorer(api_key, xgb_classifier_Oregon)
      else:
        valid=0
        return 'Location not supported, please try a location that we support', 'no category'
      if valid==1:
        weather_data = scorer.fetch_current_weather(lat, lon)
        fire_risk = scorer.predict_fire_risk(weather_data)# the probability
        fire_risk=round(fire_risk,8)
        category=scorer.categorize(fire_risk)# no risk category
        return fire_risk, category

        # return "The fire risk for area near {user_input} today is "+str(fire_risk)[:9],category+' the lattitude and longitude we used for prediction is:'+str(lat)+str(lon)






import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt



PAGE_CONFIG = {"page_title": "Wildfire Predictor Project", "page_icon": "\U0001F525", "layout": "centered"}
st.set_page_config(**PAGE_CONFIG)

def main():
    st.title("Wildfire Predictor Project")

    # Create a selectbox to switch between pages
    page = st.sidebar.selectbox("Select a page", ["Past Data", "See Where Fires Are Occurring!", "Predict Where Wildfires Will Occur!"])

    if page == "Past Data":
        display_past_data()
    elif page == "See Where Fires Are Occurring!":
        display_current_data()
    elif page == "Predict Where Wildfires Will Occur!":
      display_user_choice()

def display_past_data():
    st.subheader("Canada")
    st.write("Here, you can find information about the total number of fires that have occurred in Canada:")
    st.components.v1.html(open("canada_fire_map.html", "r").read(), width=1000, height=600)
    st.components.v1.html(open("fires1_plot.html", "r").read(), width=1000, height=600)
    st.subheader("United States of America (California and Oregon)")
    st.write("Here, you can find information about the total number of fires that have occurred in the U.S. (zoom in to see available data):")
    st.components.v1.html(open("us_fire_map.html", "r").read(), width=1000, height=600)
    st.components.v1.html(open("fires2_plot.html", "r").read(), width=1000, height=600)
    st.components.v1.html(open("fires3_plot.html", "r").read(), width=1000, height=600)





def display_current_data():
    st.header("Current Data")
    st.write("Here you can find information about current wildfires in Canada.")

    # map_canada = folium.Map(location=[56.1304, -106.3468], zoom_start=4)

    # # Add points to the map
    # for index, row in live_fire.iterrows():
    #     folium.Marker(
    #         location=[row['latitude'], row['longitude']],
    #         icon=folium.CustomIcon(icon_image='https://cdn-icons-png.flaticon.com/512/3304/3304555.png', icon_size=(40, 40))
    #     ).add_to(map_canada)

    # # Display the map
    # folium_static(map_canada)
    st.components.v1.html(open("canada_map.html", "r").read(), width=1000, height=600)

# Color gradient // Thresholds: [0.001,0.007,0.015,0.05,0.8]
def get_color_by_risk(percent):
    if percent < 0.01:
        return 'rgba(76, 175, 80, 0.6)'  # green
    elif percent < 0.07:
        return 'rgba(255, 235, 59, 0.6)'  # yellow
    elif percent < 0.15:
        return 'rgba(255, 193, 7, 0.6)'   # amber
    elif percent < 0.5:
        return 'rgba(255, 87, 34, 0.6)'   # deep orange
    elif percent < 0.8:
        return 'rgba(244, 67, 54, 0.6)'   # red
    else:
        return 'rgba(103, 58, 183, 0.6)'  # deep purple




def display_user_choice():
    user_input_text = st.text_input("Enter the name of any place or coordinates (Latitude, Longitude) of any location in Northern California (North of San Francisco), Oregon, or British Columbia, Canada: ", "")

    if user_input_text:
        fire_risk, category = predictfire(user_input_text)
        if category=='no category':
            st.write(fire_risk)
        else:
            fire_risk_percent = round(float(fire_risk) * 100, 4)
            st.write(f'''
            <h3 style="color:#ADD8E6; font-size: 20px">
            The wildfire risk in {user_input_text} is <span style="color:#FF69B4">{category}<br><br><span style="color:#ADD8E6">Probability: <span style="color:#FF69B4">{fire_risk_percent}%</span>
            </h3>
            ''', unsafe_allow_html=True)


            color = get_color_by_risk(fire_risk_percent/10)

            fig = go.Figure(go.Bar(
                x=[fire_risk_percent / 100],  # Convert percentage for plotting
                y=['Risk Level'],
                orientation='h',
                marker=dict(
                    color=color,  # Dynamic color based on risk
                    line=dict(color=color, width=3)  # Same color for border
                ),
                width=0.75  # Adjust bar thickness here if needed
            ))


            fig.update_layout(
                xaxis=dict(
                  title='Fire Risk Probability (%)',
                  range=[0, 0.01],  # Set the range from 0% to 1% (i.e., 0.0 to 0.01 in scale)
                  tickvals=[i * 0.001 for i in range(11)],  # Define tick positions manually
                  ticktext=[f"{i*0.1:.1f}%" for i in range(11)],  # Define tick labels manually
                  tickmode='array'
                ),
                yaxis=dict(
                    showticklabels=False  # Hide y-axis labels for a cleaner look
                ),
                showlegend=False,
                width=800,
                height=100
            )

            # Display the figure in Streamlit
            st.plotly_chart(fig, use_container_width=True)




if __name__ == "__main__":
    main()
