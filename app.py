# Map Visualization

# Install the SQLAlchemy library if it is not installed
!sudo apt-get install python3-dev libmysqlclient-dev > /dev/null
!pip install mysqlclient > /dev/null
!sudo pip3 install -U sql_magic > /dev/null
!pip install psycopg2-binary > /dev/null'

# Google colab upgraded the default SQL Alchemy … this break pandas read_sql
# We are downgrading SQL Alchemy for now.
!pip install -U 'sqlalchemy<2.0'

# @title File Compressor
!pip install bz2file

import bz2file as bz2
import pickle

import pandas as pd
from sqlalchemy import create_engine

conn_string = 'mysql://{user}:{password}@{host}:{port}/{db}?charset=utf8'.format(
    user='Team_SAAW',
    password='oo03vDi+vzk=',
    host = 'jsedocc7.scrc.nyu.edu',
    port     = 3306,
    encoding = 'utf-8',
    db = 'SAAW'
)
engine = create_engine(conn_string)

# Prepare sql_magic library that enable to query to database easily.
%reload_ext sql_magic
%config SQL.conn_name = 'engine'

canada_location_fires = pd.read_sql('SELECT DISTINCT lat, lon, SUM(fire) FROM fullfire GROUP BY lat, lon HAVING SUM(fire) > 0 ', con= engine)
california_location_fires = pd.read_sql('SELECT DISTINCT lat, lon, SUM(fire) FROM Californiafullfire GROUP BY lat, lon HAVING SUM(fire) > 0 ', con= engine)
oregon_location_fires = pd.read_sql('SELECT DISTINCT lat, lon, SUM(fire) FROM Oregonfullfire GROUP BY lat, lon HAVING SUM(fire) > 0 ', con= engine)

!pip install -q  -U geopandas

%%capture
!sudo apt-get install -y libgeos-dev python3-rtree libspatialindex-dev
!pip install -U geopandas fiona shapely pyproj rtree pygeos mapclassify



import geopandas as gpd
import rtree
import pygeos

gpd.options.use_pygeos = True

from google.colab import drive


drive.mount('/content/drive')

file_path1 = '/content/drive/My Drive/Team SAAW/georef-canada-census-subdivision@public.geojson'
file_path2 = '/content/drive/My Drive/Team SAAW/county.geojson'

canada_shapefile = gpd.read_file(file_path1)
us_shapefile = gpd.read_file(file_path2)

'''
You can download the Canada shapefile file from this website: https://data.opendatasoft.com/explore/dataset/georef-canada-census-subdivision%40public/export/?disjunctive.prov_name_en&disjunctive.cd_name_en&disjunctive.csd_name_en
upload it to Google Colab, and run the following the code instead of the one above:
canada_shapefile = gpd.read_file(/content/georef-canada-census-subdivision@public.geojson)


Similarly, you can also download the United States shapefile from this website: https://exploratory.io/map
upload it, and run the following code instead of the one above:
us_shapefile = gpd.read_file(/content/county.geojson)
'''

import folium

# Latitude and longitude coordinates of British Columbia (approximate center)
bc_coordinates = [53.7267, -127.6476]
# Latitude and longitude coordinates of US (for the second map)
us_coordinates = [37.0902, -95.7129]

# Create the first map centered around British Columbia
fmap1 = folium.Map(location=bc_coordinates, zoom_start=6, tiles='cartodbpositron')

# # Create the second map centered around another location
fmap2 = folium.Map(location=us_coordinates, zoom_start=10, tiles='cartodbpositron')

fmap1 = canada_shapefile.explore()


square_coordinates = [
    (49, -127),
    (54.9, -127),
    (54.9, -115),
    (49, -115),
  # Closing coordinate to complete the polygon
]


folium.Polygon(
    locations=square_coordinates,
    color='orange',
    fill_color='orange',
    fill_opacity=0.5
).add_to(fmap1)

for name, row in canada_location_fires.iterrows():


    if row["SUM(fire)"] < 357:
        color = "green"
    elif row["SUM(fire)"] < 714:
        color = "yellow"
    else:
        color = "red"


    if color == 'red':
      opacity = 1
    elif color == 'yellow':
      opacity = 0.5
    else:
      opacity = 0.1


    size = (row['SUM(fire)'] / 118134.0) * 1000

    html = f"""
           <p style='font-family:sans-serif;font-size:11px'>
           <strong># of fires: </strong> {row["SUM(fire)"]}
           <br><strong>Lon: </strong> {row['lon']}
           <br><strong>Lat: </strong> {row["lat"]}
    """
    iframe = folium.IFrame(html=html, width=200, height=60)
    popup = folium.Popup(iframe, max_width=200)

    # Create a marker on the map and add it to the map
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=size,
        color='black',
        weight=0.5,
        popup=popup,
        fill=True,
        fill_opacity=opacity,
        fill_color=color
    ).add_to(fmap1)



fmap1

fmap1.save("canada_fire_map.html")

fmap2 = us_shapefile.explore()

# Define the coordinates of the square area
polygon_coordinates = [
    (42.72, -124.48),
    (42.64, -124.4),
    (42.32, -124.4),
    (42.08, -124.32),
    (42.0, -124.16),
    (41.36, -124.08),
    (41.04, -124.08),
    (40.48, -124.4),
    (40.24, -124.32),
    (39.76, -123.76),
    (39.52, -123.76),
    (39.12, -123.68),
    (39.04, -123.68),
    (38.88, -123.68),
    (38.48, -123.2),
    (38, -122.88),
    (38, -120.08),
    (42.72, -120.08),
    (42.72, -124.48)
]



# Add the square area to the map as a Polygon
folium.Polygon(
    locations=polygon_coordinates,
    color='orange',
    fill_color='orange',
    fill_opacity=0.5
).add_to(fmap2)

for name, row in california_location_fires.iterrows():

    # Make the color green for the working stations, red otherwise
    if row["SUM(fire)"] < 187:
        color = "green"
    elif row["SUM(fire)"] < 374:
        color = "yellow"
    else:
        color = "red"


    if color == 'red':
      opacity = 1
    elif color == 'yellow':
      opacity = 0.5
    else:
      opacity = 0.1

    # The size of the marker is proportional to the number of docks
    size = (row['SUM(fire)'] / 114614.0) * 1000

    html = f"""
           <p style='font-family:sans-serif;font-size:11px'>
           <strong># of fires: </strong> {row["SUM(fire)"]}
           <br><strong>Lon: </strong> {row['lon']}
           <br><strong>Lat: </strong> {row["lat"]}
    """
    iframe = folium.IFrame(html=html, width=200, height=60)
    popup = folium.Popup(iframe, max_width=200)

    # Create a marker on the map and add it to the map
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=size,
        color='black',
        weight=0.5,
        popup=popup,
        fill=True,
        fill_opacity=opacity,
        fill_color=color
    ).add_to(fmap2)

# Define the coordinates of the square area
polygon_coordinates = [
    (42.8, -124.48),
    (42.96, -124.48),
    (43.2, -124.4),
    (43.36, -124.32),
    (43.6, -124.24),
    (43.68, -124.16),
    (43.76, -124.16),
    (43.92, -124.16),
    (44, -124.08),
    (44.4, -124.08),
    (44.8, -124.08),
    (44.88, -124.0),
    (44.96, -124.0),
    (45.04, -123.92),
    (45.6, -123.92),
    (46.16, -123.92),
    (46.32, -124),
    (46.72, -124.08),
    (46.88, -124.08),
    (47.04, -124.16),
    (47.2, -124.16),
    (47.28, -124.24),
    (47.44, -124.32),
    (47.52, -124.32),
    (47.52, -120.08),
    (42.8, -120.08),
    (42.8, -124.48)
]

# Add the square area to the map as a Polygon
folium.Polygon(
    locations=polygon_coordinates,
    color='orange',
    fill_color='orange',
    fill_opacity=0.5
).add_to(fmap2)

for name, row in oregon_location_fires.iterrows():

    # Make the color green for the working stations, red otherwise
    if row["SUM(fire)"] < 140:
        color = "green"
    elif row["SUM(fire)"] < 280:
        color = "yellow"
    else:
        color = "red"


    if color == 'red':
      opacity = 1
    elif color == 'yellow':
      opacity = 0.5
    else:
      opacity = 0.1

    # The size of the marker is proportional to the number of docks
    size = (row['SUM(fire)'] / 34437.0) * 1000

    html = f"""
           <p style='font-family:sans-serif;font-size:11px'>
           <strong># of fires: </strong> {row["SUM(fire)"]}
           <br><strong>Lon: </strong> {row['lon']}
           <br><strong>Lat: </strong> {row["lat"]}
    """
    iframe = folium.IFrame(html=html, width=200, height=60)
    popup = folium.Popup(iframe, max_width=200)

    # Create a marker on the map and add it to the map
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=size,
        color='black',
        weight=0.5,
        popup=popup,
        fill=True,
        fill_opacity=opacity,
        fill_color=color
    ).add_to(fmap2)



fmap2

fmap2.save("us_fire_map.html")

# Time Series Visualization

canada_time_series_data = pd.read_sql('SELECT time, fire FROM fullfire ', con= engine)
california_time_series_data = pd.read_sql('SELECT time, fire FROM Californiafullfire', con= engine)
oregon_time_series_data = pd.read_sql('SELECT time, fire FROM Oregonfullfire', con= engine)

import pandas as pd

# Assuming 'df' is your DataFrame with Date and Value columns
# Convert 'Date' column to datetime if it's not already
canada_time_series_data['time'] = pd.to_datetime(canada_time_series_data['time'])
california_time_series_data['time'] = pd.to_datetime(california_time_series_data['time'])
oregon_time_series_data['time'] = pd.to_datetime(oregon_time_series_data['time'])

# Aggregate values by date (taking the mean)
aggregated_df1 = canada_time_series_data.groupby('time').sum().reset_index()
aggregated_df2 = california_time_series_data.groupby('time').sum().reset_index()
aggregated_df3 = oregon_time_series_data.groupby('time').sum().reset_index()


import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.widgets import Slider
fig1, ax = plt.subplots(figsize = (20,10))

# Plotting the aggregated data
plt.plot(aggregated_df1['time'], aggregated_df1['fire'])
plt.xlabel('Date')
plt.ylabel('Number of Fires')
plt.title('Aggregated Time Series Plot')

import plotly.express as px
fig1 = px.line(aggregated_df1, x = 'time', y = 'fire', title = 'Canada: Number of Fires (Aggregated)')
fig1.update_xaxes(
    rangeslider_visible = True
)

fig1.show()

fig1.write_html("fires1_plot.html")

fig2, ax = plt.subplots(figsize = (20,10))

# Plotting the aggregated data
plt.plot(aggregated_df2['time'], aggregated_df2['fire'])
plt.xlabel('Date')
plt.ylabel('Number of Fires')
plt.title('Aggregated Time Series Plot')

import plotly.express as px
fig2 = px.line(aggregated_df2, x = 'time', y = 'fire', title = 'California: Number of Fires (Aggregated)')
fig2.update_xaxes(
    rangeslider_visible = True
)

fig2.show()

fig2.write_html("fires2_plot.html")

fig3, ax = plt.subplots(figsize = (20,10))

# Plotting the aggregated data
plt.plot(aggregated_df3['time'], aggregated_df2['fire'])
plt.xlabel('Date')
plt.ylabel('Number of Fires')
plt.title('Aggregated Time Series Plot')

import plotly.express as px
fig3 = px.line(aggregated_df2, x = 'time', y = 'fire', title = 'Oregon: Number of Fires (Aggregated)')
fig3.update_xaxes(
    rangeslider_visible = True
)

fig3.show()

fig3.write_html("fires3_plot.html")

# Current Data

import pandas as pd
import requests
from io import StringIO


url = "https://firms.modaps.eosdis.nasa.gov/api/country/csv/346890fab580d0e8069611a820bbaf3d/MODIS_NRT/CAN/2"

# Make the GET request
response = requests.get(url)

# Check if status 200
if response.status_code == 200:

    live_fire = pd.read_csv(StringIO(response.text))


    print(live_fire)
else:
    print("Error:", response.status_code)

url2 = "https://firms.modaps.eosdis.nasa.gov/api/country/csv/346890fab580d0e8069611a820bbaf3d/MODIS_NRT/USA/2"
response = requests.get(url2)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the CSV response
    live_fire2 = pd.read_csv(StringIO(response.text))

    # Display the data
    print(live_fire2)
else:
    print("Error:", response.status_code)

live_fire = pd.concat([live_fire, live_fire2], ignore_index=True)

live_fire

import folium

# Create a map centered at Canada's coordinates
canada_center = [56.1304, -106.3468]  # Center coordinates of Canada
map_canada = folium.Map(location=canada_center, zoom_start=4)

import folium
import pandas as pd

# Create a map of Canada
canada_center = [56.1304, -106.3468]  # Center coordinates of Canada
map_canada = folium.Map(location=canada_center, zoom_start=3, min_zoom=4.5, max_zoom=10)

# Add points to the map and highlight points from the DataFrame
for index, row in pd.DataFrame(live_fire).iterrows():
    popup_text = f"<b>Fire Location</b><br>Latitude: {row['latitude']}<br>Longitude: {row['longitude']}"
    # Default popup text for all locations with fire, including latitude and longitude
    icon = folium.Icon(color='red', icon='fire', prefix='fa')  # Fire icon

    marker = folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=folium.Popup(popup_text, max_width=300),  # Add popup with customized text
        icon=icon
    ).add_to(map_canada)

# Display the map
map_canada

# Save the map to an HTML file
map_canada.save('canada_map.html')

# Install the SQLAlchemy library if it is not installed
!sudo apt-get install python3-dev libmysqlclient-dev > /dev/null
!pip install mysqlclient > /dev/null
!sudo pip3 install -U sql_magic > /dev/null
!pip install psycopg2-binary > /dev/null
# Google colab upgraded the default SQL Alchemy … this break pandas read_sql
# We are downgrading SQL Alchemy for now.
!pip install -U 'sqlalchemy<2.0'

!pip install streamlit-folium

# Streamlit

!pip install -q streamlit
!pip install -q pyngrok

# Authenticate the NGROK
!pip install -q pyngrok
!ngrok authtoken 2gBm3tSmYcTNYv7fePC8k75a7mX_3RuvUoXipRfH33UDNmNAP # other account: 2fSemc0Ep85gdRQMsWap4tWNaUE_4H38eBxsmY8tSLwsYRqDA #William's: 2fSehqUxmyQ2GRDQmNX51p3WWO0_78HcXtmo4ip8aTfgbUkHi

!pip install streamlit-folium


# !pip install rtree
# !pip install pygeos
import geopandas as gpd
import rtree
import pygeos

gpd.options.use_pygeos = True

#### Main part of Streamlit

from google.colab import drive
drive.mount('/content/drive')

%%writefile app.py
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
file_path = '/content/drive/My Drive/Team SAAW/xgboost_model_Canada.pkl'
with open(file_path, 'rb') as file:
    xgb_classifier_Canada = pickle.load(file)
file_path = '/content/drive/My Drive/Team SAAW/xgboost_model_California.pkl'
with open(file_path, 'rb') as file:
    xgb_classifier_California = pickle.load(file)

file_path = '/content/drive/My Drive/Team SAAW/xgboost_model_Oregon.pkl'
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


!streamlit run app.py &>/dev/null &

from pyngrok import ngrok

ngrok.kill()


from pyngrok import ngrok
# Setup a tunnel to the streamlit port 8501
public_url = ngrok.connect(8501)
public_url
