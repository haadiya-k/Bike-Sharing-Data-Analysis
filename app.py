import streamlit as st
import requests

import main_functions

my_keys = main_functions.read_from_file("api_key.json")
my_weather_api_key = my_keys["weather_key"]

st.title("Data Analysis of Weather and Bike Usage in NYC Times Square")
st.header("API Data")

category= st.selectbox("Choose an API", options=["","Weather", "Bike Usage"])

if category == "Weather":
    api_url = "https://api.tomorrow.io/v4/weather/forecast"
    params = {
        "location": "40.7580,-73.9855",  # Latitude and Longitude for Times Square, NY
        "apikey": my_weather_api_key,  # Your API key
        "units": "metric",
    }
    # Make the GET request
    response = requests.get(api_url, params=params).json()

    if response:
        try:
            # Extracting the relevant data from the 'minutely' timeline
            hourly_data = response['timelines']['hourly']  # Adjust if needed

            # Loop through the first few timepoints for demonstration
            st.subheader("Hourly Weather Data")
            for hour in hourly_data[:24]:  # Limit to the first 24 hours for brevity
                time = hour["time"]
                temperature = hour["values"]["temperature"]
                humidity = hour["values"]["humidity"]
                wind_speed = hour["values"]["windSpeed"]
                precipitationProbability = hour["values"]["precipitationProbability"]

                # Displaying the weather information
                st.write(f"Time: {time}")
                st.write(f"Temperature: {temperature} °C")
                st.write(f"Humidity: {humidity} %")
                st.write(f"Wind Speed: {wind_speed} m/s")
                st.write(f"Precipitation: {precipitationProbability} %")
                st.write("---")

        except KeyError:
            st.error("Error: Unable to fetch data. Check the API response structure.")
        else:
            st.error("No data returned from the API.")

