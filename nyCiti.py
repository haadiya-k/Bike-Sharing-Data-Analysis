import requests
import pandas as pd
from datetime import datetime
import time
import os

# Set your frequency for capturing data (every 5 minutes)
DATA_INTERVAL = 5  # in minutes

# Define the coordinates for Times Square
TIMES_SQUARE_COORDINATES = (40.7580, -73.9855)
RADIUS = 0.01  # Approximate radius in degrees (~1.1 km)

# Define the file name for storing data
file_name = 'times_square_rental_data.csv'

# Check if the file already exists and load existing data
if os.path.exists(file_name):
    rental_data = pd.read_csv(file_name)
else:
    # Initialize a DataFrame to store rental data if the file does not exist
    rental_data = pd.DataFrame(columns=["date", "time", "station_name", "num_bikes_available", "num_docks_available", "bikes_rented", "bikes_returned"])

# Dictionary to store previous bike availability for each station
previous_bike_count = {}

def is_near_times_square(lat, lon):
    """Check if a station is within the specified radius of Times Square."""
    return (TIMES_SQUARE_COORDINATES[0] - RADIUS <= lat <= TIMES_SQUARE_COORDINATES[0] + RADIUS and
            TIMES_SQUARE_COORDINATES[1] - RADIUS <= lon <= TIMES_SQUARE_COORDINATES[1] + RADIUS)

while True:
    try:
        # Fetch station status
        station_status_url = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"
        status_response = requests.get(station_status_url)
        status_response.raise_for_status()  # Raise an error for bad responses
        station_status_data = status_response.json()

        # Fetch station information
        station_info_url = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
        info_response = requests.get(station_info_url)
        info_response.raise_for_status()  # Raise an error for bad responses
        station_info_data = info_response.json()

        # Get the current timestamp
        current_time = datetime.now()

        # Create a list to store the new data
        new_data = []

        # Store bike rental data for stations near Times Square
        for station in station_status_data['data']['stations']:
            station_id = station['station_id']
            station_name = None
            station_latitude = None
            station_longitude = None

            # Find the station name and coordinates from station information
            for info in station_info_data['data']['stations']:
                if info['station_id'] == station_id:
                    station_name = info['name']
                    station_latitude = info['lat']
                    station_longitude = info['lon']
                    break

            # Check if the station is within the radius of Times Square
            if is_near_times_square(station_latitude, station_longitude):

                # Get the current number of bikes available
                current_bikes_available = station['num_bikes_available']

                # Calculate bike usage (rented or returned)
                bikes_rented = bikes_returned = 0
                if station_id in previous_bike_count:
                    previous_bikes_available = previous_bike_count[station_id]

                    if current_bikes_available < previous_bikes_available:
                        bikes_rented = previous_bikes_available - current_bikes_available
                    elif current_bikes_available > previous_bikes_available:
                        bikes_returned = current_bikes_available - previous_bikes_available

                # Store the current bike count for future comparison
                previous_bike_count[station_id] = current_bikes_available

                # Add station data to new_data
                new_data.append({
                    "date": current_time.date(),
                    "time": current_time.strftime("%H:%M:%S"),
                    "station_name": station_name,
                    "num_bikes_available": current_bikes_available,
                    "num_docks_available": station['num_docks_available'],
                    "bikes_rented": bikes_rented,
                    "bikes_returned": bikes_returned
                })

        # Convert new data to DataFrame
        new_data_df = pd.DataFrame(new_data)

        # Append new data to rental_data DataFrame
        if not new_data_df.empty:
            rental_data = pd.concat([rental_data, new_data_df], ignore_index=True)

        # Print the new data for debugging
        print(f"New Data Captured at {current_time}:")
        print(new_data_df)
        print("Total Data Collected:")
        print(rental_data)

        # Save rental_data to CSV
        rental_data.to_csv(file_name, index=False)
        print(f"Data has been updated in '{file_name}'")

        # Wait for the specified data interval before the next capture
        time.sleep(DATA_INTERVAL * 60)  # Convert minutes to seconds

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        time.sleep(60)  # Wait for a minute before trying again
