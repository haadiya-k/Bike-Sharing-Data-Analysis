import requests
import pytz
import csv
import time
from datetime import datetime
import main_functions

def fetch_weather_data():
    # Read API key from the file
    my_keys = main_functions.read_from_file("api_key.json")
    my_weather_api_key = my_keys["weather_key"]

    # WeatherAPI endpoint
    api_url = "https://api.tomorrow.io/v4/weather/forecast"
    params = {
        "location": "40.7580,-73.9855",  # Latitude and Longitude for Times Square, NY
        "apikey": my_weather_api_key,  # Your API key
        "units": "metric",
    }

    # Make the GET request
    response = requests.get(api_url, params=params)

    # Parse the response as JSON
    data = response.json()

    # Define the Times Square timezone
    times_square_tz = pytz.timezone('America/New_York')

    # Filter for today's forecast every 10 minutes
    today_forecast = []
    try:
        # Check if the 'timelines' and 'minutely' keys exist
        if 'timelines' in data and 'minutely' in data['timelines']:
            for interval in data['timelines']['minutely']:
                # Extract the date from the timestamp
                timestamp = interval['time']

                # Convert timestamp to a datetime object
                utc_time = datetime.fromisoformat(timestamp[:-1])

                # Convert to local time
                local_time = utc_time.replace(tzinfo=pytz.utc).astimezone(times_square_tz)

                # Only include data if the time is less than or equal to the current time
                if local_time <= datetime.now(times_square_tz):
                    values = interval['values']  # Collect the values
                    values['date'] = local_time.strftime('%Y-%m-%d')  # Format and store date
                    values['time'] = local_time.strftime('%H:%M:%S')  # Format and store time
                    today_forecast.append(values)

    except KeyError as e:
        print(f"KeyError: {e}. Please check the API response structure.")

    # Save to CSV
    if today_forecast:
        print(f"Weather Forecast Update for Times Square on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:")
        print()

        # Create or open the CSV file in append mode
        with open('times_square_weather_forecast.csv', mode='a', newline='') as csv_file:
            fieldnames = ['date', 'time', 'temperature', 'humidity', 'windSpeed', 'precipitationProbability']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            # Write the header only if the file is new
            if csv_file.tell() == 0:
                writer.writeheader()  # Write the header row

            for forecast in today_forecast:
                # Print the data
                if 'date' in forecast and 'time' in forecast:
                    print(f"Date: {forecast['date']}")
                    print(f"Time: {forecast['time']}")
                    print(f"Temperature: {forecast['temperature']} °C")
                    print(f"Humidity: {forecast['humidity']} %")
                    print(f"Wind Speed: {forecast['windSpeed']} kph")
                    print(f"Precipitation: {forecast['precipitationProbability']} %")
                    print("-------------------------")

                    # Write the data to the CSV file
                    writer.writerow({
                        "date": forecast['date'],  # Use the formatted local date
                        "time": forecast['time'],  # Use the formatted local time
                        'temperature': forecast['temperature'],
                        'humidity': forecast['humidity'],
                        'windSpeed': forecast['windSpeed'],
                        'precipitationProbability': forecast['precipitationProbability']
                    })

        print("Data has been updated in 'times_square_weather_forecast.csv'.")
    else:
        print("No forecast data found for today.")

# Main loop to fetch weather data every 10 minutes
if __name__ == "__main__":
    while True:
        fetch_weather_data()
        # Wait for 10 minutes before fetching data again
        time.sleep(600)  # 600 seconds = 10 minutes
