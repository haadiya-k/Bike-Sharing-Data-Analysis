import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Read the .csv files

bike_df = pd.read_csv('times_square_rental_data.csv')
weather_df = pd.read_csv('times_square_weather_forecast.csv')

print('Bike Data Information')
print(bike_df.head())
print()
print('Weather Data Information')
print(weather_df.head())

print('Bike Data Descriptive Stats')
print(bike_df.describe())
print()
print('Weather Data Descriptive Stats')
print (weather_df.describe())

print("Rental Data Missing Values:")
print(bike_df.isnull().sum())
print()
print("Weather Data Missing Values:")
print(weather_df.isnull().sum())

bike_df['datetime'] = pd.to_datetime(bike_df['date'] + ' ' + bike_df['time'])
weather_df['datetime'] = pd.to_datetime(weather_df['date'] + ' ' + weather_df['time'])

print(bike_df.head())
print(weather_df.head())

#Drop date and time columns for bikes and weather
bike_df = bike_df.drop(columns=['date', 'time'])
bike_df.head()

weather_df = weather_df.drop(columns=['date', 'time'])
weather_df.head()

print(bike_df.columns)
print(weather_df.columns)

# Merge both weather and bike data frames
merged_df = pd.merge(weather_df, bike_df, on='datetime', how='outer')

merged_df['rounded_time'] = merged_df['datetime'].dt.round('10min')

# Check the results
print(merged_df[['datetime', 'rounded_time']].head())
print(merged_df.columns)

# Aggregate data by rounded time
aggregated_df = merged_df.groupby('rounded_time').agg({
    'bikes_rented': 'sum',
    'bikes_returned': 'sum',
    'temperature': 'mean',
    'humidity': 'mean',
    'precipitationProbability' : 'mean',
    'windSpeed' : 'mean'
}).reset_index()

# Check the aggregated results
print(aggregated_df.head())

aggregated_df= aggregated_df.drop(columns = ['precipitationProbability'])

missing_data_summary = aggregated_df.isnull().sum()
print(missing_data_summary)

aggregated_df[['temperature', 'humidity', 'windSpeed']] = aggregated_df[['temperature', 'humidity', 'windSpeed']].interpolate(method='linear')
aggregated_df

missing_data = aggregated_df.isnull().sum()
print(missing_data)

"""HANDLING OUTLIERS"""

# Define columns to plot for boxplots
columns_to_plot = ['bikes_rented', 'bikes_returned', 'temperature', 'humidity', 'windSpeed']

# Create a boxplot for each column to visually inspect outliers
plt.figure(figsize=(15, 10))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, 3, i)
    plt.boxplot(aggregated_df[column].dropna(), vert=True)
    plt.title(column)
    plt.ylabel(column)

plt.tight_layout()
plt.show()

# Capping outliers using IQR method
capped_df = aggregated_df.copy()

for column in columns_to_plot:

  # Calculate Q1, Q3, and IQR
  Q1 = capped_df[column].quantile(0.25)
  Q3 = capped_df[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  # Cap the values to the lower and upper bounds
  capped_df[column] = capped_df[column].clip(lower=lower_bound, upper=upper_bound)

  # Now capped_df will have the extreme outliers limited to the threshold values

print(capped_df)

capped_df.describe()

capped_df.to_csv('data.csv')

columns_to_plot = ['bikes_rented', 'bikes_returned', 'temperature', 'humidity', 'windSpeed']

# Create a boxplot for each column to visually inspect outliers
plt.figure(figsize=(15, 10))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, 3, i)
    plt.boxplot(capped_df[column].dropna(), vert=True)
    plt.title(column)
    plt.ylabel(column)

plt.tight_layout()
plt.show()

"""
SCATTER PLOTS
"""
# Scatter plot of bikes rented and returned vs Temperature
plt.figure(figsize=(10, 7))
plt.scatter(capped_df['temperature'], capped_df['bikes_rented'], color='blue', label='Bikes Rented', alpha=0.5)
plt.scatter(capped_df['temperature'], capped_df['bikes_returned'], color='cyan', label='Bikes Returned', alpha=0.5)
plt.title('Bikes Rented/Returned vs Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Number of Bikes')
plt.legend()
plt.show()

# Scatter plot of bikes rented and returned vs Humidity
plt.figure(figsize=(10, 7))
plt.scatter(capped_df['humidity'], capped_df['bikes_rented'], color='green', label='Bikes Rented', alpha=0.5)
plt.scatter(capped_df['humidity'], capped_df['bikes_returned'], color='orange', label='Bikes Returned', alpha=0.5)
plt.title('Bikes Rented/Returned vs Humidity')
plt.xlabel('Humidity (%)')
plt.legend()
plt.show()

# Scatter plots of bikes rented and returned vs Wind Speed
plt.figure(figsize=(10, 7))
plt.scatter(capped_df['windSpeed'], capped_df['bikes_rented'], color='orange', label='Bikes Rented', alpha=0.5)
plt.scatter(capped_df['windSpeed'], capped_df['bikes_returned'], color='red', label='Bikes Returned', alpha=0.5)
plt.title('Bikes Rented/Returned vs Wind Speed')
plt.xlabel('Wind Speed (kph)')
plt.ylabel('Number of Bikes')
plt.legend()
plt.show()

# Line Graphs

# Create the plot with two y-axes
fig, ax1 = plt.subplots(figsize=(25, 12))

# Plot bikes rented and returned on the first y-axis
color = 'tab:red'
ax1.set_xlabel('Time')
ax1.set_ylabel('Bikes')
ax1.plot(capped_df['rounded_time'], capped_df['bikes_rented'], color='blue', label='Bikes Rented')
ax1.plot(capped_df['rounded_time'], capped_df['bikes_returned'], color='green', label='Bikes Returned')
ax1.tick_params(axis='y')
ax1.legend(loc='upper left')

# Create a second y-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Temperature (°C)')
ax2.plot(capped_df['rounded_time'], capped_df['temperature'], color='red', label='Temperature')
ax2.tick_params(axis='y')
ax2.legend(loc='upper right')

# Rotate x-axis labels for better readability
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Set the plot title
plt.title('Bikes Rented/Returned and Temperature Over Time')

# Show the plot
plt.tight_layout()
plt.show()

# Create the plot with two y-axes
fig, ax1 = plt.subplots(figsize=(25, 12))

# Plot bikes rented and returned on the first y-axis
ax1.set_xlabel('Time')
ax1.set_ylabel('Bikes')
ax1.plot(capped_df['rounded_time'], capped_df['bikes_rented'], color='blue', label='Bikes Rented')
ax1.plot(capped_df['rounded_time'], capped_df['bikes_returned'], color='green', label='Bikes Returned')
ax1.tick_params(axis='y')
ax1.legend(loc='upper left')

# Create a second y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Temperature (°C)')
ax2.plot(capped_df['rounded_time'], capped_df['temperature'], color='red', label='Temperature')
ax2.tick_params(axis='y')
ax2.legend(loc='upper right')

# Format the x-axis to show the full date and set ticks daily
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.xaxis.set_major_locator(mdates.DayLocator())

# Rotate x-axis labels for better readability
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Set the plot title
plt.title('Bikes Rented/Returned and Temperature Over Time')

# Show the plot
plt.tight_layout()
plt.show()

# Create the plot with two y-axes
fig, ax1 = plt.subplots(figsize=(25, 12))

# Plot bikes rented and returned on the first y-axis
color = 'tab:red'
ax1.set_xlabel('Time')
ax1.set_ylabel('Bikes')
ax1.plot(capped_df['rounded_time'], capped_df['bikes_rented'], color='blue', label='Bikes Rented')
ax1.plot(capped_df['rounded_time'], capped_df['bikes_returned'], color='green', label='Bikes Returned')
ax1.tick_params(axis='y')
ax1.legend(loc='upper left')

# Create a second y-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Humidity (%)')
ax2.plot(capped_df['rounded_time'], capped_df['humidity'], color='red', label='Humidity')
ax2.tick_params(axis='y')
ax2.legend(loc='upper right')

# Format the x-axis to show the full date and set ticks daily
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.xaxis.set_major_locator(mdates.DayLocator())

# Rotate x-axis labels for better readability
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Set the plot title
plt.title('Bikes Rented/Returned and Humidity Over Time')

# Show the plot
plt.tight_layout()
plt.show()

# Create the plot with two y-axes
fig, ax1 = plt.subplots(figsize=(25, 12))

# Plot bikes rented and returned on the first y-axis
color = 'tab:red'
ax1.set_xlabel('Time')
ax1.set_ylabel('Bikes')
ax1.plot(capped_df['rounded_time'], capped_df['bikes_rented'], color='blue', label='Bikes Rented')
ax1.plot(capped_df['rounded_time'], capped_df['bikes_returned'], color='green', label='Bikes Returned')
ax1.tick_params(axis='y')
ax1.legend(loc='upper left')

# Create a second y-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Wind Speed (kph)')
ax2.plot(capped_df['rounded_time'], capped_df['windSpeed'], color='red', label='Wind Speed (kph)')
ax2.tick_params(axis='y')
ax2.legend(loc='upper right')

# Format the x-axis to show the full date and set ticks daily
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.xaxis.set_major_locator(mdates.DayLocator())

# Rotate x-axis labels for better readability
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Set the plot title
plt.title('Bikes Rented/Returned and Wind Speed Over Time')

# Show the plot
plt.tight_layout()
plt.show()

# Calculate the correlation matrix
correlation_matrix = capped_df.corr(numeric_only=True)

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

#Original Data plots

# Scatter plot: Bikes Rented and Returned vs. Weather factors (Temperature, Humidity, Wind Speed, Precipitation)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Scatter plot for Temperature
axs[0, 0].scatter(capped_df['temperature'], capped_df['bikes_rented'], color='blue', label='Bikes Rented', alpha=0.5)
axs[0, 0].scatter(capped_df['temperature'], capped_df['bikes_returned'], color='cyan', label='Bikes Returned', alpha=0.5)
axs[0, 0].set_title('Bikes Rented/Returned vs Temperature')
axs[0, 0].set_xlabel('Temperature (°C)')
axs[0, 0].set_ylabel('Number of Bikes')
axs[0, 0].legend()

# Scatter plot for Humidity
axs[0, 1].scatter(capped_df['humidity'], capped_df['bikes_rented'], color='green', label='Bikes Rented', alpha=0.5)
axs[0, 1].scatter(capped_df['humidity'], capped_df['bikes_returned'], color='orange', label='Bikes Returned', alpha=0.5)
axs[0, 1].set_title('Bikes Rented/Returned vs Humidity')
axs[0, 1].set_xlabel('Humidity (%)')
axs[0, 1].legend()

# Scatter plot for Wind Speed
axs[1, 0].scatter(capped_df['windSpeed'], capped_df['bikes_rented'], color='orange', label='Bikes Rented', alpha=0.5)
axs[1, 0].scatter(capped_df['windSpeed'], capped_df['bikes_returned'], color='red', label='Bikes Returned', alpha=0.5)
axs[1, 0].set_title('Bikes Rented/Returned vs Wind Speed')
axs[1, 0].set_xlabel('Wind Speed (kph)')
axs[1, 0].set_ylabel('Number of Bikes')
axs[1, 0].legend()

fig.delaxes(axs[1, 1])
plt.suptitle('Scatter Plots for Original Data', fontsize=16)

plt.tight_layout()
plt.show()

# Set up the subplot grid
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart for the original data
axs[0].pie(
    capped_df['day_period'].value_counts(),
    labels=capped_df['day_period'].value_counts().index,
    autopct='%1.1f%%',
    startangle=140,
    colors=['skyblue', 'lightcoral']
)
axs[0].set_title('Proportion of Daytime and Nighttime in Original Data')

# Define 'daytime' and 'nighttime' periods based on 'rounded_time'
capped_df['day_period'] = capped_df['rounded_time'].dt.hour.apply(lambda x: 'daytime' if 8 <= x < 20 else 'nighttime')

# Perform stratified sampling: Take 10% from each 'daytime' and 'nighttime' group
stratified_sample_capped = capped_df.groupby('day_period', group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=1)).reset_index(drop=True)

# Display the stratified sample for verification
stratified_sample_capped.head()

stratified_sample_capped

# Scatter plot: Bikes Rented and Returned vs. Weather factors (Temperature, Humidity, Wind Speed, Precipitation)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Scatter plot for Temperature
axs[0, 0].scatter(stratified_sample_capped['temperature'], stratified_sample_capped['bikes_rented'], color='blue', label='Bikes Rented', alpha=0.5)
axs[0, 0].scatter(stratified_sample_capped['temperature'], stratified_sample_capped['bikes_returned'], color='cyan', label='Bikes Returned', alpha=0.5)
axs[0, 0].set_title('Bikes Rented/Returned vs Temperature')
axs[0, 0].set_xlabel('Temperature (°C)')
axs[0, 0].set_ylabel('Number of Bikes')
axs[0, 0].legend()

# Scatter plot for Humidity
axs[0, 1].scatter(stratified_sample_capped['humidity'], stratified_sample_capped['bikes_rented'], color='green', label='Bikes Rented', alpha=0.5)
axs[0, 1].scatter(stratified_sample_capped['humidity'], stratified_sample_capped['bikes_returned'], color='orange', label='Bikes Returned', alpha=0.5)
axs[0, 1].set_title('Bikes Rented/Returned vs Humidity')
axs[0, 1].set_xlabel('Humidity (%)')
axs[0, 1].legend()

# Scatter plot for Wind Speed
axs[1, 0].scatter(stratified_sample_capped['windSpeed'], stratified_sample_capped['bikes_rented'], color='orange', label='Bikes Rented', alpha=0.5)
axs[1, 0].scatter(stratified_sample_capped['windSpeed'], stratified_sample_capped['bikes_returned'], color='red', label='Bikes Returned', alpha=0.5)
axs[1, 0].set_title('Bikes Rented/Returned vs Wind Speed')
axs[1, 0].set_xlabel('Wind Speed (kph)')
axs[1, 0].set_ylabel('Number of Bikes')
axs[1, 0].legend()

fig.delaxes(axs[1, 1])
plt.suptitle('Scatter Plots for Sample Data', fontsize=16)

plt.tight_layout()
plt.show()

# Pie chart for the stratified sample
axs[1].pie(
    stratified_sample_capped['day_period'].value_counts(),
    labels=stratified_sample_capped['day_period'].value_counts().index,
    autopct='%1.1f%%',
    startangle=140,
    colors=['skyblue', 'lightcoral']
)
axs[1].set_title('Proportion of Daytime and Nighttime in Sample')

# Display the plots
plt.show()

capped_df.head()

'''Random Sampling'''

# Perform random sampling: Take 10% of the entire DataFrame
random_sample_capped = capped_df.sample(frac=0.1, random_state=1).reset_index(drop=True)

# Display the random sample for verification
random_sample_capped.head()

import matplotlib.pyplot as plt

# Set up the subplot grid
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart for the original data
axs[0].pie(
    capped_df['day_period'].value_counts(),
    labels=capped_df['day_period'].value_counts().index,
    autopct='%1.1f%%',
    startangle=140,
    colors=['skyblue', 'lightcoral']
)
axs[0].set_title('Proportion of Daytime and Nighttime in Original Data')

# Pie chart for the stratified sample
axs[1].pie(
    random_sample_capped['day_period'].value_counts(),
    labels=stratified_sample_capped['day_period'].value_counts().index,
    autopct='%1.1f%%',
    startangle=140,
    colors=['skyblue', 'lightcoral']
)
axs[1].set_title('Proportion of Daytime and Nighttime in Random Sample')

# Display the plots
plt.show()

# Scatter plot: Bikes Rented and Returned vs. Weather factors (Temperature, Humidity, Wind Speed, Precipitation)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Scatter plot for Temperature
axs[0, 0].scatter(random_sample_capped['temperature'], random_sample_capped['bikes_rented'], color='blue', label='Bikes Rented', alpha=0.6)
axs[0, 0].scatter(random_sample_capped['temperature'], random_sample_capped['bikes_returned'], color='cyan', label='Bikes Returned', alpha=0.6)
axs[0, 0].set_title('Bikes Rented/Returned vs Temperature')
axs[0, 0].set_xlabel('Temperature (°C)')
axs[0, 0].set_ylabel('Number of Bikes')
axs[0, 0].legend()

# Scatter plot for Humidity
axs[0, 1].scatter(random_sample_capped['humidity'], random_sample_capped['bikes_rented'], color='green', label='Bikes Rented', alpha=0.6)
axs[0, 1].scatter(random_sample_capped['humidity'], random_sample_capped['bikes_returned'], color='orange', label='Bikes Returned', alpha=0.6)
axs[0, 1].set_title('Bikes Rented/Returned vs Humidity')
axs[0, 1].set_xlabel('Humidity (%)')
axs[0, 1].legend()

# Scatter plot for Wind Speed
axs[1, 0].scatter(random_sample_capped['windSpeed'], random_sample_capped['bikes_rented'], color='orange', label='Bikes Rented', alpha=0.6)
axs[1, 0].scatter(random_sample_capped['windSpeed'], random_sample_capped['bikes_returned'], color='red', label='Bikes Returned', alpha=0.6)
axs[1, 0].set_title('Bikes Rented/Returned vs Wind Speed')
axs[1, 0].set_xlabel('Wind Speed (kph)')
axs[1, 0].set_ylabel('Number of Bikes')
axs[1, 0].legend()

fig.delaxes(axs[1, 1])
plt.suptitle('Scatter Plots for Random Sample Data', fontsize=16)

plt.tight_layout()
plt.show()


# Sort data by each weather factor for smoother line plots
random_sample_capped = random_sample_capped.sort_values('temperature')
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Line plot for Temperature
axs[0, 0].plot(random_sample_capped['temperature'], random_sample_capped['bikes_rented'], color='blue', label='Bikes Rented')
axs[0, 0].plot(random_sample_capped['temperature'], random_sample_capped['bikes_returned'], color='cyan', label='Bikes Returned')
axs[0, 0].set_title('Bikes Rented/Returned vs Temperature')
axs[0, 0].set_xlabel('Temperature (°C)')
axs[0, 0].set_ylabel('Number of Bikes')
axs[0, 0].legend()

# Sort data by Humidity for the next plot
random_sample_capped = random_sample_capped.sort_values('humidity')
# Line plot for Humidity
axs[0, 1].plot(random_sample_capped['humidity'], random_sample_capped['bikes_rented'], color='green', label='Bikes Rented')
axs[0, 1].plot(random_sample_capped['humidity'], random_sample_capped['bikes_returned'], color='lime', label='Bikes Returned')
axs[0, 1].set_title('Bikes Rented/Returned vs Humidity')
axs[0, 1].set_xlabel('Humidity (%)')
axs[0, 1].legend()

# Sort data by Wind Speed for the next plot
random_sample_capped = random_sample_capped.sort_values('windSpeed')
# Line plot for Wind Speed
axs[1, 0].plot(random_sample_capped['windSpeed'], random_sample_capped['bikes_rented'], color='orange', label='Bikes Rented')
axs[1, 0].plot(random_sample_capped['windSpeed'], random_sample_capped['bikes_returned'], color='gold', label='Bikes Returned')
axs[1, 0].set_title('Bikes Rented/Returned vs Wind Speed')
axs[1, 0].set_xlabel('Wind Speed (kph)')
axs[1, 0].set_ylabel('Number of Bikes')
axs[1, 0].legend()

# Remove unused subplot (bottom right)
fig.delaxes(axs[1, 1])

# Add an overall title and adjust layout
plt.suptitle('Line Plots for Random Sample Data', fontsize=16)
plt.tight_layout()
plt.show()