import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Load the bike usage and weather data files
bike_df = pd.read_csv('times_square_rental_data.csv')
weather_df = pd.read_csv('times_square_weather_forecast.csv')

# Convert date and time to datetime and round to the nearest hour
bike_df['datetime'] = pd.to_datetime(bike_df['date'] + ' ' + bike_df['time']).dt.round('H')
weather_df['datetime'] = pd.to_datetime(weather_df['date'] + ' ' + weather_df['time']).dt.round('H')

# Grouping the bike data and weather data by hour
bike_agg_df = bike_df.groupby('datetime').agg({
    'bikes_rented': 'sum',
    'bikes_returned': 'sum',
    'num_bikes_available': 'mean',
    'num_docks_available': 'mean'
}).reset_index()

weather_agg_df = weather_df.groupby('datetime').agg({
    'temperature': 'mean',
    'humidity': 'mean',
    'windSpeed': 'mean',
    'precipitationProbability': 'mean'
}).reset_index()

# Merging both datasets on the rounded datetime column
merged_df = pd.merge(bike_agg_df, weather_agg_df, on='datetime', how='inner')

# Streamlit Dashboard
st.title("Bike Usage and Weather Dashboard")

st.markdown(
    "This dashboard provides insights into **bike usage** and **weather conditions** in Times Square. Use the tabs below to explore the data through different visualizations.")

# Tabs for different types of visualizations
scatterPlot, linePlot, heatmap, tables = st.tabs(["Scatter Plot", "Line Plot", "Heat Map", "Raw Data"])

#Scatter Plot Tab
with scatterPlot:
    st.header("Scatter Plot: Weather vs Bike Usage")
    st.markdown("Explore how weather conditions affect bike rentals and returns.")

    # Select weather parameter for scatter plot
    weather_parameter = st.selectbox("Choose a weather parameter to compare with bike usage:",
                                     options=["temperature", "humidity", "windSpeed"])

    fig1 = px.scatter(merged_df,
                      x=weather_parameter,
                      y=["bikes_rented", "bikes_returned"],
                      labels={"value": "Bike Rentals/Returns", weather_parameter: weather_parameter.capitalize()})

    st.plotly_chart(fig1)

#Line Plot Tab
with linePlot:
    st.header("Line Plot: Bike Usage and Weather Over Time")
    st.markdown("Analyze bike usage trends over time along with selected weather conditions.")

    # Select weather parameter for line plot
    weather_parameter = st.selectbox("Choose a weather parameter to compare over time:",
                                     options=["temperature", "humidity", "windSpeed"])

    # Create the figure for line plot
    fig3 = go.Figure()

    # Add bike rental and return data
    fig3.add_trace(go.Scatter(x=merged_df['datetime'], y=merged_df['bikes_rented'],
                              mode='lines', name='Bikes Rented'))
    fig3.add_trace(go.Scatter(x=merged_df['datetime'], y=merged_df['bikes_returned'],
                              mode='lines', name='Bikes Returned'))

    # Add weather data on secondary y-axis
    fig3.add_trace(go.Scatter(x=merged_df['datetime'], y=merged_df[weather_parameter],
                              mode='lines', name=weather_parameter.capitalize(), yaxis="y2"))

    # Configure layout for two y-axes
    fig3.update_layout(
        xaxis_title="Time",
        yaxis_title="Number of Bikes",
        yaxis2=dict(title=weather_parameter.capitalize(), overlaying='y', side='right'),
        hovermode="x unified"
    )

    st.plotly_chart(fig3)

#Heatmap Tab
with heatmap:
    st.header("Correlation Heatmap: Bike Usage vs Weather")
    st.markdown("Discover correlations between bike usage and weather parameters.")

    # Calculate correlation matrix
    corr_df = merged_df[['bikes_rented', 'bikes_returned', 'temperature', 'humidity', 'windSpeed']]
    corr_matrix = corr_df.corr()

    # Display heatmap
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

#Raw Data and Statistics Tab
with tables:
    st.header("Raw Data and Descriptive Statistics")
    st.markdown("Explore the raw data used in this analysis, along with summary statistics.")

    st.write("### Raw Data")
    st.dataframe(merged_df)

    st.write("### Descriptive Statistics")
    st.dataframe(merged_df.describe())
