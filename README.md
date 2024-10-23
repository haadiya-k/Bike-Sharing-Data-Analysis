# Times Square Bike-Sharing Data Analysis

**Description**:  
This project explores the impact of weather conditions on bike-sharing usage in Times Square, NYC. Using data collected from the Citi Bike API and Tomorrow.io weather API, the project aims to uncover patterns and correlations between weather factors (such as temperature, precipitation, wind, and humidity) and the number of daily bike rentals and returns. The data was collected over four consecutive days at 5-minute and 10-minute intervals, providing a detailed, real-time snapshot of bike usage and weather conditions.

**Key Features**:
- Python scripts for data collection from Citi Bike and Tomorrow.io APIs.
- Real-time data gathering for bike availability and weather conditions.
- Future analysis will include data visualization, trend analysis, and predictive modeling.

**Technologies**: PyCharm, Google Colab, Python, APIs, Pandas, Matplotlib, Seaborne

## Visualisations

### Scatter Plot: Bikes Rented and Bikes Returned vs Temperature

This plot visualizes the relationship between temperature and the number of bikes rented and returned in Times Square. It highlights trends where bike usage generally increases with warmer temperatures, but there are fluctuations in the data that require further exploration.

![Scatter Plot](https://github.com/user-attachments/assets/60ef6330-7406-4ac6-a7f6-13feb22aec7f)

**Key Insights:**
- Positive Correlation: Warmer temperatures strongly correlate with increased bike-sharing activity.
- Ideal Biking Conditions: The temperature range of 18°C-22°C is the sweet spot for peak usage.
- Outliers: Certain peaks, particularly at 12°C and 22°C, may represent special events or anomalies that require further investigation.

### Scatter Plot: Bikes Rented and Bikes Returned vs Wind Speed

This plot highlights how varying wind speeds may influence bike rentals and returns. While lower wind speeds seem to correspond with a range of rental and return values, higher wind speeds show a decrease in bike activity.

![wind speed scatter plot](https://github.com/user-attachments/assets/9c8b698b-8a51-49ab-b72f-ad754c93e27b)

**Key Insights:**
- Moderate Humidity (55%-70%) is Ideal: This range sees the most bike usage.
- High Humidity Deters Rentals: As humidity rises above 80%, fewer bikes are rented.
- Outliers Need Further Exploration: The spike in rentals at low humidity and the gap between rentals and returns at high humidity may point to specific events or external factors.


### Scatter Plot: Bikes Rented and Bikes Returned vs Humidity

This plot explores the relationship between humidity and bike usage. We can observe how changes in humidity might impact the number of bikes rented or returned, with bike activity generally decreasing at higher humidity levels.

![humidity scatter plot](https://github.com/user-attachments/assets/3a9b5c0e-4f8d-48a4-9d16-967ebc782a93)

**Key Insights:**
- Calmer Winds are Ideal for Biking: Wind speeds between 1 and 3 kph seem to provide the most favorable conditions for bike rentals and returns.
- Stronger Winds Discourage Rentals: As wind speeds exceed 4 kph, fewer people rent bikes, possibly due to discomfort or safety concerns related to biking in windy conditions.
- Outliers Require Further Investigation: The spike in rentals at low wind speeds (~1 kph) suggests that specific events or favorable weather conditions may have contributed to the increase.

### Time Series Plot: Bikes Rented and Returned Over Time with Temperature

This time series plot tracks the number of bikes rented and returned over time alongside the temperature. It offers insights into peak hours of usage and how temperature variations might influence bike-sharing activity.

![time series](https://github.com/user-attachments/assets/4cc5b430-5a43-4d76-bccd-68e71f36bd3a)

**Key Insights:**
- Bike Usage Peaks at Warmer Temperatures: The number of bikes rented and returned rises sharply as temperatures increase, indicating that warmer weather conditions are ideal for biking.
- Cold Weather Reduces Bike Rentals: There is a noticeable drop in bike usage during colder temperature periods, suggesting that colder weather discourages people from renting or returning bikes.
- Bikes Rented and Returned Show Similar Patterns: The peaks and troughs of bike rentals and returns often align, indicating that most users rent and return bikes within a short period or single outing.
- Outliers Require Further Investigation: Some spikes in bike returns don’t align with the temperature trends, suggesting possible external factors such as special events or local conditions influencing bike returns.

### Heat Map

![Heat Map](https://github.com/user-attachments/assets/57d477a2-b12a-4343-889d-8f6117c39472)

**Key Insights:**
- Positive Correlation Between Rentals and Returns: The correlation between bikes rented and bikes returned is strong (0.80), suggesting a clear pattern of users returning bikes soon after renting.
- Temperature's Significant Impact: There is a strong positive correlation between temperature and both bikes rented (0.74) and bikes returned (0.64), indicating that warmer temperatures significantly increase bike usage.
- Humidity Negatively Affects Bike Usage: A moderate negative correlation exists between humidity and both bikes rented (-0.58) and bikes returned (-0.52). This suggests that higher humidity levels may deter users from renting bikes.
- Wind Speed's Mixed Influence: Wind speed shows a moderate positive correlation with temperature (0.65) and a weaker correlation with bikes rented (0.52). This implies that mild winds, usually associated with warmer weather, might encourage biking, but stronger winds might reduce rentals.

