# Times Square Bike-Sharing Data Analysis Dashboard

## Live Dashboard

Explore the interactive dashboard [here](https://bikeusage.streamlit.app/).

This dashboard provides insights into the impact of weather conditions on bike-sharing usage in Times Square, NYC. It combines interactive visualizations and machine learning models to uncover patterns and trends in bike rentals and returns under varying weather conditions.

---

## Features

### 1. **Home Page**
- Provides an overview of the dashboard’s purpose and features.
- Explains how to navigate between sections using the sidebar.

### 2. **Data**
- Overview of the preprocessed bike-sharing and weather data.
- Description of key columns and preprocessing steps.

### 3. **Scatter Plots**
- Visualize the relationship between bike-sharing metrics (`bikes_rented`, `bikes_returned`) and weather parameters (`temperature`, `humidity`, `windSpeed`).
- Key observations for each weather parameter.

### 4. **Line Plots**
- Track bike rentals and returns over time alongside weather conditions.
- Analyze trends and patterns in usage.

### 5. **Heat Maps**
- Correlation matrix between bike-sharing metrics and weather parameters.
- Identify strong relationships and dependencies.

### 6. **Sampling**
- **Stratified Sampling**: View data grouped by specific conditions, e.g. daytime and nighttime.
- **Random Sampling**: Analyze a random subset of the data to identify trends.

### 7. **Models**
- **Model Training and Evaluation**:
  - Random Forest Regressor
  - Linear Regression
- **Residual Plots**: Understand model behavior by visualizing residuals.
- **Predicted vs. Actual Plots**: Compare predicted values with actual observations.
- **Feature Importance**: Highlight key features impacting bike-sharing activity.
- **McNemar's Test**: Compare the correctness of predictions between models.

---

## Data

### Sources:
- **Citi Bike API**: Provides bike-sharing data (e.g., bikes rented, bikes returned).
- **Tomorrow.io Weather API**: Supplies weather data (e.g., temperature, humidity, wind speed).

### Key Columns:
- `temperature`: Weather temperature in degrees Celsius.
- `humidity`: Humidity percentage.
- `windSpeed`: Wind speed in km/h.
- `bikes_rented`: Number of bikes rented in an hour.
- `bikes_returned`: Number of bikes returned in an hour.

### Preprocessing Steps:
1. Combined bike-sharing and weather data.
2. Synchronized timestamps and rounded them to 10 minute intervals.
3. Aggregated data by calculating sums or averages.
4. Handled missing values and outliers to improve model accuracy.

---

## Scatter Plots

### Insights:
- **Temperature**: Bike rentals and returns increase with higher temperatures, indicating a strong positive correlation.
- **Humidity**: Higher humidity levels correspond to fewer bike rentals and returns, but the relationship is less direct than temperature.
- **Wind Speed**: Stronger winds discourage bike usage, showing a noticeable decline in rentals and returns.

---

## Line Plots

### Features:
- Visualize trends in bike-sharing metrics alongside weather parameters over time.
- Identify patterns, such as higher bike usage during warmer periods.

---

## Heat Maps

### Features:
- Displays correlations between bike-sharing metrics and weather parameters.
- Helps identify relationships such as:
  - **Strong Positive Correlation**: Temperature with bike rentals and returns.
  - **Negative Correlation**: Humidity with bike rentals and returns.

---

## Sampling

### Stratified Sampling
- Groups data by specific conditions (e.g., daytime vs nighttime).
- Provides a general view of patterns in the dataset.
  
### Random Sampling
- Analyzes a random 10% subset of the data.
- Provides a general view of patterns in the dataset.

---

## Models

### Model Training
- Uses `Random Forest Regressor` and `Linear Regression` to predict:
  - `bikes_rented`
  - `bikes_returned`

### Model Evaluation Metrics:
### 1. **Mean Absolute Error (MAE)**
- The average absolute difference between the predicted and actual values.
- Indicates how far off predictions are from actual values, on average.
- Lower values mean better accuracy.

### 2. **Mean Squared Error (MSE)**
- The average of the squared differences between predicted and actual values.
- Penalizes larger errors more heavily than smaller ones.
- Useful for identifying significant deviations.

### 3. **Root Mean Squared Error (RMSE)**
- The square root of the Mean Squared Error.
- Provides an error metric in the same units as the target variable.
- Balances penalizing large errors with interpretability.

### 4. **R² (Coefficient of Determination)**
- Measures how well the model explains the variance in the target variable.
- Values range from 0 to 1, where higher values indicate better performance.
- An R² of 1 means the model explains all the variance in the data.

| Metric               | Bikes Rented (RF) | Bikes Rented (LR) | Bikes Returned (RF) | Bikes Returned (LR) |
|-----------------------|-------------------|-------------------|----------------------|----------------------|
| **MAE**              | 3.21              | 5.45              | 2.87                 | 4.91                 |
| **MSE**              | 12.34             | 18.87             | 11.12                | 16.44                |
| **RMSE**             | 3.51              | 4.34              | 3.33                 | 4.05                 |
| **R²**               | 0.73              | 0.51              | 0.62                 | 0.44                 |

### Visualizations:
1. **Residual Plots**:
   - Random Forest residuals are more evenly spread, indicating better performance.
   - Linear Regression residuals show trends, suggesting underfitting.
2. **Predicted vs. Actual Plots**:
   - Random Forest predictions align closely with actual values.
   - Linear Regression predictions deviate, especially at extreme values.
3. **Feature Importance**:
   - Temperature is the most influential predictor for both rentals and returns.
   - Wind speed has a larger impact on returns than rentals.
4. **McNemar's Test**:
   - Compares the correctness of predictions between Random Forest and Linear Regression to identify statistical differences.


