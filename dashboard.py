import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import KBinsDiscretizer
from statsmodels.stats.contingency_tables import mcnemar

# Raw Data
raw_bikes_df = pd.read_csv('times_square_rental_data.csv')
raw_weather_df = pd.read_csv('times_square_weather_forecast.csv')

# Load the preprocessed dataset
data = pd.read_csv('data.csv')
data = data.drop(columns=['Unnamed: 0'])

data['rounded_time'] = pd.to_datetime(data['rounded_time'])

# Streamlit Dashboard Layout
# Sidebar Navigation
st.sidebar.title("Navigation")
pages = ["Home", "Data", "Scatter Plots", "Line Plots", "Heat Maps", "Sampling", "Models"]
selected_page = st.sidebar.radio("Go to", pages)

# Home Page
if selected_page == "Home":
    st.title("Bike Sharing and Weather Analysis Dashboard")
    st.markdown("""
    Welcome to the **Bike Sharing and Weather Analysis Dashboard**! This interactive platform provides insights into how weather conditions 
    influence bike rentals and returns in Times Square. 

    ### Features:
    - **Data**: View and interact with each dataset.
    - **Scatter Plots**: Explore the relationship between bike usage and weather parameters such as temperature, humidity, and wind speed.
    - **Line Plots**: Analyze trends in bike usage and weather conditions over time.
    - **Heat Maps**: Discover correlations between different variables.
    - **Sampling**: Explore stratified or random sampling.
    - **Models**: Train and evaluate machine learning models (Random Forest and Linear Regression) to predict bike usage.

    ### Navigation:
    Use the **sidebar** to switch between different sections of the dashboard. Each page provides visualizations and insights for the selected analysis.

    ### About:
    This dashboard integrates weather and bike-sharing data to help uncover patterns and trends that can be used to optimize bike-sharing operations or make weather-informed decisions.
    """)

# Scatter Plot
if selected_page == "Scatter Plots":
    st.title("Scatter Plots")
    st.header("Bike Usage vs Weather")
    st.markdown("Explore how weather conditions affect bike rentals and returns.")

    weather_parameter = st.selectbox("Choose a weather parameter to compare with bike usage:",
                                     options=["temperature", "humidity", "windSpeed"])

    fig1 = px.scatter(data,
                      x=weather_parameter,
                      y=["bikes_rented", "bikes_returned"],
                      labels={"value": "Bike Rentals/Returns", weather_parameter: weather_parameter.capitalize()},
                      title=f"Bike Usage vs. {weather_parameter.capitalize()}")
    st.plotly_chart(fig1)

    if weather_parameter == "temperature":
        st.write("""
        **Insights**:
        - This scatter plot illustrates the number of bikes rented (in dark blue) and returned (in light blue) against varying temperatures.
        - A positive trend is visible, where bike rentals and returns increase with higher temperatures.
        - This suggests that warmer weather encourages more people to use the bike-sharing service, indicating a strong positive correlation between temperature and bike usage.
        """)

    elif weather_parameter == "humidity":
        st.write("""
        **Insights**:
        - This scatter plot shows bike rentals (in green) and returns (in yellow) in relation to different humidity levels.
        - The plot indicates a slight decrease in bike usage as humidity rises, particularly at higher humidity levels.
        - This suggests that higher humidity may discourage bike rentals and returns, although the relationship appears less consistent than with temperature.
        """)

    elif weather_parameter == "windSpeed":
        st.write("""
        **Insights**:
        - This scatter plot shows the number of bikes rented (in orange) and returned (in red) against varying wind speeds.
        - There is a noticeable decline in bike rentals and returns as wind speed increases.
        - Higher wind speeds appear to correlate with reduced bike usage, suggesting that windy conditions may deter people from renting or returning bikes.
        """)

# Line Plot
if selected_page == "Line Plots":
    st.title("Line Plots")
    st.header("Bike Usage and Weather Over Time")
    st.markdown("Analyze trends in bike usage and weather conditions over time.")

    weather_parameter = st.selectbox("Choose a weather parameter to visualize over time:",
                                     options=["temperature", "humidity", "windSpeed"])

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=data['rounded_time'], y=data['bikes_rented'],
                              mode='lines', name='Bikes Rented'))
    fig2.add_trace(go.Scatter(x=data['rounded_time'], y=data['bikes_returned'],
                              mode='lines', name='Bikes Returned'))
    fig2.add_trace(go.Scatter(x=data['rounded_time'], y=data[weather_parameter],
                              mode='lines', name=weather_parameter.capitalize(), yaxis="y2"))

    fig2.update_layout(
        title=f"Bike Usage vs. {weather_parameter.capitalize()}",
        xaxis_title="Time",
        yaxis=dict(title="Number of Bikes"),
        yaxis2=dict(title=weather_parameter.capitalize(), overlaying="y", side="right"),
        hovermode="x unified"
    )
    st.plotly_chart(fig2)

    if weather_parameter == "temperature":
        st.write("""
        **Insights**:
        - This line plot shows how bike rentals and returns varied with temperature across the time period. 
        - It is observed there is a clear increase in bike rentals and returns on days with higher temperatures. 
        - Warmer days showed a higher volume of bike usage, suggesting that favorable temperatures encouraged more people to use the bike-sharing service.
        """)

    elif weather_parameter == "humidity":
        st.write("""
        **Insights**:
        - This plot displays the relationship between bike rentals and returns over time in relation to humidity levels. 
        - It is observed that bike rentals and returns generally fluctuated with humidity levels, though the impact was less direct than with temperature. 
        - Higher humidity correlates with a dip in rentals and returns, suggesting that elevated humidity may discourage bike usage.
        """)

    elif weather_parameter == "windSpeed":
        st.write("""
        **Insights**:
        - This line plot shows how bike rentals and returns varied with wind speed across the time period. 
        - The plot shows a noticeable decrease in bike rentals and returns as wind speed increases. 
        - Higher wind speeds appeared to coincide with lower bike usage, indicating that strong winds likely discourage people from renting or returning bikes due to discomfort or safety concerns.
        """)

# Heatmap Tab
if selected_page == "Heat Maps":
    st.title("Heat Map")
    st.header("Correlation Heatmap: Bike Usage vs Weather")
    st.markdown("Discover correlations between bike usage and weather parameters.")

    corr_matrix = data[['bikes_rented', 'bikes_returned', 'temperature', 'humidity', 'windSpeed']].corr()

    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Raw Data and Statistics
if selected_page == "Data":
    st.title("Data")
    st.header("Raw Data, Preprocessed Data, and Descriptive Statistics")
    st.markdown("Explore the raw data and the preprocessed data used in this analysis, along with summary statistics.")

    st.subheader("Raw Bike Usage Data")
    st.dataframe(raw_bikes_df)
    st.subheader("Raw Weather Data")
    st.dataframe(raw_weather_df)

    st.write("### Preprocessed Data")
    st.dataframe(data)

    st.write("### Descriptive Statistics of Preprocessed Data")
    st.dataframe(data.describe())

# Sampling
if selected_page == "Sampling":
    st.title("Sampling Techniques")
    st.header("Sampling: Stratified vs Random Sampling")
    st.markdown("""
    Explore the effects of different sampling techniques on the data. This page shows **stratified** and **random** sampling methods applied to the dataset, with scatter plots for visualizing the samples.
    """)

    st.write("""
    **Note**: 
    - **Stratified sampling**: Based on the daytime vs nighttime split.
    - **Random sampling**: Simple random selection of 10% of the dataset.
    """)

    with st.expander("Stratified Sampling"):
        stratified_sample_capped = data.copy()
        stratified_sample_capped['day_period'] = stratified_sample_capped['rounded_time'].dt.hour.apply(
            lambda x: 'daytime' if 8 <= x < 20 else 'nighttime')
        stratified_sample = stratified_sample_capped.groupby('day_period', group_keys=False).apply(
            lambda x: x.sample(frac=0.1, random_state=1)).reset_index(drop=True)

        st.write("### Stratified Sample")
        st.dataframe(stratified_sample)

        # Scatter Plots for Stratified Sample
        st.write("#### Scatter Plots: Stratified Sample")
        fig3, axs = plt.subplots(2, 2, figsize=(14, 10))

        axs[0, 0].scatter(stratified_sample['temperature'], stratified_sample['bikes_rented'], color='blue',
                          label='Bikes Rented', alpha=0.5)
        axs[0, 0].scatter(stratified_sample['temperature'], stratified_sample['bikes_returned'], color='cyan',
                          label='Bikes Returned', alpha=0.5)
        axs[0, 0].set_title('Bikes Rented/Returned vs Temperature')
        axs[0, 0].set_xlabel('Temperature (°C)')
        axs[0, 0].legend()

        axs[0, 1].scatter(stratified_sample['humidity'], stratified_sample['bikes_rented'], color='green',
                          label='Bikes Rented', alpha=0.5)
        axs[0, 1].scatter(stratified_sample['humidity'], stratified_sample['bikes_returned'], color='orange',
                          label='Bikes Returned', alpha=0.5)
        axs[0, 1].set_title('Bikes Rented/Returned vs Humidity')
        axs[0, 1].set_xlabel('Humidity (%)')
        axs[0, 1].legend()

        axs[1, 0].scatter(stratified_sample['windSpeed'], stratified_sample['bikes_rented'], color='orange',
                          label='Bikes Rented', alpha=0.5)
        axs[1, 0].scatter(stratified_sample['windSpeed'], stratified_sample['bikes_returned'], color='red',
                          label='Bikes Returned', alpha=0.5)
        axs[1, 0].set_title('Bikes Rented/Returned vs Wind Speed')
        axs[1, 0].set_xlabel('Wind Speed (kph)')
        axs[1, 0].legend()

        fig3.delaxes(axs[1, 1])
        plt.tight_layout()
        st.pyplot(fig3)

        st.write("The trends observed in these plots were similar to those in the original dataset, "
                 "indicating that the stratified sample accurately reflects the overall patterns in bike usage across different weather conditions.")

    # Random Sampling
    with st.expander("Random Sampling"):
        random_sample = data.sample(frac=0.1, random_state=1)

        st.write("### Random Sample")
        st.dataframe(random_sample)

        # Scatter Plots for Random Sample
        st.write("#### Scatter Plots: Random Sample")
        fig4, axs = plt.subplots(2, 2, figsize=(14, 10))

        axs[0, 0].scatter(random_sample['temperature'], random_sample['bikes_rented'], color='blue',
                          label='Bikes Rented', alpha=0.5)
        axs[0, 0].scatter(random_sample['temperature'], random_sample['bikes_returned'], color='cyan',
                          label='Bikes Returned', alpha=0.5)
        axs[0, 0].set_title('Bikes Rented/Returned vs Temperature')
        axs[0, 0].set_xlabel('Temperature (°C)')
        axs[0, 0].legend()

        axs[0, 1].scatter(random_sample['humidity'], random_sample['bikes_rented'], color='green', label='Bikes Rented',
                          alpha=0.5)
        axs[0, 1].scatter(random_sample['humidity'], random_sample['bikes_returned'], color='orange',
                          label='Bikes Returned', alpha=0.5)
        axs[0, 1].set_title('Bikes Rented/Returned vs Humidity')
        axs[0, 1].set_xlabel('Humidity (%)')
        axs[0, 1].legend()

        axs[1, 0].scatter(random_sample['windSpeed'], random_sample['bikes_rented'], color='orange',
                          label='Bikes Rented', alpha=0.5)
        axs[1, 0].scatter(random_sample['windSpeed'], random_sample['bikes_returned'], color='red',
                          label='Bikes Returned', alpha=0.5)
        axs[1, 0].set_title('Bikes Rented/Returned vs Wind Speed')
        axs[1, 0].set_xlabel('Wind Speed (kph)')
        axs[1, 0].legend()

        fig4.delaxes(axs[1, 1])
        plt.tight_layout()
        st.pyplot(fig4)

        st.write("The trends observed in these plots were similar to those in the original dataset, "
                 "indicating that the random sample accurately reflects the overall patterns in bike usage across different weather conditions.")

# Model
if selected_page == "Models":
    st.title("Models")
    st.header("Model Training and Evaluation")
    st.markdown("""
    To predict bike usage, specifically the number of bikes rented and returned during given time intervals, 
    two regression models were developed: Random Forest Regressor and Linear Regression. 
    These models were chosen to analyse the relationship between weather conditions and bike usage, 
    leveraging their complementary strengths to gain comprehensive insights into the data.
    This section shows the evaluation of both models.
    It also includes feature importance, prediction vs. actual plots, residual plots, and McNemar's test for model comparison.
    """)

    # Select features and targets
    features = data[['temperature', 'humidity', 'windSpeed']]
    targets = ['bikes_rented', 'bikes_returned']

    # Dictionary to store model details
    model_details = {}
    evaluation_results = {}

    # Train models for each target
    for target in targets:
        y = data[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

        # Initialize models
        rf_model = RandomForestRegressor(random_state=42)
        lr_model = LinearRegression()

        # Train models
        rf_model.fit(X_train, y_train)
        lr_model.fit(X_train, y_train)

        # Store model details
        model_details[target] = {
            'Random Forest': rf_model,
            'Linear Regression': lr_model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

        # Generate predictions
        rf_predictions = rf_model.predict(X_test)
        lr_predictions = lr_model.predict(X_test)


        # Evaluate models
        def evaluate_model(true, predicted):
            mae = mean_absolute_error(true, predicted)
            mse = mean_squared_error(true, predicted)
            rmse = np.sqrt(mse)
            r2 = r2_score(true, predicted)
            return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


        evaluation_results[target] = {
            'Random Forest': evaluate_model(y_test, rf_predictions),
            'Linear Regression': evaluate_model(y_test, lr_predictions)
        }

    # Model Evaluation Results
    with st.expander("Model Evaluation Results"):
        st.subheader("Evaluation Results")
        evaluation_results_cleaned = {
            "Metric": ["MAE", "MSE", "RMSE", "R^2"],
            "Bikes Rented (Random Forest)": [
                evaluation_results['bikes_rented']['Random Forest']['MAE'],
                evaluation_results['bikes_rented']['Random Forest']['MSE'],
                evaluation_results['bikes_rented']['Random Forest']['RMSE'],
                evaluation_results['bikes_rented']['Random Forest']['R2']
            ],
            "Bikes Rented (Linear Regression)": [
                evaluation_results['bikes_rented']['Linear Regression']['MAE'],
                evaluation_results['bikes_rented']['Linear Regression']['MSE'],
                evaluation_results['bikes_rented']['Linear Regression']['RMSE'],
                evaluation_results['bikes_rented']['Linear Regression']['R2']
            ],
            "Bikes Returned (Random Forest)": [
                evaluation_results['bikes_returned']['Random Forest']['MAE'],
                evaluation_results['bikes_returned']['Random Forest']['MSE'],
                evaluation_results['bikes_returned']['Random Forest']['RMSE'],
                evaluation_results['bikes_returned']['Random Forest']['R2']
            ],
            "Bikes Returned (Linear Regression)": [
                evaluation_results['bikes_returned']['Linear Regression']['MAE'],
                evaluation_results['bikes_returned']['Linear Regression']['MSE'],
                evaluation_results['bikes_returned']['Linear Regression']['RMSE'],
                evaluation_results['bikes_returned']['Linear Regression']['R2']
            ],
        }

        # Convert to DataFrame for display
        evaluation_df = pd.DataFrame(evaluation_results_cleaned)
        st.dataframe(evaluation_df)

        st.write("""
        **Insights:**
        - Random Forest consistently outperforms Linear Regression across all metrics. It achieves lower error values (MAE, MSE, RMSE) and significantly higher R² scores.

        **Target-Specific Observations:**
        - **Bikes Rented**: Random Forest captures more variance (73.1% vs. 51.1%), showing a clear advantage over Linear Regression.
        - **Bikes Returned**: Although the gap is smaller, Random Forest still explains significantly more variance (62.3% vs. 44.0%).

        **Additional:**
        - Random Forest effectively models non-linear relationships between features (e.g., weather conditions) and target variables, which Linear Regression struggles to handle.
        """)

    # Feature Importance
    with st.expander("Feature Importance (Random Forest)"):
        st.subheader("Feature Importance (Random Forest)")
        for target in targets:
            rf_model = model_details[target]['Random Forest']
            feature_importance = rf_model.feature_importances_

            # Sort feature importance
            sorted_idx = feature_importance.argsort()
            plt.figure(figsize=(8, 6))
            plt.barh(features.columns[sorted_idx], feature_importance[sorted_idx], color="skyblue")
            plt.title(f"Feature Importance for Random Forest ({target})")
            plt.xlabel("Importance")
            plt.ylabel("Features")
            st.pyplot(plt)
            plt.clf()

        st.write("""
        **Insights:**

        **Bikes Rented**
        - **Temperature** is the most important feature, contributing significantly to the model's predictive power (importance > 0.6).
        - **Humidity** and **Wind Speed** are less influential but still contribute meaningfully, with humidity slightly more important than wind speed.

        **Bikes Returned**
        - **Temperature** again dominates as the most significant predictor (importance ~0.5).
        - **Wind Speed** has a more considerable influence compared to its role in predicting bikes_rented, showing that environmental factors like wind speed may more directly impact returns.
        - **Humidity** has the least impact but still plays a role.
        """)

    # Prediction vs. Actual Plots
    with st.expander("Prediction vs. Actual Plots (Random Forest & Linear Regression)"):
        st.subheader("Prediction vs Actual")
        for target in targets:
            rf_model = model_details[target]['Random Forest']
            lr_model = model_details[target]['Linear Regression']
            X_test = model_details[target]['X_test']
            y_test = model_details[target]['y_test']

            rf_predictions = rf_model.predict(X_test)
            lr_predictions = lr_model.predict(X_test)

            # Prediction vs Actual Scatter Plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, rf_predictions, alpha=0.6, label="Random Forest", color='blue')
            plt.scatter(y_test, lr_predictions, alpha=0.6, label="Linear Regression", color='orange')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--",
                     label="Perfect Fit")
            plt.title(f"Prediction vs Actual for {target}")
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.legend()
            st.pyplot(plt)
            plt.clf()

        st.write("""
        **Insights:**

        **Bikes Rented**
        - The **Random Forest** predictions (blue points) align more closely with the diagonal "perfect fit" line, indicating better performance and less deviation from the actual values.
        - The **Linear Regression** predictions (orange points) show more scatter, particularly at the higher range of actual values. This suggests that the linear model struggles to capture the complexity of the data for bikes_rented.

        **Bikes Returned**
        - Similar to bikes_rented, the **Random Forest** predictions (blue points) generally stay closer to the diagonal line, indicating better alignment with actual values.
        - The **Linear Regression** predictions (orange points) show significant deviation at the higher actual values, reinforcing that the linear model doesn't fit as well in this case either.
        """)

    # Residual Plots
    with st.expander("Residual Plots (Random Forest & Linear Regression)"):
        st.subheader("Residual Plots")
        for target in targets:
            rf_model = model_details[target]['Random Forest']
            lr_model = model_details[target]['Linear Regression']
            X_test = model_details[target]['X_test']
            y_test = model_details[target]['y_test']

            rf_predictions = rf_model.predict(X_test)
            lr_predictions = lr_model.predict(X_test)

            # Residual Plot for Random Forest
            plt.figure(figsize=(10, 6))
            sns.residplot(x=rf_predictions, y=y_test - rf_predictions, lowess=True, color="green",
                          line_kws={"color": "red"})
            plt.title(f"Residual Plot for Random Forest ({target})")
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            plt.axhline(0, color='black', linestyle='--', linewidth=1)
            st.pyplot(plt)
            plt.clf()

            # Residual Plot for Linear Regression
            plt.figure(figsize=(10, 6))
            sns.residplot(x=lr_predictions, y=y_test - lr_predictions, lowess=True, color="blue",
                          line_kws={"color": "orange"})
            plt.title(f"Residual Plot for Linear Regression ({target})")
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            plt.axhline(0, color='black', linestyle='--', linewidth=1)
            st.pyplot(plt)
            plt.clf()

        st.write("""
        **Insights:**

        **Random Forest**
        - **Bikes Rented**:
            - Residuals seem fairly evenly spread around 0, which suggests that the model doesn't exhibit strong bias.
            - The red line's slight curvature indicates some non-linearity that the model hasn't fully captured but still performs decently.
        - **Bikes Returned**:
            - Like the previous Random Forest bikes_rented plot, the residuals appear fairly random and centered around 0, which indicates good performance.

        **Linear Regression**
        - **Bikes Rented**:
            - The residuals have a noticeable trend (non-random spread) around the red line, especially at lower and higher predicted values.
            - This suggests that the linear model struggles with capturing some of the underlying relationships in the data.
        - **Bikes Returned**:
            - Again, a noticeable trend in residuals (especially at the extremes of predicted values) indicates that a simple linear model doesn't fit the data as well as the Random Forest.
        """)

    # McNemar's Test
    with st.expander("McNemar's Test"):

        st.write("""
        ### McNemar's Test Analysis
        McNemar’s Test was performed to compare the performance of the **Random Forest** and **Linear Regression** models for predicting **bikes_rented** and **bikes_returned**. 
        The test evaluates whether there is a statistically significant difference in the correctness of predictions between the two models. To adapt this test for regression, continuous predictions were discretized into categories (e.g., low, medium, high) to allow for a direct comparison of correct and incorrect classifications.
        """)

        discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')


        def analyze_target_with_mcnemar(target):
            y_test = model_details[target]['y_test']
            rf_predictions = model_details[target]['Random Forest'].predict(model_details[target]['X_test'])
            lr_predictions = model_details[target]['Linear Regression'].predict(model_details[target]['X_test'])

            # Discretize actual and predicted values
            y_test_binned = discretizer.fit_transform(y_test.values.reshape(-1, 1)).flatten()
            rf_predictions_binned = discretizer.transform(rf_predictions.reshape(-1, 1)).flatten()
            lr_predictions_binned = discretizer.transform(lr_predictions.reshape(-1, 1)).flatten()

            # Contingency Table
            correct_rf = (y_test_binned == rf_predictions_binned)
            correct_lr = (y_test_binned == lr_predictions_binned)

            both_correct = sum(correct_rf & correct_lr)
            rf_only_correct = sum(correct_rf & ~correct_lr)
            lr_only_correct = sum(correct_lr & ~correct_rf)
            both_wrong = sum(~correct_rf & ~correct_lr)

            contingency_table = [
                [int(both_correct), int(rf_only_correct)],
                [int(lr_only_correct), int(both_wrong)]
            ]

            # Perform McNemar's Test
            result = mcnemar(contingency_table, exact=True)

            return {
                "Contingency Table": contingency_table,
                "McNemar Test Statistic": result.statistic,
                "P-Value": result.pvalue
            }


        # Analyze for both targets
        for target in targets:
            results = analyze_target_with_mcnemar(target)
            contingency_table = results["Contingency Table"]

            # Display results in a formatted way
            st.markdown(f"### **{target.replace('_', ' ').capitalize()}**")
            st.markdown("#### Contingency Table:")
            st.markdown(f"""
            ```
            Both Correct:         {contingency_table[0][0]}
            Random Forest Only:   {contingency_table[0][1]}
            Linear Regression Only:  {contingency_table[1][0]}
            Both Wrong:           {contingency_table[1][1]}
            ```
            """)
            st.markdown(f"#### McNemar Test Statistic: `{results['McNemar Test Statistic']}`")
            st.markdown(f"#### P-Value: `{results['P-Value']}`")
