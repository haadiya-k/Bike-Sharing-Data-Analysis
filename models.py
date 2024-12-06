import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from statsmodels.stats.contingency_tables import mcnemar

df = pd.read_csv('data.csv')

print('Preprocessed data from preprocessing.py')
print(df.head())

df = df.drop(columns=['Unnamed: 0'])
print(df.columns)

# Select features and targets
features = df[['temperature', 'humidity', 'windSpeed']]
targets = ['bikes_rented', 'bikes_returned']

# Initialize a dictionary to store model details
model_details = {}

# Train models for each target
for target in targets:
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

    # Initialize models
    rf_model = RandomForestRegressor(random_state=42)
    lr_model = LinearRegression()

    # Train models
    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)

    # Store models and training details
    model_details[target] = {
        'Random Forest': rf_model,
        'Linear Regression': lr_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

# Confirm successful training
print("Models trained successfully for both targets: bikes_rented and bikes_returned.")

# Define a function to evaluate models
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, predicted)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

# Initialize a dictionary to store evaluation results
evaluation_results = {}

# Iterate through each target
for target in targets:
    # Retrieve model details
    rf_model = model_details[target]['Random Forest']
    lr_model = model_details[target]['Linear Regression']
    X_test = model_details[target]['X_test']
    y_test = model_details[target]['y_test']

    # Generate predictions
    rf_predictions = rf_model.predict(X_test)
    lr_predictions = lr_model.predict(X_test)

    # Evaluate models
    rf_metrics = evaluate_model(y_test, rf_predictions)
    lr_metrics = evaluate_model(y_test, lr_predictions)

    # Store results
    evaluation_results[target] = {
        'Random Forest': rf_metrics,
        'Linear Regression': lr_metrics
    }

# Display evaluation results
print("Evaluation Results:")
print(evaluation_results)

# Organize the evaluation results into a cleaner DataFrame
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

# Convert to DataFrame for better presentation
evaluation_df = pd.DataFrame(evaluation_results_cleaned)
evaluation_results_rounded = evaluation_df.round(3)


# Create visualizations for both targets
for target in targets:
    rf_model = model_details[target]['Random Forest']
    lr_model = model_details[target]['Linear Regression']
    X_test = model_details[target]['X_test']
    y_test = model_details[target]['y_test']
    rf_predictions = rf_model.predict(X_test)
    lr_predictions = lr_model.predict(X_test)

    # Residual Plot (Random Forest)
    plt.figure(figsize=(15, 10))
    sns.residplot(x=rf_predictions, y=y_test - rf_predictions, lowess=True, color="green", line_kws={"color": "red"})
    plt.title(f"Residual Plot for Random Forest ({target})")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.show()

    # Residual Plot (Linear Regression)
    plt.figure(figsize=(15, 10))
    sns.residplot(x=lr_predictions, y=y_test - lr_predictions, lowess=True, color="green", line_kws={"color": "red"})
    plt.title(f"Residual Plot for Linear Regression ({target})")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.show()


# Create visualizations for both targets
for target in targets:
    # Extract data for the current target
    y_test = model_details[target]['y_test']
    rf_predictions = model_details[target]['Random Forest'].predict(model_details[target]['X_test'])
    lr_predictions = model_details[target]['Linear Regression'].predict(model_details[target]['X_test'])

    # Prediction vs Actual Scatter Plot
    plt.figure(figsize=(15, 10))
    plt.scatter(y_test, rf_predictions, alpha=0.6, label="Random Forest", color='blue')
    plt.scatter(y_test, lr_predictions, alpha=0.6, label="Linear Regression", color='orange')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Perfect Fit")
    plt.title(f"Prediction vs Actual for {target}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.show()

# Feature Importance (Random Forest)
target = "bikes_rented"  # Replace with "bikes_returned" for the other target
rf_model = model_details[target]['Random Forest']
feature_importance = rf_model.feature_importances_

# Sort the feature importance values
sorted_idx = feature_importance.argsort()

# Plot the feature importance
plt.figure(figsize=(8, 6))
plt.barh(features.columns[sorted_idx], feature_importance[sorted_idx], color="skyblue")
plt.title(f"Feature Importance for Random Forest ({target})")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# Feature Importance (Random Forest)
target = "bikes_returned"
rf_model = model_details[target]['Random Forest']
feature_importance = rf_model.feature_importances_

# Sort the feature importance values
sorted_idx = feature_importance.argsort()

# Plot the feature importance
plt.figure(figsize=(8, 6))
plt.barh(features.columns[sorted_idx], feature_importance[sorted_idx], color="skyblue")
plt.title(f"Feature Importance for Random Forest ({target})")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# Convert regression targets to discrete categories (e.g., Low, Medium, High)
k_bins = 3  # Define number of bins
discretizer = KBinsDiscretizer(n_bins=k_bins, encode='ordinal', strategy='uniform')

def analyze_target_with_mcnemar(target):
    # Extract test data and predictions
    y_test = model_details[target]['y_test']
    rf_predictions = model_details[target]['Random Forest'].predict(model_details[target]['X_test'])
    lr_predictions = model_details[target]['Linear Regression'].predict(model_details[target]['X_test'])

    # Discretize actual and predicted values
    y_test_binned = discretizer.fit_transform(y_test.values.reshape(-1, 1)).flatten()
    rf_predictions_binned = discretizer.transform(rf_predictions.reshape(-1, 1)).flatten()
    lr_predictions_binned = discretizer.transform(lr_predictions.reshape(-1, 1)).flatten()

    # Create binary outcomes for correct/incorrect predictions
    correct_rf = (y_test_binned == rf_predictions_binned)
    correct_lr = (y_test_binned == lr_predictions_binned)

    # Create the contingency table
    both_correct = sum(correct_rf & correct_lr)
    rf_only_correct = sum(correct_rf & ~correct_lr)
    lr_only_correct = sum(correct_lr & ~correct_rf)
    both_wrong = sum(~correct_rf & ~correct_lr)

    contingency_table = [[both_correct, rf_only_correct],
                         [lr_only_correct, both_wrong]]

    # Perform McNemar's Test
    result = mcnemar(contingency_table, exact=True)

    # Return results
    return {
        "Target": target,
        "Contingency Table": contingency_table,
        "McNemar Test Statistic": result.statistic,
        "P-Value": result.pvalue
}

# Analyze both targets
results_bikes_rented = analyze_target_with_mcnemar("bikes_rented")
results_bikes_returned = analyze_target_with_mcnemar("bikes_returned")

# Display results
[results_bikes_rented, results_bikes_returned]