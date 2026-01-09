import pandas as pd
import numpy as np
import os
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------
# Create output directories
# --------------------------------------------------
os.makedirs("outputs/model", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
data = pd.read_csv("dataset/winequality-red.csv", sep=";")

X = data.drop("quality", axis=1)
y = data["quality"]

# --------------------------------------------------
# Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

experiment_results = []

# --------------------------------------------------
# Standardization (shared)
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# 1️⃣ Linear Regression
# --------------------------------------------------
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

lr_result = {
    "model_name": "Linear Regression",
    "parameters": lr.get_params(),
    "metrics": {
        "MSE": mean_squared_error(y_test, y_pred_lr),
        "R2": r2_score(y_test, y_pred_lr)
    }
}

experiment_results.append(lr_result)
joblib.dump(lr, "outputs/model/linear_regression.pkl")

print("\nMODEL: Linear Regression")
print(lr_result)

# --------------------------------------------------
# 2️⃣ Ridge Regression
# --------------------------------------------------
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

y_pred_ridge = ridge.predict(X_test_scaled)

ridge_result = {
    "model_name": "Ridge Regression",
    "parameters": ridge.get_params(),
    "metrics": {
        "MSE": mean_squared_error(y_test, y_pred_ridge),
        "R2": r2_score(y_test, y_pred_ridge)
    }
}

experiment_results.append(ridge_result)
joblib.dump(ridge, "outputs/model/ridge_regression.pkl")

print("\nMODEL: Ridge Regression")
print(ridge_result)

# --------------------------------------------------
# 3️⃣ Random Forest (50 Trees)
# --------------------------------------------------
rf_50 = RandomForestRegressor(
    n_estimators=50,
    random_state=42
)

rf_50.fit(X_train, y_train)
y_pred_rf50 = rf_50.predict(X_test)

rf50_result = {
    "model_name": "Random Forest (50 Trees)",
    "parameters": rf_50.get_params(),
    "metrics": {
        "MSE": mean_squared_error(y_test, y_pred_rf50),
        "R2": r2_score(y_test, y_pred_rf50)
    }
}

experiment_results.append(rf50_result)
joblib.dump(rf_50, "outputs/model/random_forest_50.pkl")

print("\nMODEL: Random Forest (50 Trees)")
print(rf50_result)

# --------------------------------------------------
# 4️⃣ Random Forest (100 Trees + Feature Selection)
# --------------------------------------------------
rf_100 = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_100.fit(X_train, y_train)

# Feature importance
importances = rf_100.feature_importances_
top_features = (
    pd.Series(importances, index=X.columns)
    .sort_values(ascending=False)
    .head(5)
    .index.tolist()
)

X_train_fs = X_train[top_features]
X_test_fs = X_test[top_features]

rf_100_fs = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_100_fs.fit(X_train_fs, y_train)

y_pred_rf100 = rf_100_fs.predict(X_test_fs)

rf100_result = {
    "model_name": "Random Forest (100 Trees + Feature Selection)",
    "parameters": rf_100_fs.get_params(),
    "selected_features": top_features,
    "metrics": {
        "MSE": mean_squared_error(y_test, y_pred_rf100),
        "R2": r2_score(y_test, y_pred_rf100)
    }
}

experiment_results.append(rf100_result)
joblib.dump(rf_100_fs, "outputs/model/random_forest_100_fs.pkl")

print("\nMODEL: Random Forest (100 Trees + Feature Selection)")
print(rf100_result)

# --------------------------------------------------
# Save experiment tracking file
# --------------------------------------------------
with open("outputs/results/metrics.json", "w") as f:
    json.dump(experiment_results, f, indent=4)
