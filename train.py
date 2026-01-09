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

results = {}

# --------------------------------------------------
# 1️⃣ Linear Regression (with Standardization)
# --------------------------------------------------
scaler_lr = StandardScaler()
X_train_lr = scaler_lr.fit_transform(X_train)
X_test_lr = scaler_lr.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_lr, y_train)

y_pred_lr = lr.predict(X_test_lr)

results["Linear_Regression"] = {
    "MSE": mean_squared_error(y_test, y_pred_lr),
    "R2": r2_score(y_test, y_pred_lr)
}

joblib.dump(lr, "outputs/model/linear_regression.pkl")

print("Linear Regression")
print(results["Linear_Regression"])

# --------------------------------------------------
# 2️⃣ Ridge Regression (with Standardization)
# --------------------------------------------------
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_lr, y_train)

y_pred_ridge = ridge.predict(X_test_lr)

results["Ridge_Regression"] = {
    "MSE": mean_squared_error(y_test, y_pred_ridge),
    "R2": r2_score(y_test, y_pred_ridge)
}

joblib.dump(ridge, "outputs/model/ridge_regression.pkl")

print("\nRidge Regression")
print(results["Ridge_Regression"])

# --------------------------------------------------
# 3️⃣ Random Forest (50 Trees)
# --------------------------------------------------
rf_50 = RandomForestRegressor(
    n_estimators=50,
    random_state=42
)

rf_50.fit(X_train, y_train)
y_pred_rf50 = rf_50.predict(X_test)

results["Random_Forest_50"] = {
    "MSE": mean_squared_error(y_test, y_pred_rf50),
    "R2": r2_score(y_test, y_pred_rf50)
}

joblib.dump(rf_50, "outputs/model/random_forest_50.pkl")

print("\nRandom Forest (50 Trees)")
print(results["Random_Forest_50"])

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
feature_names = X.columns

# Select top 5 features
top_features = pd.Series(importances, index=feature_names)\
                  .sort_values(ascending=False)\
                  .head(5).index.tolist()

X_train_fs = X_train[top_features]
X_test_fs = X_test[top_features]

rf_100_fs = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_100_fs.fit(X_train_fs, y_train)
y_pred_rf100 = rf_100_fs.predict(X_test_fs)

results["Random_Forest_100_Feature_Selected"] = {
    "Selected_Features": top_features,
    "MSE": mean_squared_error(y_test, y_pred_rf100),
    "R2": r2_score(y_test, y_pred_rf100)
}

joblib.dump(rf_100_fs, "outputs/model/random_forest_100_fs.pkl")

print("\nRandom Forest (100 Trees + Feature Selection)")
print(results["Random_Forest_100_Feature_Selected"])

# --------------------------------------------------
# Save metrics
# --------------------------------------------------
with open("outputs/results/metrics.json", "w") as f:
    json.dump(results, f, indent=4)
