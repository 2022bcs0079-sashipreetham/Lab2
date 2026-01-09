import pandas as pd
import os
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------
# Helper function to print metrics in required format
# --------------------------------------------------
def print_experiment(exp_id, title, model_name, hyperparams,
                     preprocessing, feature_select,
                     mse, r2):
    print("\nEvaluation Metrics\n")
    print(f"{exp_id}: {title}")
    print(f"Model           : {model_name}")
    print(f"Hyperparameters : {hyperparams}")
    print(f"Preprocessing   : {preprocessing}")
    print(f"Feature Select  : {feature_select}")
    print("Train/Test Split: 80/20")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score        : {r2:.4f}")

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

results = []

# --------------------------------------------------
# Standardization (for Linear & Ridge)
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# EXP-02: Ridge Regression + Standardization
# --------------------------------------------------
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

y_pred_ridge = ridge.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print_experiment(
    "EXP-02",
    "Ridge Regression + Standardization",
    "Ridge Regression",
    "alpha=1.0",
    "Standardization (StandardScaler)",
    "All features",
    mse_ridge,
    r2_ridge
)

joblib.dump(ridge, "outputs/model/ridge_regression.pkl")

results.append({
    "Experiment": "EXP-02",
    "Model": "Ridge Regression",
    "Hyperparameters": ridge.get_params(),
    "Preprocessing": "StandardScaler",
    "Feature_Selection": "All",
    "MSE": mse_ridge,
    "R2": r2_ridge
})
# --------------------------------------------------
# Save metrics
# --------------------------------------------------
with open("outputs/results/metrics.json", "w") as f:
    json.dump(results, f, indent=4)
