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
# EXP-01: Linear Regression + Standardization
# --------------------------------------------------
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print_experiment(
    "EXP-01",
    "Linear Regression + Standardization",
    "Linear Regression",
    "Default",
    "Standardization (StandardScaler)",
    "All features",
    mse_lr,
    r2_lr
)

joblib.dump(lr, "outputs/model/linear_regression.pkl")

results.append({
    "Experiment": "EXP-01",
    "Model": "Linear Regression",
    "Hyperparameters": lr.get_params(),
    "Preprocessing": "StandardScaler",
    "Feature_Selection": "All",
    "MSE": mse_lr,
    "R2": r2_lr
})

# --------------------------------------------------
# Save metrics
# --------------------------------------------------
with open("outputs/results/metrics.json", "w") as f:
    json.dump(results, f, indent=4)
