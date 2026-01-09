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
# EXP-03: Random Forest (50 Trees)
# --------------------------------------------------
rf_50 = RandomForestRegressor(n_estimators=50, random_state=42)
rf_50.fit(X_train, y_train)

y_pred_rf50 = rf_50.predict(X_test)
mse_rf50 = mean_squared_error(y_test, y_pred_rf50)
r2_rf50 = r2_score(y_test, y_pred_rf50)

print_experiment(
    "EXP-03",
    "Random Forest (50 Trees)",
    "Random Forest Regressor",
    "n_estimators=50",
    "Not required",
    "All features",
    mse_rf50,
    r2_rf50
)

joblib.dump(rf_50, "outputs/model/random_forest_50.pkl")

results.append({
    "Experiment": "EXP-03",
    "Model": "Random Forest (50 Trees)",
    "Hyperparameters": rf_50.get_params(),
    "Preprocessing": "None",
    "Feature_Selection": "All",
    "MSE": mse_rf50,
    "R2": r2_rf50
})
# --------------------------------------------------
# Save metrics
# --------------------------------------------------
with open("outputs/results/metrics.json", "w") as f:
    json.dump(results, f, indent=4)
