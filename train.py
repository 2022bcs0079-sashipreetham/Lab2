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
# EXP-04: Random Forest (100 Trees + Feature Selection)
# --------------------------------------------------
rf_100 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_100.fit(X_train, y_train)

importances = rf_100.feature_importances_
top_features = (
    pd.Series(importances, index=X.columns)
    .sort_values(ascending=False)
    .head(5)
    .index.tolist()
)

X_train_fs = X_train[top_features]
X_test_fs = X_test[top_features]

rf_100_fs = RandomForestRegressor(n_estimators=100, random_state=42)
rf_100_fs.fit(X_train_fs, y_train)

y_pred_rf100 = rf_100_fs.predict(X_test_fs)
mse_rf100 = mean_squared_error(y_test, y_pred_rf100)
r2_rf100 = r2_score(y_test, y_pred_rf100)

print_experiment(
    "EXP-04",
    "Random Forest (100 Trees + Feature Selection)",
    "Random Forest Regressor",
    "n_estimators=100",
    "Not required",
    ", ".join(top_features),
    mse_rf100,
    r2_rf100
)

joblib.dump(rf_100_fs, "outputs/model/random_forest_100_fs.pkl")

results.append({
    "Experiment": "EXP-04",
    "Model": "Random Forest (100 Trees + Feature Selection)",
    "Hyperparameters": rf_100_fs.get_params(),
    "Selected_Features": top_features,
    "MSE": mse_rf100,
    "R2": r2_rf100
})
# --------------------------------------------------
# Save metrics
# --------------------------------------------------
with open("outputs/results/metrics.json", "w") as f:
    json.dump(results, f, indent=4)
