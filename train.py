import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def main():
    # 1. Load dataset
    red_path = "dataset/winequality-red.csv"
    white_path = "dataset/winequality-white.csv"

    red = pd.read_csv(red_path, sep=";")
    white = pd.read_csv(white_path, sep=";")

    data = pd.concat([red, white], axis=0)

    # 2. Split features and target
    X = data.drop("quality", axis=1)
    y = data["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Feature Selection
    selector = SelectKBest(score_func=f_regression, k=8)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    # 5. Train model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_selected, y_train)

    # 6. Evaluation
    y_pred = model.predict(X_test_selected)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 7. Save outputs (MATCHES GITHUB ACTIONS)
    os.makedirs("outputs/model", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "selector": selector
        },
        "outputs/model/model.pkl"
    )

    metrics = {
        "Mean Squared Error": mse,
        "R2 Score": r2
    }

    with open("outputs/results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 8. Print metrics (for GitHub Actions logs)
    print("Model Evaluation Results")
    print("------------------------")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")


if __name__ == "__main__":
    main()
