import os
import sys
import joblib
import pandas as pd

# Allow backend to access ml folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.dataset_generator import generate_dataset


# -----------------------------
# Load Model & Scaler
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(BASE_DIR, "ml", "logistic_model_v2.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "ml", "scaler_v2.joblib")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# -----------------------------
# Prediction Function
# -----------------------------
def run_prediction(threshold=0.7):

    # Step 1: Generate fresh raw dataset
    df = generate_dataset(num_drones=5, stops_per_drone=5)

    # Step 2: Select features (IMPORTANT: must match training order)
    feature_columns = [
        "distance_km",
        "payload_kg",
        "wind_speed_mps",
        "rain_mm",
        "battery_before"
    ]

    X = df[feature_columns]

    # Step 3: Scale features
    X_scaled = scaler.transform(X)

    # Step 4: Predict probabilities
    probabilities = model.predict_proba(X_scaled)[:, 1]

    df["success_probability"] = probabilities

    # Step 5: Apply threshold filter
    filtered_df = df[df["success_probability"] >= threshold].copy()

    return df, filtered_df


if __name__ == "__main__":
    full_df, safe_df = run_prediction(threshold=0.7)

    print("\n--- FULL DATA ---\n")
    print(full_df[["drone_id", "distance_km", "success_probability"]])

    print("\n--- SAFE STOPS AFTER THRESHOLD ---\n")
    print(safe_df[["drone_id", "distance_km", "success_probability"]])
def run_prediction_on_df(df, threshold=0.7):

    feature_columns = [
        "distance_km",
        "payload_kg",
        "wind_speed_mps",
        "rain_mm",
        "battery_before"
    ]

    X = df[feature_columns]
    X_scaled = scaler.transform(X)

    probabilities = model.predict_proba(X_scaled)[:, 1]
    df["success_probability"] = probabilities

    filtered_df = df[df["success_probability"] >= threshold].copy()

    return df, filtered_df