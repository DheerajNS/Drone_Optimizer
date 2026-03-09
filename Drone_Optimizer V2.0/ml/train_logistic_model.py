import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


# -----------------------------
# Sigmoid Function
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# -----------------------------
# Generate Training Data
# -----------------------------
def generate_training_data(num_samples=5000):

    np.random.seed(42)

    data = []

    for _ in range(num_samples):

        # Geographic realistic range (1–12 km)
        distance_km = np.random.uniform(1, 12)

        # Payload distribution
        payload_kg = np.random.uniform(0.3, 5)

        # Weather
        wind_speed_mps = np.random.uniform(1, 12)
        rain_mm = np.random.uniform(0, 20)

        # Battery
        battery_before = np.random.uniform(70, 100)

        # -----------------------------
        # Physics-Inspired Risk Score
        # -----------------------------
        risk = (
            (distance_km * 0.4)
            + (payload_kg * 0.3)
            + (wind_speed_mps * 0.5)
            + (rain_mm * 0.2)
            - (battery_before * 0.05)
        )

        # Add controlled Gaussian noise
        risk += np.random.normal(0, 1.5)

        probability = sigmoid(risk / 8)

        success = 1 if probability > 0.5 else 0

        data.append([
            distance_km,
            payload_kg,
            wind_speed_mps,
            rain_mm,
            battery_before,
            success
        ])

    columns = [
        "distance_km",
        "payload_kg",
        "wind_speed_mps",
        "rain_mm",
        "battery_before",
        "success"
    ]

    return pd.DataFrame(data, columns=columns)


# -----------------------------
# Train Model
# -----------------------------
def train_model():

    df = generate_training_data(5000)

    X = df.drop("success", axis=1)
    y = df["success"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\nModel Performance:")
    print("Accuracy:", round(accuracy, 4))
    print("ROC-AUC:", round(roc_auc, 4))

    joblib.dump(model, "logistic_model_v2.joblib")
    joblib.dump(scaler, "scaler_v2.joblib")

    print("\nModel and scaler saved successfully.")


if __name__ == "__main__":
    train_model()