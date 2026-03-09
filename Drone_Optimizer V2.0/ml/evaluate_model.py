import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Load dataset
df = pd.read_csv("YOUR_DATASET_NAME.csv")

# Define features
feature_columns = [
    "distance_km",
    "payload_kg",
    "wind_speed_mps",
    "rain_mm",
    "battery_before"
]

X = df[feature_columns]
y = df["success"]

# Load trained model and scaler
model = joblib.load("logistic_model_v2.joblib")
scaler = joblib.load("scaler_v2.joblib")

# Scale
X_scaled = scaler.transform(X)

# Predict probabilities
y_probs = model.predict_proba(X_scaled)[:, 1]

# Compute ROC-AUC
auc = roc_auc_score(y, y_probs)
print("ROC-AUC:", auc)

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y, y_probs)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()