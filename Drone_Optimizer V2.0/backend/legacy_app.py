# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import sys
import pandas as pd
import numpy as np
import logging

# --------------------------------------------------
# Ensure project root is in path (so ml module works)
# --------------------------------------------------
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from ml.tsp.tsp import solve_tsp

logger = logging.getLogger("backend")
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# MODEL + SCALER PATHS (V2)
# --------------------------------------------------
MODEL_PATH = os.path.join(root, "ml", "logistic_model_v2.joblib")
SCALER_PATH = os.path.join(root, "ml", "scaler_v2.joblib")

model = None
model_cols = None
scaler = None


def load_model():
    global model, model_cols, scaler

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")

    data = joblib.load(MODEL_PATH)

    if isinstance(data, dict) and "model" in data:
        model = data["model"]
        model_cols = data.get("columns", None)
    else:
        raise ValueError("Model file format incorrect. Must contain model + columns.")

    scaler = joblib.load(SCALER_PATH)

    logger.info("Loaded model from %s", MODEL_PATH)
    logger.info("Loaded scaler from %s", SCALER_PATH)
    logger.info("Model columns: %s", model_cols)


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# --------------------------------------------------
# MULTI-DRONE OPTIMIZATION ENDPOINT
# --------------------------------------------------
@app.route("/optimize", methods=["POST"])
def optimize():

    data = request.get_json(force=True)
    deliveries = data.get("deliveries", [])
    min_prob = data.get("min_success_prob", 0.0)

    if model_cols is None:
        return jsonify({"error": "server model missing column metadata"}), 500

    df = pd.DataFrame(deliveries)

    if df.empty:
        return jsonify({"error": "no deliveries provided"}), 400

    missing = [c for c in model_cols if c not in df.columns]
    if missing:
        return jsonify({"error": "deliveries missing model columns", "missing": missing}), 400

    # --------------------------------------------------
    # SCALE INPUTS BEFORE PREDICTION
    # --------------------------------------------------
    X = df[model_cols].astype(float)
    X_scaled = scaler.transform(X)

    probs = model.predict_proba(X_scaled)[:, 1]
    df["prob"] = probs

    if "drone_id" not in df.columns:
        return jsonify({"error": "deliveries must include drone_id"}), 400

    results = []

    # --------------------------------------------------
    # GROUP BY DRONE
    # --------------------------------------------------
    for drone_id, group in df.groupby("drone_id"):

        filtered = group[group["prob"] >= min_prob].copy()

        if filtered.empty:
            continue

        # Multi-depot support
        depot_lat = float(group.iloc[0]["depot_lat"])
        depot_lon = float(group.iloc[0]["depot_lon"])

        base = {"lat": depot_lat, "lon": depot_lon}

        tsp_result = solve_tsp(
            deliveries=filtered.to_dict(orient="records"),
            base=base,
            min_success_prob=min_prob,
            use_2opt=True
        )

        results.append({
            "drone_id": int(drone_id),
            "depot": base,
            "coords": tsp_result.get("coords", []),
            "route_order": tsp_result.get("route_order", []),
            "route_distance_km": tsp_result.get("route_distance_km", 0.0),
            "num_stops": len(filtered)
        })

    return jsonify({"drones": results})


# --------------------------------------------------
# START SERVER
# --------------------------------------------------
if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=True)