# back_end/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd
import numpy as np
from ml.tsp.tsp import solve_tsp  # we will make tsp a module under ml/tsp
import logging

logger = logging.getLogger("backend")
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "drone_success_model.joblib")
# Fallback: allow ../ml/drone_success_model.joblib
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "drone_success_model.joblib")

model = None
model_cols = None

def load_model(path=MODEL_PATH):
    global model, model_cols
    data = joblib.load(path)
    if isinstance(data, dict) and "model" in data:
        model = data["model"]
        model_cols = data.get("columns", None)
    else:
        model = data
        model_cols = None
    logger.info("Loaded model from %s (columns: %s)", path, model_cols)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"})

@app.route("/predict", methods=["POST"])
def predict_single():
    payload = request.get_json(force=True)
    # Accept either single dict or list-of-dicts (we handle single here)
    if isinstance(payload, list):
        df = pd.DataFrame(payload)
    else:
        df = pd.DataFrame([payload])
    if model_cols is None:
        return jsonify({"error":"server model missing column metadata"}), 500
    missing = [c for c in model_cols if c not in df.columns]
    if missing:
        return jsonify({"error":"missing columns", "missing": missing}), 400
    X = df[model_cols].astype(float)
    probs = model.predict_proba(X)[:,1]
    preds = (probs >= 0.5).astype(int)
    out = []
    for i, row in df.iterrows():
        out.append({"input_index": int(i), "probability": float(probs[i]), "label": int(preds[i])})
    return jsonify(out if len(out)>1 else out[0])

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    # Accept CSV file upload OR JSON array
    if "file" in request.files:
        f = request.files["file"]
        df = pd.read_csv(f)
    else:
        payload = request.get_json(force=True)
        df = pd.DataFrame(payload)
    if model_cols is None:
        return jsonify({"error":"server model missing column metadata"}), 500
    missing = [c for c in model_cols if c not in df.columns]
    if missing:
        return jsonify({"error":"missing columns", "missing": missing}), 400
    X = df[model_cols].astype(float)
    probs = model.predict_proba(X)[:,1]
    preds = (probs >= 0.5).astype(int)
    df_out = df.copy()
    df_out["success_prob"] = probs
    df_out["prediction"] = preds
    # Return JSON by default
    return df_out.to_json(orient="records")

@app.route("/optimize", methods=["POST"])
def optimize():
    """
    expected payload:
    {
      "base": {"lat":12.97,"lon":77.59},
      "deliveries": [
        {"id":1,"lat":12.98,"lon":77.60, "payload_kg":1.2, ...},
        {"id":2,"lat":12.95,"lon":77.62, ...}
      ],
      "min_success_prob": 0.6
    }
    """
    data = request.get_json(force=True)
    base = data.get("base")
    deliveries = data.get("deliveries", [])
    min_prob = data.get("min_success_prob", 0.0)

    if model_cols is None:
        return jsonify({"error":"server model missing column metadata"}), 500

    df = pd.DataFrame(deliveries)
    # Predict probs if necessary
    missing = [c for c in model_cols if c not in df.columns]
    if missing:
        return jsonify({"error":"deliveries missing model columns", "missing": missing}), 400
    X = df[model_cols].astype(float)
    probs = model.predict_proba(X)[:,1]
    df["prob"] = probs
    # filter
    filtered = df[df["prob"]>=min_prob].copy()
    if filtered.empty:
        return jsonify({"filtered_deliveries": [], "route_order": [], "route_distance_km": 0.0})

    # build list of points for TSP
    pts = []
    ids = []
    for _, r in filtered.iterrows():
        if ("lat" not in r) or ("lon" not in r):
            return jsonify({"error":"each delivery must have lat and lon for optimization"}), 400
        pts.append((float(r["lat"]), float(r["lon"])))
        ids.append(int(r["id"]) if "id" in r else None)

    result = solve_tsp(deliveries=filtered.to_dict(orient="records"), base=base, use_2opt=True)
    return jsonify(result)

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000)