# front_end/app.py (Streamlit)
import streamlit as st
import pandas as pd
import json
import os
import requests
import pydeck as pdk
import sys

ROOT = os.path.dirname(__file__)
st.set_page_config(page_title="SmartDrone Demo", layout="wide")

st.title("SmartDrone — Delivery safety & route demo")

st.sidebar.header("Configuration")
use_backend = st.sidebar.checkbox("Use backend API (if checked, set URL)", value=False)
backend_url = st.sidebar.text_input("Backend base URL (e.g. http://127.0.0.1:5000)", value="http://127.0.0.1:5000")
min_prob = st.sidebar.slider("min_success_prob for route filter", 0.0, 1.0, 0.6, 0.05)

st.markdown(
    "Upload CSV of deliveries or use sample. CSV must include: "
    "id,lat,lon,payload_kg,distance_km,battery_pct,wind_speed_mps,rain_mm,num_stops"
)

# -------------------------
# Helper: route call wrapper
# -------------------------
def get_route_result(deliveries, base, min_prob=0.5, use_backend=False, backend_url="http://127.0.0.1:5000"):
    """
    deliveries: list of dicts (rows)
    base: {"lat":..., "lon":...}
    returns: dict with keys coords (list of [lon,lat]), route_order, filtered_deliveries, route_distance_km
    """
    if use_backend:
        payload = {"base": base, "deliveries": deliveries, "min_success_prob": min_prob}
        resp = requests.post(backend_url.rstrip("/") + "/optimize", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    else:
        # call local tsp module
        sys.path.append(os.path.abspath(os.path.join(ROOT, "..", "ml", "tsp")))
        import tsp
        return tsp.solve_tsp(deliveries=deliveries, base=base, min_success_prob=min_prob, use_2opt=True)

# -------------------------
# Session-state helpers
# -------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "preds" not in st.session_state:
    st.session_state.preds = None
if "opt_result" not in st.session_state:
    st.session_state.opt_result = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None

def load_sample_csv():
    sample_path = os.path.join(ROOT, "..", "ml", "drone_deliveries.csv")
    if not os.path.exists(sample_path):
        st.error("Sample CSV not found at " + sample_path)
        return
    df_local = pd.read_csv(sample_path)
    # add fake coords if missing
    if "lat" not in df_local.columns or "lon" not in df_local.columns:
        import numpy as np
        df_local["lat"] = 12.97 + np.random.normal(0, 0.01, len(df_local))
        df_local["lon"] = 77.59 + np.random.normal(0, 0.01, len(df_local))
        df_local["id"] = range(1, len(df_local) + 1)
    st.session_state.df = df_local
    st.session_state.preds = None
    st.session_state.opt_result = None
    st.session_state.last_error = None

def set_uploaded_df(upl):
    try:
        df_u = pd.read_csv(upl)
    except Exception as e:
        st.error("Failed to read uploaded CSV: " + str(e))
        return
    st.session_state.df = df_u
    st.session_state.preds = None
    st.session_state.opt_result = None
    st.session_state.last_error = None

def run_local_predictions():
    if st.session_state.df is None:
        st.error("No data loaded. Load sample CSV or upload your CSV first.")
        return
    model_path = os.path.join(ROOT, "..", "ml", "drone_success_model.joblib")
    try:
        import joblib
        data = joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model from {model_path}: {e}")
        return
    model = data.get("model")
    cols = data.get("columns", None)
    if cols is None or model is None:
        st.error("Model metadata incomplete. Re-train model or ensure joblib contains {'model','columns'}.")
        return
    missing = [c for c in cols if c not in st.session_state.df.columns]
    if missing:
        st.error(f"Missing columns for model: {missing}")
        return
    X = st.session_state.df[cols].astype(float)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    df2 = st.session_state.df.copy()
    df2["success_prob"] = probs
    df2["prediction"] = preds
    st.session_state.df = df2
    st.session_state.preds = df2[["id", "success_prob", "prediction"]] if "id" in df2.columns else df2[["success_prob", "prediction"]]
    st.session_state.opt_result = None
    st.success("Local predictions added")

def run_backend_predictions():
    if st.session_state.df is None:
        st.error("No data loaded. Load sample CSV or upload your CSV first.")
        return
    try:
        resp = requests.post(backend_url.rstrip("/") + "/predict_batch", json=st.session_state.df.to_dict(orient="records"), timeout=15)
        if resp.status_code != 200:
            st.error(f"Backend error: {resp.status_code} {resp.text}")
            return
        preds = pd.read_json(resp.text)
        df2 = pd.concat([st.session_state.df.reset_index(drop=True), preds[["success_prob", "prediction"]]], axis=1)
        st.session_state.df = df2
        st.session_state.preds = df2[["id", "success_prob", "prediction"]] if "id" in df2.columns else df2[["success_prob", "prediction"]]
        st.session_state.opt_result = None
        st.success("Got predictions from backend")
    except Exception as e:
        st.error(f"Backend predict_batch failed: {e}")

def run_optimize():
    if st.session_state.df is None:
        st.error("No data loaded to optimize. Load sample or upload CSV.")
        return
    if "success_prob" not in st.session_state.df.columns:
        st.error("Predictions missing — run local or backend predictions first.")
        return
    payload_deliveries = st.session_state.df.to_dict(orient="records")
    base = {"lat": 12.97, "lon": 77.59}
    try:
        res = get_route_result(deliveries=payload_deliveries, base=base, min_prob=float(min_prob),
                               use_backend=use_backend, backend_url=backend_url)
        st.session_state.opt_result = res
        st.session_state.last_error = None
    except Exception as e:
        st.error(f"Optimization failed: {e}")
        st.session_state.opt_result = {}
        st.session_state.last_error = str(e)

# -------------------------
# UI: file uploader + controls
# -------------------------
uploaded = st.file_uploader("Upload deliveries CSV", type=["csv"], on_change=lambda: set_uploaded_df(uploaded) )
col1, col2 = st.columns(2)

with col1:
    if st.button("Load sample CSV"):
        load_sample_csv()

with col2:
    # Buttons for predictions
    if use_backend:
        if st.button("Send to backend and get predictions"):
            run_backend_predictions()
    else:
        if st.button("Run local predictions (load local model)"):
            run_local_predictions()

# show df preview if available in session_state
if st.session_state.df is not None:
    st.subheader("Preview deliveries")
    st.dataframe(st.session_state.df.head(20))
else:
    st.info("Upload or load sample CSV to begin.")

# Optimize button area
if st.session_state.df is not None and "success_prob" in st.session_state.df.columns:
    if st.button("Optimize route (local)"):
        run_optimize()

# -------------------------
# Show optimization results (if any)
# -------------------------
if st.session_state.opt_result:
    res = st.session_state.opt_result
    coords = res.get("coords", [])
    route_order = res.get("route_order", [])
    filtered = res.get("filtered_deliveries", [])
    route_distance = res.get("route_distance_km", None)

    if coords:
        # Ensure coords are [lon, lat]
        try:
            stops = pd.DataFrame(coords, columns=["lon", "lat"])
        except Exception:
            # If coords are [lat,lon], convert
            stops = pd.DataFrame(coords, columns=["lat", "lon"])[["lon", "lat"]]
        stops["order"] = list(range(1, len(stops) + 1))
        stops["label"] = stops["order"].astype(str)

        # path as list of points
        path = [coords]
        paths_df = pd.DataFrame({"path": path})

        path_layer = pdk.Layer(
            "PathLayer",
            data=paths_df,
            get_path="path",
            get_width=4,
            get_color=[0, 255, 0],   # green route line
            pickable=False
)
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=stops,
            get_position=["lon", "lat"],
            get_radius=60,
            get_fill_color=[255, 0, 0],   # red color
            pickable=True
)


        # Build deck
        initial_view = pdk.ViewState(latitude=stops["lat"].mean(), longitude=stops["lon"].mean(), zoom=13, pitch=0)
        deck = pdk.Deck(layers=[path_layer, scatter_layer], initial_view_state=initial_view, tooltip={"text": "Stop: {order}"})
        st.pydeck_chart(deck)

    # show filtered deliveries and route info
    st.subheader("Optimization result")
    if filtered:
        try:
            df_f = pd.DataFrame(filtered)
            st.dataframe(df_f)
        except Exception:
            st.write(filtered)
    else:
        st.write("No deliveries passed the min_success_prob filter.")
    st.write("Route order (IDs):", route_order)
    st.write("Route distance (km):", route_distance)

# show any last error message stored
if st.session_state.last_error:
    st.error("Last error: " + str(st.session_state.last_error))
