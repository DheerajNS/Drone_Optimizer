import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.predict_with_threshold import run_prediction_on_df, run_prediction
from ml.tsp.tsp import solve_tsp


def run_full_pipeline(threshold=0.7, uploaded_df=None, selected_drones=None):

    # STEP 1: Get dataset
    if uploaded_df is not None:
        full_df, safe_df = run_prediction_on_df(uploaded_df, threshold)
    else:
        full_df, safe_df = run_prediction(threshold=threshold)

    if selected_drones:
        safe_df = safe_df[safe_df["drone_id"].isin(selected_drones)]

    drones_output = []
    total_distance = 0
    active_drones = 0

    # STEP 2: Run TSP
    for drone_id, group in safe_df.groupby("drone_id"):

        if len(group) == 0:
            continue

        active_drones += 1

        deliveries = []
        for idx, row in group.iterrows():
            deliveries.append({
                "id": int(idx),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "prob": float(row["success_probability"])
            })

        base = {
            "lat": float(group.iloc[0]["depot_lat"]),
            "lon": float(group.iloc[0]["depot_lon"])
        }

        result = solve_tsp(
            deliveries,
            base=base,
            min_success_prob=threshold,
            use_2opt=True
        )

        route_distance = result["route_distance_km"]
        total_distance += route_distance

        depot_coord = [base["lon"], base["lat"]]
        route_coords = result["coords"]

        # START + END at depot
        full_route = [depot_coord] + route_coords + [depot_coord]

        drones_output.append({
            "drone_id": int(drone_id),
            "safe_stops": len(result["filtered_deliveries"]),
            "route_distance_km": round(route_distance, 2),
            "route": full_route,
            "depot": depot_coord
        })

    summary = {
        "total_drones_generated": int(full_df["drone_id"].nunique()),
        "active_drones": active_drones,
        "total_distance_km": round(total_distance, 2),
        "threshold_used": threshold
    }

    return {
        "drones": drones_output,
        "summary": summary,
        "full_dataset": full_df
    }