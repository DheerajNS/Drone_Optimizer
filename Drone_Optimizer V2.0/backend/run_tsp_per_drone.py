import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.predict_with_threshold import run_prediction
from ml.tsp.tsp import solve_tsp


threshold = 0.7

print("Running multi-drone ML + TSP pipeline...\n")

# Step 1: Run dynamic prediction
full_df, safe_df = run_prediction(threshold=threshold)

overall_distance = 0
active_drones = 0

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
    overall_distance += route_distance

    print(f"🚁 Drone {drone_id}")
    print("  Safe Stops:", len(result["filtered_deliveries"]))
    print("  Route Distance (km):", round(route_distance, 2))
    print("-" * 40)


print("\n============================")
print("ACTIVE DRONES:", active_drones)
print("TOTAL ROUTE DISTANCE (km):", round(overall_distance, 2))
print("============================")