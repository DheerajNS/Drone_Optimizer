import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import random

# -----------------------------
# CONFIGURATION
# -----------------------------
NUM_DRONES = 80
MAX_STOPS_PER_DRONE = 25
INITIAL_BATTERY = 100

BASE_LAT = 12.9716   # Bengaluru center
BASE_LON = 77.5946

ALPHA = 1.6   # distance weight
BETA = 2.0    # payload weight
GAMMA = 1.3   # wind weight

# -----------------------------
# HAVERSINE
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

rows = []

for drone_id in range(1, NUM_DRONES + 1):

    # Each drone has its own depot
    depot_lat = BASE_LAT + np.random.uniform(-0.03, 0.03)
    depot_lon = BASE_LON + np.random.uniform(-0.03, 0.03)

    battery = INITIAL_BATTERY
    prev_lat, prev_lon = depot_lat, depot_lon

    for stop_index in range(MAX_STOPS_PER_DRONE):

        lat = BASE_LAT + np.random.uniform(-0.07, 0.07)
        lon = BASE_LON + np.random.uniform(-0.07, 0.07)

        leg_distance = haversine(prev_lat, prev_lon, lat, lon)

        payload = np.random.uniform(0.5, 3.5)
        wind = np.random.uniform(0, 15)
        rain = np.random.uniform(0, 8)

        battery_before = battery

        # Battery drain physics-inspired
        battery_drop = (
            ALPHA * leg_distance +
            BETA * payload +
            GAMMA * wind
        )

        battery -= battery_drop

        if battery <= 0:
            battery_after = 0
        else:
            battery_after = battery

        # Risk model (linear-friendly)
        risk_score = (
            0.5 * leg_distance +
            0.3 * payload +
            0.4 * wind +
            0.8 * (1 / (battery_after + 1))
        )

        threshold = random.uniform(7.5, 10.5)
        success = 1 if risk_score < threshold else 0

        rows.append([
            drone_id,
            depot_lat,
            depot_lon,
            lat,
            lon,
            leg_distance,
            payload,
            wind,
            rain,
            battery_before,
            battery_after,
            success
        ])

        if battery_after == 0:
            break

        prev_lat, prev_lon = lat, lon

df = pd.DataFrame(rows, columns=[
    "drone_id",
    "depot_lat",
    "depot_lon",
    "lat",
    "lon",
    "leg_distance_km",
    "payload_kg",
    "wind_speed_mps",
    "rain_mm",
    "battery_before",
    "battery_after",
    "success"
])

df.to_csv("research_multi_drone_dataset.csv", index=False)

print("Dataset generated.")
print("Total rows:", len(df))
print("Success rate:", round(df["success"].mean(), 3))
print("Min battery:", round(df["battery_after"].min(), 2))
print("Max battery:", round(df["battery_after"].max(), 2))