import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt

# -----------------------------
# Haversine Distance Function
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))

    return R * c


# -----------------------------
# Weather Distribution
# -----------------------------
def generate_wind():
    r = np.random.rand()
    if r < 0.7:
        return np.random.uniform(1, 6)
    elif r < 0.9:
        return np.random.uniform(6, 10)
    else:
        return np.random.uniform(10, 12)


def generate_rain():
    r = np.random.rand()
    if r < 0.6:
        return 0
    elif r < 0.9:
        return np.random.uniform(1, 5)
    else:
        return np.random.uniform(10, 25)


def generate_payload():
    r = np.random.rand()
    if r < 0.6:
        return np.random.uniform(1, 3)
    elif r < 0.9:
        return np.random.uniform(3, 5)
    else:
        return np.random.uniform(0.3, 1)


# -----------------------------
# Main Dataset Generator
# -----------------------------
def generate_dataset(num_drones=5, stops_per_drone=5):

    np.random.seed(42)

    # Define 5 clearly separated depots (Bengaluru scale)
    depots = [
        (12.9716, 77.5946),  # Central
        (13.0416, 77.6846),  # East
        (13.0716, 77.5246),  # North-West
        (12.9016, 77.6646),  # South-East
        (12.8916, 77.5246)   # South-West
    ]

    data = []

    for drone_id in range(num_drones):

        depot_lat, depot_lon = depots[drone_id]

        # Delivery zone radius variation
        zone = np.random.uniform(0.03, 0.06)

        for _ in range(stops_per_drone):

            lat = depot_lat + np.random.uniform(-zone, zone)
            lon = depot_lon + np.random.uniform(-zone, zone)

            distance_km = haversine(depot_lat, depot_lon, lat, lon)

            payload = generate_payload()
            wind = generate_wind()
            rain = generate_rain()
            battery_before = np.random.uniform(85, 100)

            data.append([
                drone_id + 1,
                depot_lat,
                depot_lon,
                lat,
                lon,
                round(distance_km, 2),
                round(payload, 2),
                round(wind, 2),
                round(rain, 2),
                round(battery_before, 2)
            ])

    columns = [
        "drone_id",
        "depot_lat",
        "depot_lon",
        "lat",
        "lon",
        "distance_km",
        "payload_kg",
        "wind_speed_mps",
        "rain_mm",
        "battery_before"
    ]

    df = pd.DataFrame(data, columns=columns)

    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.head())

    print("\nDistance Min:", df["distance_km"].min())
    print("Distance Max:", df["distance_km"].max())