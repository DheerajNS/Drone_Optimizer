# generate_dataset.py
# Create a synthetic dataset with same column names as your sample CSV.
import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path

def make_dataset(n=5000, seed=42, out="drone_deliveries.csv"):
    rng = np.random.RandomState(seed)
    payload_kg = np.round(rng.uniform(0.1, 5.0, n), 2)
    distance_km = np.round(rng.uniform(0.1, 20.0, n), 2)
    battery_pct = np.round(rng.uniform(10, 100, n), 1)
    wind_speed_mps = np.round(rng.uniform(0, 15, n), 2)
    rain_mm = np.round(rng.uniform(0.0, 5.0, n), 2)
    time_of_day = rng.choice(["morning","afternoon","evening","night"], n, p=[0.3,0.25,0.25,0.2])
    num_stops = rng.choice([1,2,3], n, p=[0.5,0.35,0.15])

    # create a synthetic success label probability using a simple rule + noise
    score = (1.0 - (payload_kg / 6.0)) * 0.2 + (battery_pct/100.0)*0.5 - (distance_km/20.0)*0.2 \
            - (wind_speed_mps/15.0)*0.1 - (rain_mm/5.0)*0.05
    score += rng.normal(0, 0.03, n)
    prob = 1/(1+np.exp(- ( (score-0.3)*5 )))  # squash to 0-1
    success = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "payload_kg": payload_kg,
        "distance_km": distance_km,
        "battery_pct": battery_pct,
        "wind_speed_mps": wind_speed_mps,
        "rain_mm": rain_mm,
        "time_of_day": time_of_day,
        "num_stops": num_stops,
        "success": success
    })
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--out", default="drone_deliveries.csv")
    args = p.parse_args()
    make_dataset(n=args.n, out=args.out)