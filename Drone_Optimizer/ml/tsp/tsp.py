# ml/tsp/tsp.py
import math, itertools, json, logging
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tsp")

EARTH_R = 6371.0  # km

def haversine_km(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * EARTH_R * math.asin(math.sqrt(h))

def total_route_distance_km(points: List[Tuple[float,float]], base: Optional[Tuple[float,float]]=None):
    if base is not None:
        pts = [base] + points + [base]
    else:
        pts = points
    total = 0.0
    for i in range(len(pts)-1):
        total += haversine_km((pts[i][0], pts[i][1]), (pts[i+1][0], pts[i+1][1]))
    return total

def nearest_neighbor_order(points: List[Tuple[float,float]], base: Tuple[float,float]):
    if not points:
        return []
    remaining = points.copy()
    order = []
    cur = base
    while remaining:
        # find nearest
        distances = [haversine_km(cur, p) for p in remaining]
        idx = distances.index(min(distances))
        order.append(remaining.pop(idx))
        cur = order[-1]
    return order

def two_opt(points: List[Tuple[float,float]], base: Tuple[float,float], max_iter=100):
    # points is ordered list of customer points (lat,lon)
    best = points.copy()
    best_dist = total_route_distance_km(best, base=base)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(0, len(best)-1):
            for j in range(i+1, len(best)):
                if j-i == 1: 
                    continue
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                new_dist = total_route_distance_km(new_route, base=base)
                if new_dist + 1e-6 < best_dist:
                    best = new_route
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break
    return best, best_dist

def solve_tsp(deliveries: List[Dict], base: Dict = None, min_success_prob: float = 0.0, use_2opt: bool = True):
    """
    deliveries: list of dicts with keys at least 'id','lat','lon','prob' OR full fields
    base: {"lat":..., "lon":...} or None
    Returns:
      {
        "filtered_deliveries": [{"id":..., "prob":...}, ...],
        "route_order": [id,...],
        "route_distance_km": float,
        "coords": [[lon,lat], ...]  # optional for plotting
      }
    """
    if base is None:
        raise ValueError("base must be provided as {'lat':..., 'lon':...}")
    base_pt = (float(base["lat"]), float(base["lon"]))
    # filter by min success prob
    filtered = [d for d in deliveries if float(d.get("prob", 1.0)) >= min_success_prob]
    if not filtered:
        return {"filtered_deliveries": [], "route_order": [], "route_distance_km": 0.0, "coords": []}

    points = []
    ids = []
    for d in filtered:
        lat = float(d["lat"])
        lon = float(d["lon"])
        points.append((lat, lon))
        ids.append(d.get("id"))

    ordered_points = nearest_neighbor_order(points.copy(), base_pt)
    if use_2opt and len(ordered_points) >= 3:
        ordered_points, best_dist = two_opt(ordered_points, base=base_pt)
    else:
        best_dist = total_route_distance_km(ordered_points, base=base_pt)

    # map ordered_points back to ids (match by lat/lon)
    order_ids = []
    for pt in ordered_points:
        for d in filtered:
            if float(d["lat"]) == float(pt[0]) and float(d["lon"]) == float(pt[1]):
                order_ids.append(d.get("id"))
                break

    coords = [[pt[1], pt[0]] for pt in ordered_points]  # lon,lat list for plotting

    return {
        "filtered_deliveries": [{"id": d.get("id"), "prob": float(d.get("prob", 1.0))} for d in filtered],
        "route_order": order_ids,
        "route_distance_km": round(best_dist, 3),
        "coords": coords
    }

# CLI test
if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="JSON file with deliveries (list)")
    p.add_argument("--base", required=True, help="base lat,lon e.g. 12.97,77.59")
    p.add_argument("--min_prob", type=float, default=0.0)
    args = p.parse_args()
    with open(args.input) as f:
        deliveries = json.load(f)
    blat, blon = map(float, args.base.split(","))
    res = solve_tsp(deliveries, base={"lat":blat,"lon":blon}, min_success_prob=args.min_prob)
    print(json.dumps(res, indent=2))