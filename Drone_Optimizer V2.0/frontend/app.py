import sys
import os
import streamlit as st
import pydeck as pdk
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.api_pipeline import run_full_pipeline

st.set_page_config(layout="wide")
st.title("🚁 Drone Optimizer V2.0")

# ================= SESSION STATE =================
if "result" not in st.session_state:
    st.session_state.result = None

if "params" not in st.session_state:
    st.session_state.params = {}

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Controls")

threshold = st.sidebar.slider(
    "Minimum Success Probability",
    0.3, 0.9, 0.7, 0.05
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Scenario CSV",
    type=["csv"]
)

selected_drones = st.sidebar.multiselect(
    "Select Drones",
    options=[1, 2, 3, 4, 5],
    default=[1, 2, 3, 4, 5]
)

run_button = st.sidebar.button("Run Optimization")

current_params = {
    "threshold": threshold,
    "selected_drones": tuple(selected_drones),
    "file": uploaded_file.name if uploaded_file else None
}

# ================= RUN =================
if run_button or current_params != st.session_state.params:

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.result = run_full_pipeline(
            threshold=threshold,
            uploaded_df=df,
            selected_drones=selected_drones
        )
    else:
        st.session_state.result = run_full_pipeline(
            threshold=threshold,
            selected_drones=selected_drones
        )

    st.session_state.params = current_params

# ================= DISPLAY =================
if st.session_state.result:

    result = st.session_state.result
    summary = result["summary"]
    drones = result["drones"]
    full_df = result["full_dataset"]

    st.subheader("📊 Overall Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Drones", summary["total_drones_generated"])
    c2.metric("Active Drones", summary["active_drones"])
    c3.metric("Total Distance (km)", summary["total_distance_km"])

    st.markdown("---")

    st.subheader("🚁 Per-Drone Summary")

    for d in drones:
        c1, c2, c3 = st.columns(3)
        c1.metric("Drone ID", d["drone_id"])
        c2.metric("Safe Stops", d["safe_stops"])
        c3.metric("Route Distance (km)", d["route_distance_km"])
        st.markdown("---")

    # ================= MAP =================
    st.subheader("🗺 Route Visualization")

    progress = st.slider("Route Progress", 0.0, 1.0, 1.0, 0.05)

    layers = []
    all_coords = []

    # Dark bold colors for neutral background
    drone_colors = {
        1: [120, 0, 0],       # Deep Red
        2: [0, 0, 120],       # Deep Blue
        3: [0, 100, 0],       # Deep Green
        4: [100, 0, 100],     # Deep Purple
        5: [140, 70, 0],      # Deep Orange
    }

    for d in drones:

        full_path = d["route"]
        depot = d["depot"]
        drone_id = d["drone_id"]

        if not full_path:
            continue

        cut_index = max(1, int(len(full_path) * progress))
        animated_path = full_path[:cut_index]

        all_coords.extend(full_path)

        color = drone_colors.get(drone_id, [0, 0, 0])

        # ROUTE
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=[{"path": animated_path}],
                get_path="path",
                get_width=35,
                width_scale=4,
                get_color=color,
                pickable=True,
            )
        )

        # STOPS
        stop_data = []
        for i, coord in enumerate(full_path[1:-1], start=1):
            stop_data.append({
                "position": coord,
                "label": f"Drone {drone_id} - Stop {i}"
            })

        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=stop_data,
                get_position="position",
                get_radius=400,
                get_fill_color=color,
                get_line_color=[0, 0, 0],
                line_width_min_pixels=2,
                pickable=True,
            )
        )

        # DEPOT
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=[{
                    "position": depot,
                    "label": f"Drone {drone_id} Depot"
                }],
                get_position="position",
                get_radius=750,
                get_fill_color=[255, 200, 0],
                get_line_color=[0, 0, 0],
                line_width_min_pixels=3,
                pickable=True,
            )
        )

        # MOVING DRONE ICON
        layers.append(
            pdk.Layer(
                "IconLayer",
                data=[{
                    "position": animated_path[-1],
                    "icon_data": {
                        "url": "https://imgs.search.brave.com/nVB14VbNhtX3WdWLmd9lLEcWr4qTcJpB_AiUUerfQj4/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9zdGF0/aWMudmVjdGVlenku/Y29tL3N5c3RlbS9y/ZXNvdXJjZXMvdGh1/bWJuYWlscy8wNTgv/MDQ1LzE0Ny9zbWFs/bC9kcm9uZS13aXRo/LWNhbWVyYS1ob3Zl/cmluZy1pbi1jbGVh/ci1za3ktcG5nLnBu/Zw",
                        "width": 128,
                        "height": 128,
                        "anchorY": 128
                    },
                    "label": f"Drone {drone_id} (In Transit)"
                }],
                get_icon="icon_data",
                get_size=6,
                size_scale=10,
                get_position="position",
                pickable=True,
            )
        )

    if all_coords:

        avg_lat = sum([c[1] for c in all_coords]) / len(all_coords)
        avg_lon = sum([c[0] for c in all_coords]) / len(all_coords)

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(
                latitude=avg_lat,
                longitude=avg_lon,
                zoom=11,
                pitch=50
            ),
            map_style="light",   # Balanced neutral map
            tooltip={"text": "{label}"}
        )

        st.pydeck_chart(deck)

    with st.expander("🔍 View Full Dataset"):
        st.dataframe(full_df)