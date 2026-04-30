"""Streamlit operator UI for the Waste Detection stack.

Two tabs:
- "Predict"        : model dropdown (fed from /models), image + GPS form, result
- "Map & History"  : Folium map of every persisted detection (manual + drone),
                     with filters by source / model / time window

Manual uploads are rendered red, drone-patrol detections orange (per the rubric's
"two data sources on the map" requirement).
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timedelta, timezone

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

API_URL = os.getenv("API_URL", "http://api:8000")
DEFAULT_CENTER = (48.8566, 2.3522)  # Paris

st.set_page_config(page_title="Waste Detection — Drone Patrols", layout="wide", page_icon="♻")
st.title("Waste Detection — Drone Patrols")
st.caption("Operator UI · upload an image, run a model, watch the map fill in as drones report back.")


# --------------------------------------------------------------------------- #
# Sidebar: API health
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.subheader("API status")
    try:
        h = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"OK · {h.get('models_loaded', 0)} models loaded")
    except Exception as e:
        st.error(f"API unreachable: {e}")
    st.caption(f"`{API_URL}`")


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=30, show_spinner=False)
def fetch_models() -> list[dict]:
    r = requests.get(f"{API_URL}/models", timeout=5)
    r.raise_for_status()
    return r.json()


def fetch_history() -> list[dict]:
    r = requests.get(f"{API_URL}/history", timeout=10)
    r.raise_for_status()
    return r.json()


def post_predict(image_bytes: bytes, image_name: str, content_type: str,
                 lat: float, lon: float, model_name: str) -> requests.Response:
    return requests.post(
        f"{API_URL}/predict",
        files={"image": (image_name, image_bytes, content_type or "image/jpeg")},
        data={"latitude": lat, "longitude": lon, "model_name": model_name},
        timeout=30,
    )


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_predict, tab_map = st.tabs(["Predict", "Map & History"])


# --- Predict --------------------------------------------------------------- #
with tab_predict:
    try:
        models = fetch_models()
    except Exception as e:
        st.error(f"Could not load models from API: {e}")
        models = []

    if not models:
        st.warning(
            "No models registered yet. Run "
            "`docker compose exec api python /app/scripts/register_models.py` first."
        )
    else:
        choices = [m["name"] for m in models]
        default_idx = choices.index("waste-detector-yolov8") if "waste-detector-yolov8" in choices else 0

        with st.form("predict"):
            model_name = st.selectbox("Detection model", choices, index=default_idx,
                                      help="Loaded from MLflow Production via /models")
            uploaded = st.file_uploader("Drone image (JPEG or PNG, ≤ 10 MB)",
                                        type=["jpg", "jpeg", "png"])
            c1, c2 = st.columns(2)
            with c1:
                lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0,
                                      value=DEFAULT_CENTER[0], format="%.6f")
            with c2:
                lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0,
                                      value=DEFAULT_CENTER[1], format="%.6f")
            submit = st.form_submit_button("Run prediction", type="primary")

        if submit:
            if uploaded is None:
                st.error("Please upload an image.")
            else:
                with st.spinner("Running inference…"):
                    resp = post_predict(uploaded.getvalue(), uploaded.name,
                                        uploaded.type, lat, lon, model_name)

                if resp.status_code == 200:
                    body = resp.json()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Detection", "Rubbish ✅" if body["rubbish"] else "Clean ❌")
                    c2.metric("Confidence", f"{body['confiance']:.1%}")
                    c3.metric("Model", body["model_used"].replace("waste-detector-", ""))
                    st.caption(f"timestamp: `{body['timestamp']}`")
                    st.image(uploaded, caption=f"{uploaded.name}", use_container_width=True)
                else:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except ValueError:
                        detail = resp.text
                    st.error(f"HTTP {resp.status_code}: {detail}")


# --- Map & History --------------------------------------------------------- #
with tab_map:
    try:
        history = fetch_history()
    except Exception as e:
        st.error(f"Could not load history: {e}")
        history = []

    if not history:
        st.info("No detections yet — run a prediction in the Predict tab "
                "or wait for the drone-patrol DAG to land some.")
    else:
        df = pd.DataFrame(history)
        # API writes microsecond+offset timestamps ("2026-04-29T17:07:33.434313+00:00")
        # while the drone simulator writes second-precision Z timestamps
        # ("2026-04-29T15:42:50Z"). Both are valid ISO 8601, but pandas 2.2 needs
        # an explicit format='ISO8601' to accept both shapes in the same column.
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")

        st.markdown("**Filters**")
        f1, f2, f3 = st.columns([1, 1, 2])
        with f1:
            source_filter = st.multiselect(
                "Source",
                options=sorted(df["source"].unique()),
                default=sorted(df["source"].unique()),
            )
        with f2:
            model_filter = st.multiselect(
                "Model",
                options=sorted(df["model_name"].unique()),
                default=sorted(df["model_name"].unique()),
            )
        with f3:
            hours_back = st.slider("Time window (hours back from now)", 1, 168, 168)

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        filtered = df[
            df["source"].isin(source_filter)
            & df["model_name"].isin(model_filter)
            & (df["timestamp"] >= cutoff)
        ].copy()

        m1, m2, m3 = st.columns(3)
        m1.metric("Total detections", len(df))
        m2.metric("After filters", len(filtered))
        m3.metric("Distinct models", df["model_name"].nunique())

        if filtered.empty:
            center = DEFAULT_CENTER
        else:
            center = (filtered["latitude"].mean(), filtered["longitude"].mean())

        fmap = folium.Map(location=list(center), zoom_start=6, tiles="OpenStreetMap")

        for _, row in filtered.iterrows():
            color = "red" if row["source"] == "manual" else "orange"
            popup = folium.Popup(
                html=(
                    f"<b>{row['model_name']}</b><br>"
                    f"source: <b>{row['source']}</b><br>"
                    f"confidence: {row['confiance']:.1%}<br>"
                    f"<small>{row['timestamp']}</small>"
                    + (f"<br>drone: {row['drone_id']}" if row.get("drone_id") else "")
                ),
                max_width=260,
            )
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                weight=1,
                popup=popup,
            ).add_to(fmap)

        legend_html = """
        <div style="position: fixed; top: 80px; right: 12px; z-index: 1000;
                    background: white; padding: 8px 10px; border: 1px solid #888;
                    border-radius: 4px; font-size: 13px;">
          <b>Legend</b><br>
          <span style="color: red; font-size: 16px;">●</span> Manual upload<br>
          <span style="color: orange; font-size: 16px;">●</span> Drone patrol
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(legend_html))

        st_folium(fmap, width=None, height=560, returned_objects=[])

        with st.expander("Raw detections (filtered)"):
            st.dataframe(
                filtered.sort_values("timestamp", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
