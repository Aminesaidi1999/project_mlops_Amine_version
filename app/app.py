"""
Foundation stub — placeholder UI so the Streamlit container boots.
Teammate B replaces this with: model dropdown, image upload + GPS form,
prediction display, Folium map of historical detections (manual + drone),
and filters by source/model/time.
"""

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(page_title="Waste Detection — Drone Patrols", layout="wide")
st.title("Waste Detection — Drone Patrols")
st.caption("Foundation stub. Real UI lands when Half B (Streamlit + map) is wired up.")

with st.sidebar:
    st.header("Stack health")
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        st.success(f"API: {r.json()}") if r.ok else st.error(f"API HTTP {r.status_code}")
    except Exception as e:
        st.error(f"API unreachable: {e}")
