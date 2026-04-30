"""
generate_patrol_db.py — Drone patrol mission simulator.

Each call inserts a random batch of detections (20–100) into drone_patrol.db,
all marked processed=0. Run it multiple times to simulate successive missions.
"""

import sqlite3
import random
from datetime import datetime, timedelta

DB_PATH = "drone_patrol.db"

DRONES = [
    "drone_001", "drone_002", "drone_003",
    "drone_004", "drone_005",
]

CITIES = [
    # --- Île-de-France ---
    {
        "ville": "Paris",
        "zones": ["75011", "75012", "75018", "75019", "75020"],
        "lat_range": (48.815, 48.902),
        "lon_range": (2.224, 2.470),
    },
    {
        "ville": "Saint-Denis",
        "zones": ["93200", "93210"],
        "lat_range": (48.920, 48.950),
        "lon_range": (2.330, 2.390),
    },
    {
        "ville": "Montreuil",
        "zones": ["93100"],
        "lat_range": (48.850, 48.880),
        "lon_range": (2.430, 2.470),
    },
    {
        "ville": "Nanterre",
        "zones": ["92000"],
        "lat_range": (48.880, 48.910),
        "lon_range": (2.180, 2.230),
    },
    {
        "ville": "Créteil",
        "zones": ["94000"],
        "lat_range": (48.770, 48.800),
        "lon_range": (2.440, 2.480),
    },
    # --- Grand Lyon ---
    {
        "ville": "Lyon",
        "zones": ["69001", "69003", "69006", "69007", "69008"],
        "lat_range": (45.715, 45.800),
        "lon_range": (4.780, 4.900),
    },
    {
        "ville": "Villeurbanne",
        "zones": ["69100"],
        "lat_range": (45.760, 45.790),
        "lon_range": (4.870, 4.920),
    },
    {
        "ville": "Vénissieux",
        "zones": ["69200"],
        "lat_range": (45.690, 45.730),
        "lon_range": (4.870, 4.920),
    },
    {
        "ville": "Bron",
        "zones": ["69500"],
        "lat_range": (45.730, 45.760),
        "lon_range": (4.900, 4.940),
    },
    # --- Aix-Marseille ---
    {
        "ville": "Marseille",
        "zones": ["13001", "13003", "13008", "13013", "13014"],
        "lat_range": (43.220, 43.360),
        "lon_range": (5.300, 5.460),
    },
    {
        "ville": "Aix-en-Provence",
        "zones": ["13080", "13090", "13100"],
        "lat_range": (43.510, 43.560),
        "lon_range": (5.400, 5.490),
    },
    {
        "ville": "Aubagne",
        "zones": ["13400"],
        "lat_range": (43.280, 43.310),
        "lon_range": (5.550, 5.610),
    },
    {
        "ville": "Vitrolles",
        "zones": ["13127"],
        "lat_range": (43.440, 43.470),
        "lon_range": (5.230, 5.290),
    },
    # --- Métropole de Bordeaux ---
    {
        "ville": "Bordeaux",
        "zones": ["33000", "33100", "33200", "33300"],
        "lat_range": (44.820, 44.890),
        "lon_range": (-0.620, -0.530),
    },
    {
        "ville": "Mérignac",
        "zones": ["33700"],
        "lat_range": (44.820, 44.860),
        "lon_range": (-0.680, -0.620),
    },
    {
        "ville": "Pessac",
        "zones": ["33600"],
        "lat_range": (44.790, 44.830),
        "lon_range": (-0.650, -0.590),
    },
    # --- Métropole Européenne de Lille ---
    {
        "ville": "Lille",
        "zones": ["59000", "59100", "59200", "59260"],
        "lat_range": (50.610, 50.660),
        "lon_range": (3.030, 3.110),
    },
    {
        "ville": "Roubaix",
        "zones": ["59100"],
        "lat_range": (50.680, 50.710),
        "lon_range": (3.160, 3.200),
    },
    {
        "ville": "Tourcoing",
        "zones": ["59200"],
        "lat_range": (50.710, 50.740),
        "lon_range": (3.140, 3.180),
    },
    # --- Toulouse ---
    {
        "ville": "Toulouse",
        "zones": ["31000", "31100", "31200", "31300", "31400"],
        "lat_range": (43.560, 43.650),
        "lon_range": (1.380, 1.500),
    },
    {
        "ville": "Blagnac",
        "zones": ["31700"],
        "lat_range": (43.630, 43.660),
        "lon_range": (1.370, 1.410),
    },
    {
        "ville": "Colomiers",
        "zones": ["31770"],
        "lat_range": (43.600, 43.640),
        "lon_range": (1.320, 1.370),
    },
    # --- Nice Côte d'Azur ---
    {
        "ville": "Nice",
        "zones": ["06000", "06100", "06200", "06300"],
        "lat_range": (43.680, 43.740),
        "lon_range": (7.200, 7.310),
    },
    {
        "ville": "Antibes",
        "zones": ["06600"],
        "lat_range": (43.560, 43.610),
        "lon_range": (7.080, 7.150),
    },
    {
        "ville": "Cannes",
        "zones": ["06400"],
        "lat_range": (43.530, 43.570),
        "lon_range": (6.990, 7.060),
    },
    # --- Nantes ---
    {
        "ville": "Nantes",
        "zones": ["44000", "44100", "44200", "44300"],
        "lat_range": (47.200, 47.280),
        "lon_range": (-1.600, -1.490),
    },
    {
        "ville": "Saint-Herblain",
        "zones": ["44800"],
        "lat_range": (47.200, 47.240),
        "lon_range": (-1.660, -1.600),
    },
    # --- Strasbourg ---
    {
        "ville": "Strasbourg",
        "zones": ["67000", "67100", "67200"],
        "lat_range": (48.540, 48.620),
        "lon_range": (7.710, 7.810),
    },
    {
        "ville": "Schiltigheim",
        "zones": ["67300"],
        "lat_range": (48.600, 48.630),
        "lon_range": (7.730, 7.780),
    },
]


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS drone_detections (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            drone_id       TEXT    NOT NULL,
            timestamp      TEXT    NOT NULL,
            latitude       REAL    NOT NULL,
            longitude      REAL    NOT NULL,
            ville          TEXT    NOT NULL,
            zone           TEXT    NOT NULL,
            classe         TEXT    NOT NULL DEFAULT 'rubbish',
            confiance      REAL    NOT NULL,
            image_filename TEXT    NOT NULL,
            processed      INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()


def generate_detections(n: int) -> list:
    detections = []
    now = datetime.utcnow()
    # Each mission spans up to 4 hours in the past
    mission_start = now - timedelta(hours=random.uniform(1, 4))

    for _ in range(n):
        city = random.choice(CITIES)
        zone = random.choice(city["zones"])
        lat = round(random.uniform(*city["lat_range"]), 6)
        lon = round(random.uniform(*city["lon_range"]), 6)
        confiance = round(random.betavariate(5, 2), 4)
        ts = (
            mission_start + timedelta(seconds=random.randint(0, int((now - mission_start).total_seconds())))
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        filename = f"patrol_{random.randint(100000, 999999)}.jpg"
        detections.append((
            random.choice(DRONES),
            ts,
            lat,
            lon,
            city["ville"],
            zone,
            "rubbish",
            confiance,
            filename,
            0,
        ))
    return detections


def main() -> None:
    n = random.randint(20, 100)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    detections = generate_detections(n)
    conn.executemany(
        """
        INSERT INTO drone_detections
            (drone_id, timestamp, latitude, longitude, ville, zone,
             classe, confiance, image_filename, processed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        detections,
    )
    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM drone_detections").fetchone()[0]
    confidences = [d[7] for d in detections]
    avg_conf = sum(confidences) / len(confidences)
    below = sum(1 for c in confidences if c < 0.65)
    above = n - below

    conn.close()

    print(f"✓ Mission simulated — drone_patrol.db updated")
    print(f"  {n} new detections inserted")
    print(f"  Average confidence       : {avg_conf:.2f}")
    print(f"  Below threshold 0.65     : ~{below}")
    print(f"  Will pass filter         : ~{above}")
    print(f"  Total cumulative in DB   : {total}")


if __name__ == "__main__":
    main()
