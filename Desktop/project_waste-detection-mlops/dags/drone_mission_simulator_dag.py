"""DAG 1 — drone_mission_simulator.

Simulates a drone returning from a mission: invokes the provided
`generate_patrol_db.py`, which inserts 20–100 random detections (all
processed=0) into `/data/drone_patrol.db`, then triggers DAG 2
(`drone_patrol_sync`) immediately via TriggerDagRunOperator.

Schedule is `*/5 * * * *` (test/demo mode). The rubric requires the repo to
be submitted with this schedule active so the grader sees execution history.

Layout:
    simulate_mission  -->  trigger_sync
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

with DAG(
    dag_id="drone_mission_simulator",
    description="Simulate a drone mission and trigger the patrol-sync ETL.",
    start_date=datetime(2026, 4, 1),
    schedule_interval="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["mission", "drone-patrol"],
    default_args={"retries": 0},
) as dag:

    # `cd /data` so the DB lands in the shared bind mount (the script hard-codes
    # `DB_PATH = "drone_patrol.db"` relative to cwd; we don't modify the prof's file).
    simulate_mission = BashOperator(
        task_id="simulate_mission",
        bash_command="cd /data && python /opt/airflow/generate_patrol_db.py",
    )

    trigger_sync = TriggerDagRunOperator(
        task_id="trigger_sync",
        trigger_dag_id="drone_patrol_sync",
        wait_for_completion=False,
        reset_dag_run=False,
    )

    simulate_mission >> trigger_sync
