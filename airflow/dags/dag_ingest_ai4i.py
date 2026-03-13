"""
Ingests AI4I predictive maintenance data into BigQuery for the foundry defect detection pipeline.
"""

import os
import re
from datetime import datetime

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from google.cloud import bigquery


AI4I_DIR = "/opt/airflow/data/raw/ai4i"


def _to_snake_case(name: str) -> str:
    """Convert a column name to snake_case (lowercase, non-alnum -> underscore)."""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name.lower()


def load_ai4i_to_bigquery(**context):
    """Load AI4I CSV data into BigQuery table ai4i_maintenance."""
    project_id = os.getenv("GCP_PROJECT_ID")
    dataset_id = os.getenv("BQ_DATASET")
    if not project_id or not dataset_id:
        raise ValueError("GCP_PROJECT_ID and BQ_DATASET must be set")

    # Find first CSV file in directory
    files = [f for f in os.listdir(AI4I_DIR) if f.lower().endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {AI4I_DIR}")
    csv_path = os.path.join(AI4I_DIR, files[0])

    df = pd.read_csv(csv_path)

    # Rename columns to snake_case
    df.columns = [_to_snake_case(c) for c in df.columns]

    # Add ingestion timestamp
    df["ingested_at"] = datetime.utcnow().isoformat() + "Z"

    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.ai4i_maintenance"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    load_job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    load_job.result()

    print(f"Loaded {len(df)} rows into {table_ref}")


def quality_check(**context):
    """Run basic quality checks on the ai4i_maintenance table."""
    project_id = os.getenv("GCP_PROJECT_ID")
    dataset_id = os.getenv("BQ_DATASET")
    if not project_id or not dataset_id:
        raise ValueError("GCP_PROJECT_ID and BQ_DATASET must be set")

    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.ai4i_maintenance"

    query = f"""
    SELECT
      COUNT(*) AS total,
      COUNTIF(machine_failure = 1) AS machine_failure_count,
      SUM(twf) AS twf_count,
      SUM(hdf) AS hdf_count,
      SUM(pwf) AS pwf_count,
      SUM(osf) AS osf_count,
      SUM(rnf) AS rnf_count
    FROM `{table_ref}`
    """

    result = list(client.query(query).result())[0]

    total = result.total
    machine_failure_count = result.machine_failure_count

    assert total > 5000, f"total rows {total} is not > 5000"
    assert machine_failure_count > 0, (
        f"machine failure count must be > 0; got {machine_failure_count}"
    )

    print(
        "Quality check passed: "
        f"total={total}, "
        f"machine_failure={machine_failure_count}, "
        f"twf={result.twf_count}, "
        f"hdf={result.hdf_count}, "
        f"pwf={result.pwf_count}, "
        f"osf={result.osf_count}, "
        f"rnf={result.rnf_count}"
    )


with DAG(
    dag_id="ingest_ai4i_data",
    schedule_interval="@once",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["foundry", "ingestion"],
    doc_md=__doc__,
) as dag:
    load_ai4i_to_bigquery = PythonOperator(
        task_id="load_ai4i_to_bigquery",
        python_callable=load_ai4i_to_bigquery,
    )

    quality_check = PythonOperator(
        task_id="quality_check",
        python_callable=quality_check,
    )

    load_ai4i_to_bigquery >> quality_check

