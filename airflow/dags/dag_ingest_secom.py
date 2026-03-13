"""
Ingests SECOM sensor data into BigQuery for the foundry defect detection pipeline.
"""

import os
from datetime import datetime

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from google.cloud import bigquery


SECOM_DATA_PATH = "/opt/airflow/data/raw/secom/secom.data"
SECOM_LABELS_PATH = "/opt/airflow/data/raw/secom/secom_labels.data"


def load_secom_to_bigquery(**context):
    """Load SECOM sensor readings and labels into BigQuery table secom_sensors."""
    project_id = os.getenv("GCP_PROJECT_ID")
    dataset_id = os.getenv("BQ_DATASET")
    if not project_id or not dataset_id:
        raise ValueError("GCP_PROJECT_ID and BQ_DATASET must be set")

    # Read sensor data
    df_data = pd.read_csv(
        SECOM_DATA_PATH,
        sep=r"\s+",
        header=None,
        engine="python",
    )
    df_data.columns = [f"feature_{i}" for i in range(590)]

    # Read labels data
    df_labels = pd.read_csv(
        SECOM_LABELS_PATH,
        sep=r"\s+",
        header=None,
        engine="python",
    )
    df_labels.columns = ["label", "date", "time"]
    df_labels["timestamp"] = df_labels["date"] + " " + df_labels["time"]
    df_labels = df_labels.drop(columns=["date", "time"])

    # Merge on index
    df = df_data.join(df_labels)

    # Add ingestion timestamp
    df["ingested_at"] = datetime.utcnow().isoformat() + "Z"

    # Replace NaN with None for BigQuery
    df = df.where(pd.notnull(df), None)

    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.secom_sensors"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    load_job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    load_job.result()

    print(f"Loaded {len(df)} rows into {table_ref}")


def quality_check(**context):
    """Run basic quality checks on the secom_sensors table."""
    project_id = os.getenv("GCP_PROJECT_ID")
    dataset_id = os.getenv("BQ_DATASET")
    if not project_id or not dataset_id:
        raise ValueError("GCP_PROJECT_ID and BQ_DATASET must be set")

    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.secom_sensors"

    query = f"""
    SELECT
      COUNT(*) AS total,
      COUNTIF(label = 1) AS fail_count,
      COUNTIF(label = -1) AS pass_count
    FROM `{table_ref}`
    """

    result = list(client.query(query).result())[0]

    total = result.total
    fail_count = result.fail_count
    pass_count = result.pass_count

    assert total > 1000, f"total rows {total} is not > 1000"
    assert fail_count > 0 and pass_count > 0, (
        f"both labels must be present; fail={fail_count}, pass={pass_count}"
    )

    print(
        f"Quality check passed: total={total}, "
        f"fail={fail_count}, pass={pass_count}"
    )


with DAG(
    dag_id="ingest_secom_data",
    schedule_interval="@once",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["foundry", "ingestion"],
    doc_md=__doc__,
) as dag:
    load_secom_to_bigquery = PythonOperator(
        task_id="load_secom_to_bigquery",
        python_callable=load_secom_to_bigquery,
    )

    quality_check = PythonOperator(
        task_id="quality_check",
        python_callable=quality_check,
    )

    load_secom_to_bigquery >> quality_check

