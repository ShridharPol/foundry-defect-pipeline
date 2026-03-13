"""
Ingests casting inspection images and metadata from local data into GCP (GCS and BigQuery)
for the foundry defect detection pipeline.
"""

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from google.cloud import storage, bigquery


DATA_ROOT = "/opt/airflow/data/raw/casting/casting_data/casting_data"
IMAGE_EXTENSIONS = (".jpeg", ".jpg")

LABEL_MAP = {"def_front": "defective", "ok_front": "ok"}


def upload_images_to_gcs(**context):
    """Walk casting data dir and upload .jpeg/.jpg to GCS casting/images/<filename>."""
    bucket_name = os.getenv("GCS_BUCKET_RAW")
    if not bucket_name:
        raise ValueError("GCS_BUCKET_RAW environment variable is not set")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    count = 0
    for dirpath, _dirnames, filenames in os.walk(DATA_ROOT):
        for name in filenames:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                local_path = os.path.join(dirpath, name)
                blob = bucket.blob(f"casting/images/{name}")
                blob.upload_from_filename(local_path)
                count += 1
    print(f"Uploaded {count} files to gs://{bucket_name}/casting/images/")


def load_metadata_to_bigquery(**context):
    """Build metadata rows from casting images and load into BigQuery casting_metadata."""
    project_id = os.getenv("GCP_PROJECT_ID")
    dataset_id = os.getenv("BQ_DATASET")
    bucket_name = os.getenv("GCS_BUCKET_RAW")
    if not project_id or not dataset_id or not bucket_name:
        raise ValueError("GCP_PROJECT_ID, BQ_DATASET, and GCS_BUCKET_RAW must be set")
    client = bigquery.Client(project=project_id)
    rows = []
    ingested_at = datetime.utcnow().isoformat() + "Z"
    base = os.path.normpath(DATA_ROOT)
    for dirpath, _dirnames, filenames in os.walk(DATA_ROOT):
        for name in filenames:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                rel = os.path.relpath(dirpath, base)
                parts = rel.split(os.sep)
                split = parts[0] if len(parts) >= 1 else "unknown"
                parent_folder = parts[1] if len(parts) >= 2 else "unknown"
                label = LABEL_MAP.get(parent_folder, parent_folder)
                gcs_uri = f"gs://{bucket_name}/casting/images/{name}"
                rows.append({
                    "filename": name,
                    "label": label,
                    "split": split,
                    "gcs_uri": gcs_uri,
                    "ingested_at": ingested_at,
                })
    if not rows:
        print("No image rows to load")
        return
    table_ref = f"{project_id}.{dataset_id}.casting_metadata"
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        schema=[
            bigquery.SchemaField("filename", "STRING"),
            bigquery.SchemaField("label", "STRING"),
            bigquery.SchemaField("split", "STRING"),
            bigquery.SchemaField("gcs_uri", "STRING"),
            bigquery.SchemaField("ingested_at", "STRING"),
        ],
    )
    job = client.load_table_from_json(rows, table_ref, job_config=job_config)
    job.result()
    print(f"Loaded {len(rows)} rows into {table_ref}")


def quality_check(**context):
    """Query BigQuery for row counts and assert quality constraints."""
    project_id = os.getenv("GCP_PROJECT_ID")
    dataset_id = os.getenv("BQ_DATASET")
    if not project_id or not dataset_id:
        raise ValueError("GCP_PROJECT_ID and BQ_DATASET must be set")
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.casting_metadata"
    query = f"""
    SELECT
        COUNT(*) AS total,
        COUNTIF(label = 'defective') AS defective_count,
        COUNTIF(label = 'ok') AS ok_count,
        COUNTIF(split = 'train') AS train_count,
        COUNTIF(split = 'test') AS test_count
    FROM `{table_ref}`
    """
    result = list(client.query(query).result())[0]
    total = result.total
    defective_count = result.defective_count
    ok_count = result.ok_count
    assert total > 5000, f"total rows {total} is not > 5000"
    assert defective_count > 0 and ok_count > 0, (
        f"both labels must be present; defective={defective_count}, ok={ok_count}"
    )
    print(
        f"Quality check passed: total={total}, "
        f"defective={defective_count}, ok={ok_count}, "
        f"train={result.train_count}, test={result.test_count}"
    )


with DAG(
    dag_id="ingest_casting_data",
    schedule_interval="@once",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["foundry", "ingestion"],
    doc_md=__doc__,
) as dag:
    upload_images_to_gcs = PythonOperator(
        task_id="upload_images_to_gcs",
        python_callable=upload_images_to_gcs,
    )
    load_metadata_to_bigquery = PythonOperator(
        task_id="load_metadata_to_bigquery",
        python_callable=load_metadata_to_bigquery,
    )
    quality_check = PythonOperator(
        task_id="quality_check",
        python_callable=quality_check,
    )

    upload_images_to_gcs >> load_metadata_to_bigquery >> quality_check
