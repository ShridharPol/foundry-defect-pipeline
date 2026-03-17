# Project Notes — Foundry Defect Detection Pipeline
> Internal reproduction guide. Not for public presentation.

---

## Environment

| Item | Value |
|---|---|
| OS | Windows 11 |
| Python | 3.8.10 (venv at `venv/`) |
| PyTorch | 2.4.1+cu121 (CUDA 12.1, RTX 4060) |
| GCP Project | foundry-defect-pipeline-2 |
| GCP Region | us-central1 |
| BigQuery Dataset | foundry_raw |
| GCS Buckets | foundry-pipeline-raw, foundry-pipeline-processed |
| dbt Project | dbt/foundry_quality |
| MLflow | Local, runs at http://127.0.0.1:5000 |

---

## How to Start the Project

### 1. Activate venv
```powershell
cd "E:\Hobby Project\foundry-defect\foundry-defect-pipeline"
venv\Scripts\activate
```

### 2. Start Airflow (Docker)
```powershell
docker compose up -d
```
Airflow UI → http://localhost:8080 (admin/admin)

### 3. Start MLflow UI
```powershell
mlflow ui
```
MLflow UI → http://127.0.0.1:5000

### 4. Shut everything down
```powershell
docker compose down
```

---

## GCP Authentication

If you get quota project warnings:
```powershell
gcloud config set project foundry-defect-pipeline-2
gcloud auth application-default set-quota-project foundry-defect-pipeline-2
```

---

## dbt

```powershell
cd dbt\foundry_quality
dbt run        # materialize all models
dbt test       # run all data tests
dbt run --select mart_defect_features   # run single model
cd ../..       # go back to project root
```

All 7 models land in BigQuery dataset `foundry_raw` (not `foundry_marts` — the dbt project.yml target is set to foundry_raw).

---

## ML Training

### CNN (MobileNetV2)
```powershell
python notebooks/ml/train_cnn.py
```
- Data: `data/raw/casting/casting_data/casting_data/`
- Output: `mlflow/models/best_mobilenetv2.pth`
- Logged to MLflow experiment: `foundry_defect_detection`
- Run name: `mobilenetv2_casting_classifier`

### XGBoost + SHAP
```powershell
python notebooks/ml/train_xgboost.py
```
- Data: pulled from BigQuery `foundry_raw.ai4i_maintenance`
- Output: `mlflow/models/best_xgboost.json`
- Logged to MLflow experiment: `foundry_defect_detection`
- Run name: `xgboost_process_anomaly_v2_no_leakage`
- **Note:** failure type flags (twf, hdf, pwf, osf, rnf) are intentionally excluded to prevent data leakage

---

## API (FastAPI)

### Run locally
```powershell
cd notebooks\ml\serve
uvicorn main:app --reload --port 8080
```
- Demo UI → http://localhost:8080/ui
- API Docs → http://localhost:8080/docs

### Run unit tests
```powershell
cd notebooks\ml\serve
python -m pytest test_api.py -v
```
Expected: 13/13 passing

### Build Docker image
```powershell
cd notebooks\ml\serve
docker build -t foundry-defect-api .
docker run -p 8080:8080 foundry-defect-api
```

### Redeploy to Cloud Run
```powershell
docker tag foundry-defect-api us-central1-docker.pkg.dev/foundry-defect-pipeline-2/foundry-repo/foundry-defect-api:latest
docker push us-central1-docker.pkg.dev/foundry-defect-pipeline-2/foundry-repo/foundry-defect-api:latest
gcloud run deploy foundry-defect-api `
  --image us-central1-docker.pkg.dev/foundry-defect-pipeline-2/foundry-repo/foundry-defect-api:latest `
  --platform managed `
  --region us-central1 `
  --allow-unauthenticated `
  --port 8080 `
  --memory 2Gi
```

### Live URLs
- Demo UI: https://foundry-defect-api-241173171739.us-central1.run.app/ui
- API Docs: https://foundry-defect-api-241173171739.us-central1.run.app/docs

---

## Dashboard

- Looker Studio: https://lookerstudio.google.com/reporting/772148c6-c390-42a6-9c28-3ea3d2109d5e
- Data source: BigQuery `foundry-defect-pipeline-2.foundry_raw`
- 4 pages: Defect Trend, Shift Analysis, Process Health, Model Performance

---

## Known Issues & Gotchas

- **Windows multiprocessing** — always keep `num_workers=0` in DataLoader on Windows
- **dbt dataset** — models land in `foundry_raw`, not `foundry_marts`. This is intentional (dbt project.yml target)
- **MLflow artifacts** — stored locally in `mlruns/`, not in GCS
- **XGBoost leakage** — v1 run used failure type flags and got 99.9% (leaky). v2 uses sensor-only features and gets 98.55% (honest). Always use v2.
- **Cloud Run cold start** — ~22s cold start, ~230ms warm. Scale-to-zero is enabled to save costs.
- **Python 3.8 warnings** — Google libraries warn about Python 3.8 EOL. Harmless, ignore them.
- **GCP quota project** — if you see quota warnings after switching projects, run the gcloud auth command above

---

## Data Locations

| Dataset | Local Path | BigQuery Table | GCS Path |
|---|---|---|---|
| Casting images | data/raw/casting/ | casting_metadata | foundry-pipeline-raw/casting/images/ |
| SECOM sensors | data/raw/secom/ | secom_sensors | — |
| AI4I maintenance | data/raw/ai4i/ | ai4i_maintenance | — |

---

## Key Numbers

- Casting images: 7,340 total (6,633 train / 715 val)
- GCS image storage: ~85MB
- SECOM rows: 1,567
- AI4I rows: 10,000
- CNN accuracy: 100% (453 TP, 262 TN, 0 FP, 0 FN)
- XGBoost accuracy: 98.55% · AUC: 97.26%
- API warm latency: ~230ms · Cold start: ~22s
- Unit tests: 13/13 passing
- dbt models: 7 (3 staging, 4 marts) · dbt tests: 9