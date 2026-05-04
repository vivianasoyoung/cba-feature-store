# CBA ML Feature Store

An end-to-end ML feature store pipeline for banking fraud detection. Computes account-level features from raw transaction history, stores them in Feast for consistent offline training and online serving, trains a fraud detection model tracked in MLflow, and serves real-time risk scores via a FastAPI endpoint.

## Architecture

```
Raw transaction data (cba-banking-pipeline)
        │
        ▼
Feature Computation (Python)
   └── 9 features per account (spend patterns, velocity, channel behaviour)
        │
        ▼
Feast Feature Store
   ├── Offline store (Parquet) — historical features for model training
   └── Online store (SQLite)  — latest features for real-time serving
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
ML Model Training                     FastAPI Serving Endpoint
(scikit-learn RandomForest)           POST /score → real-time risk score
        │
        ▼
MLflow Experiment Tracking
   └── AUC, feature importances, model artefacts
```

## Tech Stack

| Layer | Tool |
|---|---|
| Feature store | Feast |
| ML framework | scikit-learn |
| Experiment tracking | MLflow |
| API serving | FastAPI + Uvicorn |
| Offline store | Parquet |
| Online store | SQLite |

## Quick Start

### Prerequisites
- Python 3.10+
- cba-banking-pipeline data available at `../cba-banking-pipeline/data/raw/transactions.csv`

### 1. Install dependencies

```bash
pip install 'feast[postgres]' mlflow scikit-learn pandas fastapi uvicorn psycopg2-binary
```

### 2. Compute features

```bash
python features/compute_features.py
```

Generates 9 features for 500 accounts from raw transaction history.

### 3. Apply Feast feature store

```bash
cd feature_repo/feature_repo
feast apply
feast materialize-incremental "$(date +%Y-%m-%dT%H:%M:%S)"
```

### 4. Train the model

```bash
cd ../..
python training/train_model.py
```

### 5. Launch MLflow UI

```bash
mlflow ui --port 5001 --host 0.0.0.0
```

Open http://localhost:5001 to view experiment runs and metrics.

### 6. Start the API

```bash
uvicorn serving.fraud_api:app --port 8001 --reload
```

### 7. Score a transaction

```bash
curl -X POST http://localhost:8001/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "test-001",
    "account_id": "ACC0003936",
    "amount": 9500.00,
    "merchant_category": "ATM Withdrawal",
    "channel": "ONLINE"
  }'
```

## Features

| Feature | Description |
|---|---|
| transaction_count_7d | Number of transactions in last 7 days |
| total_spend_7d | Total spend in last 7 days |
| avg_transaction_value | Average transaction value |
| max_transaction_value | Maximum single transaction value |
| unique_categories | Number of unique merchant categories used |
| online_transaction_ratio | Ratio of online to total transactions |
| night_transaction_ratio | Ratio of night-time transactions (before 6am or after 10pm) |
| avg_daily_spend | Average daily spend across 6 months |
| is_high_risk | Binary flag — 1 if account shows high risk patterns |

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| /health | GET | Health check — returns accounts loaded count |
| /score | POST | Score a transaction — returns risk score and fraud flag |
| /account/{account_id} | GET | Retrieve stored features for an account |

## Model Performance

| Metric | Value |
|---|---|
| Model | Random Forest (100 estimators, max depth 6) |
| AUC Score | 1.00 |
| Top feature | night_transaction_ratio (76% importance) |
| Training accounts | 400 |
| Test accounts | 100 |

## Project Structure

```
cba-feature-store/
├── features/
│   └── compute_features.py      # Feature engineering from raw transactions
├── feature_repo/
│   └── feature_repo/
│       ├── feature_definitions.py   # Feast entity and feature view definitions
│       ├── feature_store.yaml       # Feast configuration
│       └── data/
│           └── account_features.parquet
├── training/
│   └── train_model.py           # Model training with MLflow tracking
├── serving/
│   └── fraud_api.py             # FastAPI real-time scoring endpoint
└── README.md
```
