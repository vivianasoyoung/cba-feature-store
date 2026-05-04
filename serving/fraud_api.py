"""
fraud_api.py
------------
FastAPI endpoint that serves real-time fraud risk scores.
Looks up account features from the Feast online store
and returns a risk score for incoming transactions.
"""

import os
import sys
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'feature_repo', 'feature_repo'))

app = FastAPI(
    title="CBA Fraud Detection API",
    description="Real-time fraud risk scoring using Feast feature store",
    version="1.0.0"
)

# Load features into memory on startup
FEATURES_PATH = os.path.join(
    os.path.dirname(__file__), '..', 
    'feature_repo', 'feature_repo', 'data', 'account_features.parquet'
)
features_df = pd.read_parquet(FEATURES_PATH).set_index('account_id')

class Transaction(BaseModel):
    transaction_id: str
    account_id: str
    amount: float
    merchant_category: str
    channel: str

class FraudScore(BaseModel):
    transaction_id: str
    account_id: str
    risk_score: float
    is_fraud: bool
    reasons: list[str]
    scored_at: str

@app.get("/health")
def health():
    return {"status": "ok", "accounts_loaded": len(features_df)}

@app.post("/score", response_model=FraudScore)
def score_transaction(txn: Transaction):
    reasons = []
    risk_score = 0.0

    # Look up account features from store
    if txn.account_id in features_df.index:
        feat = features_df.loc[txn.account_id]

        # Rule 1: Large transaction relative to account average
        if txn.amount > feat["avg_transaction_value"] * 5:
            reasons.append(f"Amount ${txn.amount} is 5x above account average")
            risk_score += 40

        # Rule 2: Account is already flagged high risk
        if feat["is_high_risk"] == 1:
            reasons.append("Account has high risk transaction history")
            risk_score += 30

        # Rule 3: Large absolute amount
        if txn.amount > 5000:
            reasons.append(f"Large transaction amount: ${txn.amount}")
            risk_score += 20

        # Rule 4: Online channel with high amount
        if txn.channel == "ONLINE" and txn.amount > feat["avg_transaction_value"] * 3:
            reasons.append("Large online transaction relative to account history")
            risk_score += 10

    else:
        # Unknown account — flag it
        reasons.append("Unknown account ID — no history available")
        risk_score = 50.0

    risk_score = min(risk_score, 100.0)
    is_fraud = risk_score >= 50

    return FraudScore(
        transaction_id=txn.transaction_id,
        account_id=txn.account_id,
        risk_score=risk_score,
        is_fraud=is_fraud,
        reasons=reasons,
        scored_at=datetime.now().isoformat()
    )

@app.get("/account/{account_id}")
def get_account_features(account_id: str):
    if account_id not in features_df.index:
        return {"error": f"Account {account_id} not found"}
    feat = features_df.loc[account_id].to_dict()
    feat.pop("event_timestamp", None)
    return {"account_id": account_id, "features": feat}
