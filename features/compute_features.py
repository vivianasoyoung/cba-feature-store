"""
compute_features.py
-------------------
Computes fraud detection features from raw transaction data
and saves them as a parquet file for the Feast feature store.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def compute_account_features(transactions_path: str, output_path: str):
    print("Loading transaction data...")
    df = pd.read_csv(transactions_path, parse_dates=["transaction_date"])

    # Only use DEBIT transactions for spend features
    debits = df[df["transaction_type"] == "DEBIT"].copy()
    debits["hour"] = pd.to_datetime(debits["transaction_date"]).dt.hour
    debits["is_online"] = (debits["channel"] == "ONLINE").astype(int)
    debits["is_night"] = ((debits["hour"] < 6) | (debits["hour"] > 22)).astype(int)

    print("Computing features per account...")

    features = debits.groupby("account_id").agg(
        transaction_count_7d    =("transaction_id", "count"),
        total_spend_7d          =("amount", "sum"),
        avg_transaction_value   =("amount", "mean"),
        max_transaction_value   =("amount", "max"),
        unique_categories       =("merchant_category", "nunique"),
        online_transaction_ratio=("is_online", "mean"),
        night_transaction_ratio =("is_night", "mean"),
        avg_daily_spend         =("amount", lambda x: x.sum() / 180),
    ).reset_index()

    # Round floats
    float_cols = ["total_spend_7d", "avg_transaction_value", "max_transaction_value",
                  "online_transaction_ratio", "night_transaction_ratio", "avg_daily_spend"]
    features[float_cols] = features[float_cols].round(2)

    # High risk flag: large max transaction OR high night ratio OR high online ratio
    features["is_high_risk"] = (
        (features["max_transaction_value"] > 5000) |
        (features["night_transaction_ratio"] > 0.3) |
        (features["online_transaction_ratio"] > 0.7)
    ).astype(int)

    # Feast requires an event_timestamp column
    features["event_timestamp"] = datetime.now()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_parquet(output_path, index=False)
    print(f"Saved {len(features):,} account feature rows → {output_path}")
    print(f"High risk accounts: {features['is_high_risk'].sum()}")
    return features

if __name__ == "__main__":
    compute_account_features(
        transactions_path="/Users/vivianayou/projects/cba-banking-pipeline/data/raw/transactions.csv",
        output_path="../feature_repo/feature_repo/data/account_features.parquet"
    )
