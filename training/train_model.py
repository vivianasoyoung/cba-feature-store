"""
train_model.py
--------------
Trains a fraud detection model using features from the Feast feature store.
Logs the experiment and model to MLflow for tracking.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'feature_repo', 'feature_repo'))

def load_features():
    df = pd.read_parquet(
        "../feature_repo/feature_repo/data/account_features.parquet"
    )
    return df

def train():
    print("Loading features from store...")
    df = load_features()

    feature_cols = [
        "transaction_count_7d",
        "total_spend_7d",
        "avg_transaction_value",
        "max_transaction_value",
        "unique_categories",
        "online_transaction_ratio",
        "night_transaction_ratio",
        "avg_daily_spend",
    ]

    X = df[feature_cols]
    y = df["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training on {len(X_train)} accounts, testing on {len(X_test)}...")

    mlflow.set_experiment("cba_fraud_detection")

    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        # Log params and metrics to MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 6)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("test_accounts", len(X_test))

        # Log feature importances
        importances = dict(zip(feature_cols, model.feature_importances_))
        for feat, imp in importances.items():
            mlflow.log_metric(f"importance_{feat}", round(imp, 4))

        # Save model
        mlflow.sklearn.log_model(model, "fraud_model")

        print(f"\nModel trained successfully!")
        print(f"AUC Score: {auc:.4f}")
        print(f"\nFeature Importances:")
        for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"  {feat}: {imp:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        return run_id

if __name__ == "__main__":
    train()
