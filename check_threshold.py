#!/usr/bin/env python
# coding: utf-8

import os
import mlflow

# Use environment variable if available, else default to local mlflow.db
mlflow_uri = os.environ.get("MLFLOW_URI", "sqlite:///mlflow.db")
print(f"Using MLflow URI: {mlflow_uri}")
mlflow.set_tracking_uri(mlflow_uri)

# Read Run ID
try:
    with open("model_info.txt") as f:
        run_id = f.read().strip()
    print(f"Run ID: {run_id}")
except FileNotFoundError:
    raise FileNotFoundError("model_info.txt not found! Make sure validation job ran successfully.")

# Get the run from MLflow
run = mlflow.get_run(run_id)

# Get accuracy - try different possible metric names
accuracy = run.data.metrics.get("val_accuracy")
if accuracy is None:
    # Try alternative metric names
    accuracy = run.data.metrics.get("val_accuracy", 0.0)
    if accuracy == 0.0:
        print("Available metrics:", list(run.data.metrics.keys()))
        raise ValueError("Could not find val_accuracy in run metrics")

print(f"Validation Accuracy: {accuracy:.4f}")

THRESHOLD = 0.85
if accuracy < THRESHOLD:
    raise ValueError(f"Accuracy {accuracy:.4f} below threshold {THRESHOLD}! Deployment stopped.")
else:
    print(f"✅ Accuracy {accuracy:.4f} meets threshold {THRESHOLD}. Deployment can proceed.")