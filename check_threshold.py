import mlflow
import sys

with open("model_info.txt") as f:
    run_id = f.read().strip()

run = mlflow.get_run(run_id)
accuracy = run.data.metrics.get("val_accuracy")  # Or train_accuracy if you prefer

print("Validation Accuracy:", accuracy)
if accuracy < 0.85:
    sys.exit(1)