"""
scripts/setup_mlflow.py
One-time setup: creates the MLflow experiment and registers an initial model placeholder.

Usage:
    python scripts/setup_mlflow.py
    python scripts/setup_mlflow.py --tracking-uri http://my-mlflow-server:5000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_mlflow(tracking_uri: str, experiment_name: str = "DeepTrace"):
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    print(f"[setup_mlflow] Tracking URI: {tracking_uri}")

    # Create experiment
    try:
        exp_id = mlflow.create_experiment(
            experiment_name,
            tags={
                "project":     "DeepTrace",
                "task":        "multi-class-image-classification",
                "description": "AI image provenance detection — SD, MJ, DALL-E 3, Flux, Real",
            },
        )
        print(f"[setup_mlflow] Created experiment '{experiment_name}' (id={exp_id})")
    except mlflow.exceptions.MlflowException:
        exp = mlflow.get_experiment_by_name(experiment_name)
        print(f"[setup_mlflow] Experiment '{experiment_name}' already exists (id={exp.experiment_id})")

    # Create registered model
    client = mlflow.MlflowClient()
    try:
        client.create_registered_model(
            name="DeepTrace",
            tags={"framework": "pytorch", "classes": "5"},
            description=(
                "Multi-generator AI image provenance classifier. "
                "Detects: stable_diffusion, midjourney, dalle3, flux, real."
            ),
        )
        print("[setup_mlflow] Registered model 'DeepTrace' created.")
    except mlflow.exceptions.MlflowException:
        print("[setup_mlflow] Registered model 'DeepTrace' already exists.")

    print("\n✓ MLflow setup complete.")
    print(f"  Open UI at: {tracking_uri}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Registered model: DeepTrace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--experiment",   default="DeepTrace")
    args = parser.parse_args()
    setup_mlflow(args.tracking_uri, args.experiment)


if __name__ == "__main__":
    main()
