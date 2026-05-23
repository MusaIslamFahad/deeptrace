"""
scripts/evaluate_model.py
Evaluate a model from MLflow registry or local checkpoint on the test split.
Used in CI/CD and for ad-hoc evaluation.

Usage:
    python scripts/evaluate_model.py --model-uri models:/DeepTrace/Staging
    python scripts/evaluate_model.py --model-uri checkpoints/best_model.pt
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.architecture import build_model, TemperatureScaledModel, CLASS_NAMES, NUM_CLASSES
from data.dataset import build_dataloaders


def load_model_from_uri(model_uri: str, device: str) -> torch.nn.Module:
    if model_uri.startswith("models:/"):
        import mlflow.pytorch
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
    else:
        state = torch.load(model_uri, map_location=device)
        base = build_model(pretrained=False)
        if "temperature" in state:
            model = TemperatureScaledModel(base)
        else:
            model = base
        model.load_state_dict(state["state_dict"])

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model_uri: str, manifest_path: str, data_root: str,
             device: str = "cpu", output_dir: str = "reports") -> dict:

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)

    print(f"Loading model from: {model_uri}")
    model = load_model_from_uri(model_uri, device)

    print("Building test dataloader...")
    loaders = build_dataloaders(manifest_path, data_root, batch_size=32)
    test_loader = loaders["test"]

    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device_obj)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # Metrics
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES, output_dict=True
    )
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

    try:
        auc = roc_auc_score(
            np.eye(NUM_CLASSES)[all_labels], all_probs,
            multi_class="ovr", average="macro"
        )
    except Exception:
        auc = 0.0

    results = {
        "model_uri": model_uri,
        "test_f1_macro": round(f1_macro, 4),
        "test_auc_macro": round(float(auc), 4),
        "test_accuracy": round(float(report["accuracy"]), 4),
        "f1_per_class": {
            CLASS_NAMES[i]: round(float(f1_per_class[i]), 4)
            for i in range(NUM_CLASSES)
        },
        "n_samples": len(all_labels),
        "classification_report": report,
    }

    print(f"\n{'='*50}")
    print(f"  Test F1 (macro):   {f1_macro:.4f}")
    print(f"  Test AUC (macro):  {auc:.4f}")
    print(f"  Test Accuracy:     {report['accuracy']:.4f}")
    print(f"{'='*50}")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Save JSON results
    results_path = Path(output_dir) / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"DeepTrace — Confusion Matrix\nF1={f1_macro:.3f} | AUC={auc:.3f}")
    plt.tight_layout()
    cm_path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a DeepTrace model")
    parser.add_argument("--model-uri",   default="checkpoints/calibrated_model.pt")
    parser.add_argument("--manifest",    default="data/manifest.csv")
    parser.add_argument("--data-root",   default="data")
    parser.add_argument("--device",      default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output-dir",  default="reports")
    args = parser.parse_args()

    results = evaluate(
        model_uri=args.model_uri,
        manifest_path=args.manifest,
        data_root=args.data_root,
        device=args.device,
        output_dir=args.output_dir,
    )

    return results


if __name__ == "__main__":
    main()
