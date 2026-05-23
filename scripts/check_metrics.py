"""
scripts/check_metrics.py
CI gate: fails with exit code 1 if model metrics are below threshold.
Reads from reports/evaluation_results.json or MLflow staging model.

Usage:
    python scripts/check_metrics.py --min-f1 0.82
    python scripts/check_metrics.py --min-f1 0.85 --min-auc 0.90
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_from_json(results_path: str, min_f1: float, min_auc: float) -> bool:
    if not Path(results_path).exists():
        print(f"[check_metrics] No results file at {results_path}. Skipping gate.")
        return True

    with open(results_path) as f:
        results = json.load(f)

    f1  = results.get("test_f1_macro", 0.0)
    auc = results.get("test_auc_macro", 0.0)

    print(f"[check_metrics] F1={f1:.4f} (min={min_f1}) | AUC={auc:.4f} (min={min_auc})")

    passed = True
    if f1 < min_f1:
        print(f"  ✗ F1 {f1:.4f} < threshold {min_f1}")
        passed = False
    else:
        print(f"  ✓ F1 {f1:.4f} ≥ threshold {min_f1}")

    if auc < min_auc:
        print(f"  ✗ AUC {auc:.4f} < threshold {min_auc}")
        passed = False
    else:
        print(f"  ✓ AUC {auc:.4f} ≥ threshold {min_auc}")

    return passed


def check_from_mlflow(min_f1: float, min_auc: float) -> bool:
    try:
        import mlflow
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions("DeepTrace", stages=["Staging"])

        if not versions:
            print("[check_metrics] No model in Staging. Gate skipped.")
            return True

        version = versions[0]
        run = client.get_run(version.run_id)
        metrics = run.data.metrics

        f1  = metrics.get("test/f1_macro", metrics.get("val/f1", 0.0))
        auc = metrics.get("test/auc_macro", metrics.get("val/auc", 0.0))

        print(f"[check_metrics] MLflow Staging v{version.version}: "
              f"F1={f1:.4f} | AUC={auc:.4f}")

        passed = (f1 >= min_f1) and (auc >= min_auc)
        if passed:
            print("  ✓ All metrics meet thresholds. Safe to promote to Production.")
        else:
            print("  ✗ Metrics below threshold. Do NOT promote to Production.")

        return passed

    except Exception as e:
        print(f"[check_metrics] MLflow check failed: {e}. Gate skipped.")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-f1",  type=float, default=0.82)
    parser.add_argument("--min-auc", type=float, default=0.88)
    parser.add_argument("--results", default="reports/evaluation_results.json")
    parser.add_argument("--source",  choices=["json", "mlflow", "auto"], default="auto")
    args = parser.parse_args()

    if args.source == "json":
        passed = check_from_json(args.results, args.min_f1, args.min_auc)
    elif args.source == "mlflow":
        passed = check_from_mlflow(args.min_f1, args.min_auc)
    else:
        # Auto: try JSON first, fall back to MLflow
        if Path(args.results).exists():
            passed = check_from_json(args.results, args.min_f1, args.min_auc)
        else:
            passed = check_from_mlflow(args.min_f1, args.min_auc)

    if not passed:
        print("\n[check_metrics] ✗ Gate FAILED. Blocking pipeline.")
        sys.exit(1)

    print("\n[check_metrics] ✓ Gate PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
