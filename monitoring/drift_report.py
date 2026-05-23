"""
DeepTrace Model Monitoring
Evidently AI drift detection + Slack alerting.
Run as a weekly cron job: python -m monitoring.drift_report
"""

import json
import os
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Embedding logger (called by InferenceService during predictions)
# ---------------------------------------------------------------------------

def log_prediction(
    image_id: str,
    predicted_source: str,
    confidence: float,
    per_class_probs: dict,
    embedding: Optional[list] = None,
    log_dir: str = "monitoring/logs",
):
    """Append prediction record to daily JSONL log."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    log_path = Path(log_dir) / f"predictions_{today}.jsonl"

    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "image_id": image_id,
        "predicted_source": predicted_source,
        "confidence": confidence,
        **{f"prob_{k}": v for k, v in per_class_probs.items()},
    }
    if embedding:
        record["embedding_norm"] = float(np.linalg.norm(embedding))

    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Load prediction logs as DataFrame
# ---------------------------------------------------------------------------

def load_prediction_logs(log_dir: str = "monitoring/logs",
                          days: int = 7) -> pd.DataFrame:
    """Load the past N days of prediction logs."""
    records = []
    for i in range(days):
        date = (datetime.date.today() - datetime.timedelta(days=i)).isoformat()
        log_path = Path(log_dir) / f"predictions_{date}.jsonl"
        if log_path.exists():
            with open(log_path) as f:
                for line in f:
                    try:
                        records.append(json.loads(line.strip()))
                    except Exception:
                        pass

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def run_drift_report(
    reference_path: str = "monitoring/reference_stats.json",
    log_dir: str = "monitoring/logs",
    report_output: str = "reports/drift/weekly_drift.html",
    alert_threshold: float = 0.25,
) -> dict:
    """
    Computes prediction distribution drift between reference and current week.
    Uses Evidently if available, falls back to manual KL divergence.
    Returns a drift summary dict.
    """
    current_df = load_prediction_logs(log_dir, days=7)

    if current_df.empty:
        print("[DriftReport] No prediction logs found for the past 7 days.")
        return {"drift_detected": False, "reason": "no_data"}

    # Load or create reference stats
    if Path(reference_path).exists():
        with open(reference_path) as f:
            reference_stats = json.load(f)
    else:
        # Bootstrap reference from first week of data
        print("[DriftReport] No reference stats found — creating from current data.")
        _save_reference_stats(current_df, reference_path)
        return {"drift_detected": False, "reason": "bootstrapped_reference"}

    # Compute current class distribution
    source_cols = [c for c in current_df.columns if c.startswith("prob_")]
    if not source_cols:
        return {"drift_detected": False, "reason": "no_prob_columns"}

    current_dist = current_df[source_cols].mean().to_dict()
    reference_dist = reference_stats.get("mean_probs", {})

    # KL divergence
    eps = 1e-8
    current_vals = np.array([current_dist.get(k, eps) for k in sorted(current_dist)])
    reference_vals = np.array([reference_dist.get(k, eps) for k in sorted(reference_dist)])
    current_vals = np.clip(current_vals, eps, 1)
    reference_vals = np.clip(reference_vals, eps, 1)

    kl_div = float(np.sum(current_vals * np.log(current_vals / reference_vals)))

    # Confidence distribution shift
    conf_shift = abs(
        current_df["confidence"].mean() - reference_stats.get("mean_confidence", 0.8)
    )

    drift_detected = kl_div > alert_threshold or conf_shift > 0.1

    summary = {
        "drift_detected": drift_detected,
        "kl_divergence": round(kl_div, 4),
        "confidence_shift": round(conf_shift, 4),
        "current_class_dist": current_dist,
        "reference_class_dist": reference_dist,
        "current_n_predictions": len(current_df),
        "alert_threshold": alert_threshold,
        "report_date": datetime.date.today().isoformat(),
    }

    # Try Evidently report
    try:
        _run_evidently_report(current_df, reference_stats, report_output)
        summary["evidently_report"] = report_output
    except ImportError:
        print("[DriftReport] Evidently not installed; skipping HTML report.")
    except Exception as e:
        print(f"[DriftReport] Evidently report failed: {e}")

    # Save summary
    summary_path = "reports/drift/summary.json"
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if drift_detected:
        _send_slack_alert(summary)

    print(f"[DriftReport] KL={kl_div:.4f} | conf_shift={conf_shift:.4f} | "
          f"drift={'YES ⚠️' if drift_detected else 'NO ✓'}")

    return summary


def _save_reference_stats(df: pd.DataFrame, output_path: str):
    source_cols = [c for c in df.columns if c.startswith("prob_")]
    stats = {
        "mean_probs": df[source_cols].mean().to_dict(),
        "mean_confidence": float(df["confidence"].mean()),
        "n_samples": len(df),
        "created_at": datetime.datetime.utcnow().isoformat(),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[DriftReport] Reference stats saved to {output_path}")


def _run_evidently_report(current_df: pd.DataFrame,
                           reference_stats: dict,
                           output_path: str):
    from evidently.report import Report
    from evidently.metrics import DatasetDriftMetric, DataDriftTable
    from evidently import ColumnMapping

    # Reconstruct reference DataFrame from stats
    n_ref = reference_stats.get("n_samples", 1000)
    prob_cols = {k: np.random.dirichlet(np.ones(5), n_ref)[:, i]
                 for i, k in enumerate(reference_stats.get("mean_probs", {}))}
    reference_df = pd.DataFrame(prob_cols)

    # Align columns
    cols = [c for c in reference_df.columns if c in current_df.columns]

    report = Report(metrics=[DatasetDriftMetric(), DataDriftTable()])
    report.run(
        reference_data=reference_df[cols],
        current_data=current_df[cols].dropna(),
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(output_path)
    print(f"[DriftReport] Evidently HTML report saved to {output_path}")


def _send_slack_alert(summary: dict):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("[DriftReport] No SLACK_WEBHOOK_URL set; skipping alert.")
        return

    try:
        import urllib.request
        message = {
            "text": (
                f"⚠️ *DeepTrace Drift Alert* — {summary['report_date']}\n"
                f"KL divergence: `{summary['kl_divergence']}`  "
                f"(threshold: `{summary['alert_threshold']}`)\n"
                f"Confidence shift: `{summary['confidence_shift']}`\n"
                f"N predictions this week: `{summary['current_n_predictions']}`\n"
                f"Consider retraining: `dvc repro train`"
            )
        }
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(message).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
        print("[DriftReport] Slack alert sent.")
    except Exception as e:
        print(f"[DriftReport] Slack alert failed: {e}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run DeepTrace drift report")
    parser.add_argument("--reference", default="monitoring/reference_stats.json")
    parser.add_argument("--log-dir", default="monitoring/logs")
    parser.add_argument("--output", default="reports/drift/weekly_drift.html")
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    result = run_drift_report(
        reference_path=args.reference,
        log_dir=args.log_dir,
        report_output=args.output,
        alert_threshold=args.threshold,
    )
    print(json.dumps(result, indent=2))
