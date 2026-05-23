"""
DeepTrace Training Pipeline
Three-phase training: head-only → partial unfreeze → full fine-tune
Integrates: MLflow experiment tracking, Optuna HPO, DVC params
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score, confusion_matrix
)
import mlflow
import mlflow.pytorch
import optuna
from tqdm import tqdm

from model.architecture import (
    build_model, TemperatureScaledModel, CLASS_NAMES, NUM_CLASSES
)
from data.dataset import build_dataloaders, CLASS_TO_IDX


# ---------------------------------------------------------------------------
# Params (loaded from params.yaml for DVC compatibility)
# ---------------------------------------------------------------------------

def load_params(params_path: str = "params.yaml") -> Dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, config: Dict, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Using device: {self.device}")

        self.dataloaders = build_dataloaders(
            manifest_path=config["data"]["manifest_path"],
            data_root=config["data"]["root"],
            batch_size=config["training"]["batch_size"],
            image_size=config["training"]["image_size"],
            num_workers=config["training"].get("num_workers", 4),
        )

        self.model = build_model(
            pretrained=True,
            dropout=config["model"]["dropout"],
            use_fft=config["model"]["use_fft"],
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config["training"]["label_smoothing"]
        )

        self.best_val_f1 = 0.0
        self.best_checkpoint_path = "checkpoints/best_model.pt"
        Path("checkpoints").mkdir(exist_ok=True)

    def _get_optimizer(self, lr: float) -> optim.Optimizer:
        return optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=self.config["training"]["weight_decay"],
        )

    def _train_epoch(self, optimizer: optim.Optimizer,
                     scheduler=None) -> Dict[str, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        pbar = tqdm(self.dataloaders["train"], desc="  train", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        if scheduler:
            scheduler.step()

        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return {
            "train/loss": total_loss / total,
            "train/acc": correct / total,
            "train/f1": f1,
        }

    @torch.no_grad()
    def _val_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels, all_probs = [], [], []

        for images, labels in tqdm(self.dataloaders["val"], desc="  val  ", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        try:
            auc = roc_auc_score(
                np.eye(NUM_CLASSES)[all_labels],
                np.array(all_probs),
                multi_class="ovr", average="macro"
            )
        except Exception:
            auc = 0.0

        return {
            "val/loss": total_loss / total,
            "val/acc": correct / total,
            "val/f1": f1,
            "val/auc": auc,
        }

    def _save_checkpoint(self, epoch: int, metrics: Dict, tag: str = "best"):
        path = f"checkpoints/{tag}_model.pt"
        torch.save({
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "class_names": CLASS_NAMES,
        }, path)
        return path

    def train_phase(self, phase: int, epochs: int, lr: float,
                    run_id: Optional[str] = None) -> float:
        """
        Train one phase. Returns best val F1 achieved.
        Phase 1: freeze backbones, train head only
        Phase 2: unfreeze top blocks, lower LR
        Phase 3: full fine-tune, cosine annealing
        """
        if phase == 1:
            self.model.freeze_backbones()
            print(f"[Phase 1] Training head only for {epochs} epochs @ lr={lr}")
        elif phase == 2:
            self.model.unfreeze_top_blocks()
            print(f"[Phase 2] Partial unfreeze for {epochs} epochs @ lr={lr}")
        elif phase == 3:
            self.model.unfreeze_all()
            print(f"[Phase 3] Full fine-tune for {epochs} epochs @ lr={lr}")

        optimizer = self._get_optimizer(lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(optimizer, scheduler)
            val_metrics = self._val_epoch()
            elapsed = time.time() - t0

            all_metrics = {**train_metrics, **val_metrics,
                          "epoch": epoch, "phase": phase, "elapsed_s": elapsed}

            print(f"  Epoch {epoch:02d}/{epochs} | "
                  f"loss={val_metrics['val/loss']:.4f} | "
                  f"acc={val_metrics['val/acc']:.4f} | "
                  f"f1={val_metrics['val/f1']:.4f} | "
                  f"auc={val_metrics['val/auc']:.4f} | "
                  f"{elapsed:.1f}s")

            mlflow.log_metrics(all_metrics, step=(phase - 1) * 100 + epoch)

            if val_metrics["val/f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["val/f1"]
                self._save_checkpoint(epoch, val_metrics, tag="best")
                print(f"  ✓ New best F1: {self.best_val_f1:.4f}")

        return self.best_val_f1

    def full_train(self) -> float:
        cfg = self.config["training"]
        best_f1 = 0.0

        best_f1 = self.train_phase(1, cfg["phase1_epochs"], cfg["phase1_lr"])
        best_f1 = self.train_phase(2, cfg["phase2_epochs"], cfg["phase2_lr"])
        best_f1 = self.train_phase(3, cfg["phase3_epochs"], cfg["phase3_lr"])

        return best_f1

    @torch.no_grad()
    def evaluate_test(self) -> Dict:
        """Load best checkpoint and evaluate on test set."""
        state = torch.load(self.best_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()

        all_preds, all_labels, all_probs = [], [], []
        for images, labels in tqdm(self.dataloaders["test"], desc="  test "):
            images = images.to(self.device)
            probs = torch.softmax(self.model(images), dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

        report = classification_report(
            all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True
        )
        cm = confusion_matrix(all_labels, all_preds).tolist()

        result = {
            "test/f1_macro": report["macro avg"]["f1-score"],
            "test/acc": report["accuracy"],
            "classification_report": report,
            "confusion_matrix": cm,
        }

        Path("reports").mkdir(exist_ok=True)
        with open("reports/test_results.json", "w") as f:
            json.dump(result, f, indent=2)

        print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
        return result

    def calibrate_model(self) -> str:
        """Apply temperature scaling on best model and save calibrated version."""
        state = torch.load(self.best_checkpoint_path, map_location=self.device)
        base_model = build_model(pretrained=False).to(self.device)
        base_model.load_state_dict(state["state_dict"])

        calibrated = TemperatureScaledModel(base_model).to(self.device)
        calibrated.calibrate(self.dataloaders["val"], device=str(self.device))

        cal_path = "checkpoints/calibrated_model.pt"
        torch.save({
            "state_dict": calibrated.state_dict(),
            "temperature": calibrated.temperature.item(),
            "class_names": CLASS_NAMES,
        }, cal_path)
        print(f"Calibrated model saved to {cal_path}")
        mlflow.log_artifact(cal_path)
        return cal_path


# ---------------------------------------------------------------------------
# Optuna HPO objective
# ---------------------------------------------------------------------------

def optuna_objective(trial: optuna.Trial, base_config: Dict) -> float:
    config = base_config.copy()
    config["model"] = {
        "dropout":  trial.suggest_float("dropout", 0.1, 0.5),
        "use_fft":  trial.suggest_categorical("use_fft", [True, False]),
    }
    config["training"]["phase1_lr"] = trial.suggest_float("phase1_lr", 1e-4, 5e-3, log=True)
    config["training"]["phase3_lr"] = trial.suggest_float("phase3_lr", 1e-5, 1e-4, log=True)
    config["training"]["label_smoothing"] = trial.suggest_float("label_smoothing", 0.05, 0.2)

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(trial.params)
        trainer = Trainer(config)
        best_f1 = trainer.full_train()
        mlflow.log_metric("best_val_f1", best_f1)

    return best_f1


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DeepTrace Training")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--hpo", action="store_true", help="Run Optuna HPO")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--experiment", default="DeepTrace")
    args = parser.parse_args()

    config = load_params(args.params)

    mlflow.set_tracking_uri(config.get("mlflow", {}).get("tracking_uri", "http://localhost:5000"))
    mlflow.set_experiment(args.experiment)

    if args.hpo:
        print(f"[HPO] Running {args.n_trials} Optuna trials...")
        study = optuna.create_study(
            direction="maximize",
            study_name="deeptrace_hpo",
            storage="sqlite:///optuna.db",
            load_if_exists=True,
        )
        with mlflow.start_run(run_name="hpo_study"):
            study.optimize(
                lambda t: optuna_objective(t, config),
                n_trials=args.n_trials,
                show_progress_bar=True,
            )
        print(f"Best trial: F1={study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        # Apply best params and do full run
        config["model"]["dropout"] = study.best_params["dropout"]
        config["model"]["use_fft"] = study.best_params["use_fft"]

    with mlflow.start_run(run_name="full_training"):
        mlflow.log_params({
            "model.dropout":        config["model"]["dropout"],
            "model.use_fft":        config["model"]["use_fft"],
            "training.batch_size":  config["training"]["batch_size"],
            "training.image_size":  config["training"]["image_size"],
            "data.manifest_path":   config["data"]["manifest_path"],
        })

        trainer = Trainer(config)
        best_f1 = trainer.full_train()

        print("\n[Evaluation] Running on test set...")
        test_metrics = trainer.evaluate_test()
        mlflow.log_metrics({
            "test/f1_macro": test_metrics["test/f1_macro"],
            "test/acc":      test_metrics["test/acc"],
        })
        mlflow.log_artifact("reports/test_results.json")

        print("\n[Calibration] Applying temperature scaling...")
        cal_path = trainer.calibrate_model()

        print(f"\n✓ Training complete. Best val F1: {best_f1:.4f}")
        print(f"  Test F1 (macro): {test_metrics['test/f1_macro']:.4f}")

        # Register model in MLflow registry
        mlflow.pytorch.log_model(
            trainer.model,
            artifact_path="model",
            registered_model_name="DeepTrace",
        )


if __name__ == "__main__":
    main()
