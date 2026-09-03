"""Training and test evaluation artifact generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def _safe_float(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _predict_labels(model, X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return np.array([], dtype=int)
    predictions = model.predict(X, verbose=0)
    return np.argmax(predictions, axis=1)


def _true_labels(y: np.ndarray) -> np.ndarray:
    return np.argmax(y, axis=1) if len(y) else np.array([], dtype=int)


def _split_metrics(model, X: np.ndarray, y: np.ndarray) -> dict:
    if len(X) == 0:
        return {"accuracy": None}
    y_true = _true_labels(y)
    y_pred = _predict_labels(model, X)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }


def save_evaluation_artifacts(
    model,
    actions: list[str],
    output_dir: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    history=None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "train": _split_metrics(model, X_train, y_train),
        "validation": _split_metrics(model, X_val, y_val),
        "test": _split_metrics(model, X_test, y_test),
    }

    y_true = _true_labels(y_test)
    y_pred = _predict_labels(model, X_test) if len(X_test) else np.array([], dtype=int)
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(actions))),
        target_names=actions,
        zero_division=0,
    ) if len(X_test) else "No test samples available.\n"
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(actions)))) if len(X_test) else np.zeros((len(actions), len(actions)), dtype=int)
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted"] + actions)
        for action, row in zip(actions, matrix):
            writer.writerow([action] + [int(value) for value in row])

    _save_plots(output_dir, matrix, actions, history)
    history_dict = history.history if history is not None else {}
    (output_dir / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "history": history_dict}, indent=2),
        encoding="utf-8",
    )
    return metrics


def _save_plots(output_dir: Path, matrix: np.ndarray, actions: list[str], history) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if matrix.size:
        fig, ax = plt.subplots(figsize=(max(6, len(actions) * 0.8), max(5, len(actions) * 0.7)))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(actions)), actions, rotation=45, ha="right")
        ax.set_yticks(range(len(actions)), actions)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "confusion_matrix.png")
        plt.close(fig)

    if history is not None:
        hist = history.history
        fig, ax = plt.subplots(figsize=(8, 5))
        for key in ["categorical_accuracy", "val_categorical_accuracy", "loss", "val_loss"]:
            if key in hist:
                ax.plot(hist[key], label=key)
        ax.legend()
        ax.set_xlabel("Epoch")
        fig.tight_layout()
        fig.savefig(output_dir / "training_history.png")
        plt.close(fig)

