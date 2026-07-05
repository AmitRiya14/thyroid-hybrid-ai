from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score


def predict_frame_level(model, loader, device, hybrid: bool = False):
    model.eval()
    probs, labels, patient_ids, rows = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if hybrid:
                images, tabular, y, pids = batch[:4]
                logits = model(images.to(device), tabular.to(device))
            else:
                images, y, pids = batch[:3]
                logits = model(images.to(device))
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
            probs.extend(batch_probs)
            labels.extend(y.numpy().astype(int))
            patient_ids.extend(list(pids))
            if len(batch) > 4:
                source_rows = batch[4]
                if hasattr(source_rows, "detach"):
                    source_rows = source_rows.detach().cpu().numpy().tolist()
                rows.extend(list(source_rows))
            else:
                rows.extend([None] * len(y))
    return pd.DataFrame(
        {
            "patient_id": patient_ids,
            "true_label": labels,
            "probability_malignant": probs,
            "source_row": rows,
        }
    )


def aggregate_patient_predictions(frame_df: pd.DataFrame, patient_col: str = "patient_id") -> pd.DataFrame:
    patient_df = (
        frame_df.groupby(patient_col)
        .agg(
            true_label=("true_label", "first"),
            probability_malignant=("probability_malignant", "mean"),
            number_of_frames=("probability_malignant", "count"),
        )
        .reset_index()
    )
    patient_df["predicted_label"] = (patient_df["probability_malignant"] >= 0.5).astype(int)
    return patient_df


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "auc": auc,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "classification_report": classification_report(y_true, y_pred, target_names=["benign", "malignant"], zero_division=0),
    }


def save_evaluation(frame_df: pd.DataFrame, patient_df: pd.DataFrame, summary: dict, output_dir, prefix: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_df.to_csv(output_dir / f"{prefix}_frame_predictions.csv", index=False)
    patient_df.to_csv(output_dir / f"{prefix}_patient_predictions.csv", index=False)
    summary_for_csv = {k: v for k, v in summary.items() if k != "classification_report"}
    pd.DataFrame([summary_for_csv]).to_csv(output_dir / f"{prefix}_summary.csv", index=False)
