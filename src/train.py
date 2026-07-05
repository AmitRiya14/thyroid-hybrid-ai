from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score


def positive_class_weight(labels) -> torch.Tensor:
    labels = np.asarray(labels).astype(int)
    negative = np.sum(labels == 0)
    positive = np.sum(labels == 1)
    return torch.tensor([negative / max(positive, 1)], dtype=torch.float32)


def make_bce_loss(labels, device) -> torch.nn.BCEWithLogitsLoss:
    return torch.nn.BCEWithLogitsLoss(pos_weight=positive_class_weight(labels).to(device))


def train_one_epoch(model, loader, optimizer, criterion, device, hybrid: bool = False) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        if hybrid:
            images, tabular, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            logits = model(images, tabular)
        else:
            images, labels = batch[0].to(device), batch[1].to(device)
            logits = model(images)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(logits, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device, hybrid: bool = False) -> dict:
    model.eval()
    total_loss = 0.0
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            if hybrid:
                images, tabular, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                logits = model(images, tabular)
            else:
                images, y = batch[0].to(device), batch[1].to(device)
                logits = model(images)
            loss = criterion(logits, y.float())
            total_loss += loss.item() * images.size(0)
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs.extend(batch_probs)
            labels.extend(y.detach().cpu().numpy())
    preds = (np.asarray(probs) >= 0.5).astype(int)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = np.nan
    return {
        "loss": total_loss / len(loader.dataset),
        "auc": auc,
        "f1": f1_score(labels, preds, zero_division=0),
    }


def fit_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    checkpoint_path,
    epochs: int = 10,
    hybrid: bool = False,
) -> pd.DataFrame:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_auc = -np.inf
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, hybrid=hybrid)
        val = validate(model, val_loader, criterion, device, hybrid=hybrid)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"], "val_auc": val["auc"], "val_f1": val["f1"]}
        if not np.isnan(val["auc"]) and val["auc"] > best_auc:
            best_auc = val["auc"]
            torch.save(model.state_dict(), checkpoint_path)
            row["saved_best"] = True
        else:
            row["saved_best"] = False
        history.append(row)
        print(row)
    return pd.DataFrame(history)
