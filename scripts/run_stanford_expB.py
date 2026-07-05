import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, CHECKPOINTS_DIR, EPOCHS, LEARNING_RATE, NUM_WORKERS, RANDOM_SEED
from src.datasets import HybridDataset, encode_stanford_tirads, image_transforms, load_stanford_dataframe, masks_from_patient_ids, patient_split
from src.evaluate import aggregate_patient_predictions, binary_metrics, predict_frame_level, save_evaluation
from src.models import HybridEfficientNet
from src.train import fit_model, make_bce_loss
from src.utils import device, ensure_dirs, set_seed


def main():
    parser = argparse.ArgumentParser(description="Stanford Experiment B: image + TI-RADS descriptors.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--checkpoint_dir", default=str(CHECKPOINTS_DIR))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    set_seed(RANDOM_SEED)
    ensure_dirs([args.output_dir, args.checkpoint_dir])
    df, images, _ = load_stanford_dataframe(args.data_dir)
    meta_df = pd.read_csv(Path(args.data_dir) / "metadata.csv")
    train_ids, val_ids, test_ids = patient_split(df.patient_id, df.label)
    train_mask, val_mask, test_mask = masks_from_patient_ids(df.patient_id, train_ids, val_ids, test_ids)
    tabular, feature_cols, _ = encode_stanford_tirads(meta_df, df, train_mask)
    pd.Series(feature_cols, name="feature").to_csv(Path(args.output_dir) / "stanford_expB_tirads_feature_columns.csv", index=False)
    train_ds = HybridDataset(images[train_mask], tabular[train_mask], df.label.values[train_mask], df.patient_id.values[train_mask], image_transforms(True))
    val_ds = HybridDataset(images[val_mask], tabular[val_mask], df.label.values[val_mask], df.patient_id.values[val_mask], image_transforms(False))
    test_ds = HybridDataset(images[test_mask], tabular[test_mask], df.label.values[test_mask], df.patient_id.values[test_mask], image_transforms(False))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    dev = device()
    model = HybridEfficientNet(tabular_dim=tabular.shape[1], pretrained=True).to(dev)
    criterion = make_bce_loss(df.label.values[train_mask], dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ckpt = Path(args.checkpoint_dir) / "stanford_expB_hybrid_tirads_best.pt"
    history = fit_model(model, train_loader, val_loader, optimizer, criterion, dev, ckpt, args.epochs, hybrid=True)
    history.to_csv(Path(args.output_dir) / "stanford_expB_training_history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    frame_df = predict_frame_level(model, test_loader, dev, hybrid=True)
    patient_df = aggregate_patient_predictions(frame_df)
    summary = {"experiment": "stanford_B_image_plus_tirads_descriptors", **binary_metrics(patient_df.true_label, patient_df.probability_malignant)}
    save_evaluation(frame_df, patient_df, summary, args.output_dir, "stanford_expB")
    print(summary["classification_report"])


if __name__ == "__main__":
    main()
