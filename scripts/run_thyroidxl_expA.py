import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, CHECKPOINTS_DIR, EPOCHS, LEARNING_RATE, NUM_WORKERS
from src.datasets import ThyroidXLImageDataset, image_transforms, load_thyroidxl_splits
from src.evaluate import aggregate_patient_predictions, binary_metrics, predict_frame_level, save_evaluation
from src.models import ImageOnlyEfficientNet
from src.train import fit_model, make_bce_loss
from src.utils import device, ensure_dirs, set_seed


def main():
    parser = argparse.ArgumentParser(description="ThyroidXL Experiment A-XL: image-only EfficientNet-B0.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--checkpoint_dir", default=str(CHECKPOINTS_DIR))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    set_seed()
    ensure_dirs([args.output_dir, args.checkpoint_dir])
    train_df, val_df, test_df = load_thyroidxl_splits(args.data_dir)
    train_loader = DataLoader(ThyroidXLImageDataset(train_df, image_transforms(True)), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(ThyroidXLImageDataset(val_df, image_transforms(False)), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(ThyroidXLImageDataset(test_df, image_transforms(False)), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    dev = device()
    model = ImageOnlyEfficientNet(pretrained=True).to(dev)
    criterion = make_bce_loss(train_df.label, dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    ckpt = Path(args.checkpoint_dir) / "thyroidxl_expA_image_only_best.pt"
    fit_model(model, train_loader, val_loader, optimizer, criterion, dev, ckpt, args.epochs).to_csv(Path(args.output_dir) / "thyroidxl_expA_training_history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    frame_df = predict_frame_level(model, test_loader, dev).join(test_df.reset_index(drop=True), on="source_row", rsuffix="_meta")
    patient_df = aggregate_patient_predictions(frame_df)
    summary = {"experiment": "thyroidxl_A_image_only", **binary_metrics(patient_df.true_label, patient_df.probability_malignant)}
    save_evaluation(frame_df, patient_df, summary, args.output_dir, "thyroidxl_expA")
    pd.DataFrame([{k: v for k, v in summary.items() if k != "classification_report"}]).to_csv(Path(args.output_dir) / "thyroidxl_summary.csv", index=False)


if __name__ == "__main__":
    main()
