import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_WORKERS, THYROIDXL_B2_CATEGORICAL_COLUMNS, THYROIDXL_B2_NUMERIC_COLUMNS
from src.datasets import ThyroidXLHybridDataset, build_tabular_preprocessor, image_transforms, load_thyroidxl_splits, prepare_final_tirads
from src.evaluate import aggregate_patient_predictions, binary_metrics, predict_frame_level, save_evaluation
from src.models import HybridEfficientNet
from src.train import fit_model, make_bce_loss
from src.utils import device, ensure_dirs, set_seed


def run_hybrid(args, numeric_cols, categorical_cols, prefix, experiment, checkpoint_name):
    set_seed()
    ensure_dirs([args.output_dir, args.checkpoint_dir])
    train_df, val_df, test_df = load_thyroidxl_splits(args.data_dir)
    if "final_tirads" in categorical_cols:
        train_df, val_df, test_df = prepare_final_tirads(train_df), prepare_final_tirads(val_df), prepare_final_tirads(test_df)
    preprocessor = build_tabular_preprocessor(train_df, numeric_cols, categorical_cols)
    columns = numeric_cols + categorical_cols
    x_train = preprocessor.transform(train_df[columns]).astype("float32")
    x_val = preprocessor.transform(val_df[columns]).astype("float32")
    x_test = preprocessor.transform(test_df[columns]).astype("float32")
    train_loader = DataLoader(ThyroidXLHybridDataset(train_df, x_train, image_transforms(True)), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(ThyroidXLHybridDataset(val_df, x_val, image_transforms(False)), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(ThyroidXLHybridDataset(test_df, x_test, image_transforms(False)), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    dev = device()
    model = HybridEfficientNet(tabular_dim=x_train.shape[1], pretrained=True).to(dev)
    criterion = make_bce_loss(train_df.label, dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    ckpt = Path(args.checkpoint_dir) / checkpoint_name
    fit_model(model, train_loader, val_loader, optimizer, criterion, dev, ckpt, args.epochs, hybrid=True).to_csv(Path(args.output_dir) / f"{prefix}_training_history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    frame_df = predict_frame_level(model, test_loader, dev, hybrid=True).join(test_df.reset_index(drop=True), on="source_row", rsuffix="_meta")
    patient_df = aggregate_patient_predictions(frame_df)
    summary = {"experiment": experiment, **binary_metrics(patient_df.true_label, patient_df.probability_malignant)}
    save_evaluation(frame_df, patient_df, summary, args.output_dir, prefix)


def main():
    parser = argparse.ArgumentParser(description="ThyroidXL Experiment B2-XL: image + final TI-RADS category.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    run_hybrid(
        args,
        numeric_cols=THYROIDXL_B2_NUMERIC_COLUMNS,
        categorical_cols=THYROIDXL_B2_CATEGORICAL_COLUMNS,
        prefix="thyroidxl_expB2",
        experiment="thyroidxl_B2_image_plus_final_tirads_category",
        checkpoint_name="thyroidxl_expB2_image_plus_final_tirads_best.pt",
    )


if __name__ == "__main__":
    main()
