import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch

from config import THYROIDXL_B2_CATEGORICAL_COLUMNS, THYROIDXL_B2_NUMERIC_COLUMNS
from src.datasets import build_tabular_preprocessor, load_thyroidxl_splits, prepare_final_tirads
from src.explainability import GradCAM, ThyroidXLCAMDataset, compute_mask_overlap, save_overlap_results
from src.models import HybridEfficientNet, ImageOnlyEfficientNet
from src.utils import device


def main():
    parser = argparse.ArgumentParser(description="ThyroidXL Experiment C-XL: Grad-CAM mask-overlap for A and B2.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--image_checkpoint", required=True)
    parser.add_argument("--hybrid_checkpoint", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()
    train_df, _, test_df = load_thyroidxl_splits(args.data_dir)
    test_df = prepare_final_tirads(test_df)
    train_df = prepare_final_tirads(train_df)
    test_df = test_df[test_df["mask_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    pre = build_tabular_preprocessor(train_df, THYROIDXL_B2_NUMERIC_COLUMNS, THYROIDXL_B2_CATEGORICAL_COLUMNS)
    x_test = pre.transform(test_df[THYROIDXL_B2_NUMERIC_COLUMNS + THYROIDXL_B2_CATEGORICAL_COLUMNS]).astype("float32")
    ds_a = ThyroidXLCAMDataset(test_df)
    ds_b2 = ThyroidXLCAMDataset(test_df, x_test)
    dev = device()
    model_a = ImageOnlyEfficientNet(pretrained=False).to(dev)
    model_a.load_state_dict(torch.load(args.image_checkpoint, map_location=dev))
    model_b2 = HybridEfficientNet(tabular_dim=x_test.shape[1], pretrained=False).to(dev)
    model_b2.load_state_dict(torch.load(args.hybrid_checkpoint, map_location=dev))
    model_a.eval()
    model_b2.eval()
    cam_a = GradCAM(model_a, model_a.gradcam_target_layer, dev)
    cam_b2 = GradCAM(model_b2, model_b2.gradcam_target_layer, dev)
    records = []
    n = len(ds_a) if args.max_images is None else min(args.max_images, len(ds_a))
    for idx in range(n):
        a = ds_a[idx]
        b = ds_b2[idx]
        heat_a = cam_a.generate(a["image"].unsqueeze(0))
        heat_b2 = cam_b2.generate(b["image"].unsqueeze(0), b["tabular"].unsqueeze(0))
        for experiment, heat in [("A_image_only", heat_a), ("B2_image_plus_final_tirads", heat_b2)]:
            records.append({"experiment": experiment, "index": idx, "patient_id": a["patient_id"], "label": a["label"], "image_path": a["image_path"], **compute_mask_overlap(heat, a["mask"])})
    cam_a.remove_hooks()
    cam_b2.remove_hooks()
    save_overlap_results(records, Path(args.output_dir) / "thyroidxl_expC_gradcam_mask_overlap.csv")


if __name__ == "__main__":
    main()
