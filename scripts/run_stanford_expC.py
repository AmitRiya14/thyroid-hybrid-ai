import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch

from src.datasets import image_transforms, load_stanford_dataframe, masks_from_patient_ids, patient_split
from src.explainability import GradCAM, compute_mask_overlap, save_overlap_results
from src.models import ImageOnlyEfficientNet
from src.utils import device


def main():
    parser = argparse.ArgumentParser(description="Stanford Experiment C: Grad-CAM mask-overlap analysis for selected frames.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--max_frames", type=int, default=6)
    args = parser.parse_args()
    df, images, masks = load_stanford_dataframe(args.data_dir)
    if masks is None:
        raise ValueError("Stanford HDF5 did not contain a 'mask' dataset.")
    _, _, test_ids = patient_split(df.patient_id, df.label)
    _, _, test_mask = masks_from_patient_ids(df.patient_id, set(), set(), test_ids)
    transform = image_transforms(False)
    dev = device()
    model = ImageOnlyEfficientNet(pretrained=False).to(dev)
    model.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    model.eval()
    cam = GradCAM(model, model.gradcam_target_layer, dev)
    records = []
    for source_idx in list(df.index[test_mask])[: args.max_frames]:
        image = transform(__import__("src.datasets", fromlist=["image_to_pil"]).image_to_pil(images[source_idx])).unsqueeze(0)
        heatmap = cam.generate(image)
        overlap = compute_mask_overlap(heatmap, masks[source_idx])
        records.append({"experiment": "stanford_C_image_only_gradcam", "patient_id": df.loc[source_idx, "patient_id"], "source_index": int(source_idx), **overlap})
    cam.remove_hooks()
    save_overlap_results(records, Path(args.output_dir) / "stanford_expC_gradcam_mask_overlap.csv")


if __name__ == "__main__":
    main()
