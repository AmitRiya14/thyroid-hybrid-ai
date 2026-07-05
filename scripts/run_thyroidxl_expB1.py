import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from config import THYROIDXL_B1_CATEGORICAL_COLUMNS, THYROIDXL_B1_NUMERIC_COLUMNS
from scripts.run_thyroidxl_expB2 import run_hybrid


def main():
    parser = argparse.ArgumentParser(description="ThyroidXL Experiment B1-XL: image + nodule metadata.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    run_hybrid(
        args,
        numeric_cols=THYROIDXL_B1_NUMERIC_COLUMNS,
        categorical_cols=THYROIDXL_B1_CATEGORICAL_COLUMNS,
        prefix="thyroidxl_expB1",
        experiment="thyroidxl_B1_image_plus_nodule_metadata",
        checkpoint_name="thyroidxl_expB1_image_plus_nodule_metadata_best.pt",
    )


if __name__ == "__main__":
    main()
