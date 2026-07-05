# Explainable Hybrid AI for Thyroid Nodule Risk Stratification Using Ultrasound Images and TI-RADS Information

This repository contains cleaned, reusable code for a thyroid ultrasound AI research project comparing image-only deep learning models with hybrid models that combine ultrasound images with TI-RADS or structured thyroid nodule information.

## Research Question

Does combining ultrasound image AI with TI-RADS or structured nodule information improve benign vs malignant thyroid nodule prediction compared with image-only AI?

## Datasets

- Stanford thyroid ultrasound dataset
- ThyroidXL dataset

Datasets and medical images are not included. Users must download the Stanford and ThyroidXL datasets from their official sources and place them locally using the expected folder structures in `examples/`.

## Experiments

Stanford:

- A: image-only EfficientNet-B0
- B: image + TI-RADS descriptors
- C: Grad-CAM and mask-overlap explainability

ThyroidXL:

- A-XL: image-only EfficientNet-B0
- B1-XL: image + nodule metadata
- B2-XL: image + final TI-RADS category
- C-XL: Grad-CAM and mask-overlap explainability

## Main Results

| Dataset | Experiment | Input | AUC | Accuracy | F1 | Sensitivity | Specificity |
|---|---|---|---:|---:|---:|---:|---:|
| Stanford | A | Image only | 0.703704 | 0.862069 | 0.333333 | 0.500000 | 0.888889 |
| Stanford | B | Image + TI-RADS descriptors | 0.944444 | 0.931034 | 0.666667 | 1.000000 | 0.925926 |
| ThyroidXL | A-XL | Image only | 0.920900 | 0.834912 | 0.818452 | 0.779037 | 0.886010 |
| ThyroidXL | B1-XL | Image + nodule metadata | 0.928709 | 0.848444 | 0.8382 | 0.8215 | 0.8731 |
| ThyroidXL | B2-XL | Image + final TI-RADS category | 0.9648 | 0.9053 | 0.9006 | 0.8980 | 0.9119 |

The Stanford hybrid result is preliminary because the test set had only 2 malignant patients. ThyroidXL B2-XL is image + final TI-RADS category fusion, not full TI-RADS descriptor fusion.

## Install

```bash
pip install -r requirements.txt
```

## Run

All scripts accept `--data_dir` and `--output_dir`.

```bash
python scripts/run_thyroidxl_expA.py --data_dir /path/to/thyroidxl --output_dir results/
python scripts/run_thyroidxl_expB2.py --data_dir /path/to/thyroidxl --output_dir results/
python scripts/run_stanford_expA.py --data_dir /path/to/stanford --output_dir results/
```

## Data Note

This repository does not include datasets, raw ultrasound images, Stanford HDF5 files, ThyroidXL image folders, prepared `.npz` files, model checkpoints, Grad-CAM image folders, zip files, or large outputs.

## Disclaimer

This is a research project and not a clinical diagnostic tool.
