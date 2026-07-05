# Evaluating the Diagnostic and Explainability Impact of Adding TI-RADS Score to Deep Learning for Thyroid Nodule Malignancy Classification

This repository contains research code, experiment scripts, documentation, and result summaries for evaluating whether adding structured TI-RADS score information improves deep learning-based thyroid nodule malignancy classification and explainability.

## Research Question

Does combining ultrasound image AI with TI-RADS or structured nodule information improve benign vs malignant thyroid nodule prediction compared with image-only AI?

## Datasets

- Stanford AIMI Thyroid Ultrasound Cine-clip dataset
- ThyroidXL dataset

Datasets and medical images are not included. Users must obtain access to the Stanford AIMI and ThyroidXL datasets from their official sources and place them locally using the expected folder structures in `examples/`.

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

## Citation and Data Attribution

This repository contains research code, experiment scripts, documentation, and result summaries for evaluating whether adding structured TI-RADS score information improves deep learning-based thyroid nodule malignancy classification and explainability.

The datasets used in this project are not included in this repository. Users must obtain access to the datasets from their official sources and follow their respective terms of use.

Stanford dataset citation:

Stanford AIMI. (2026). Thyroid Ultrasound Cine-clip (Version 1.0) [Dataset]. Redivis. [https://stanford.redivis.com/datasets/2n5z-4m99zdpyp?v=1.0](https://stanford.redivis.com/datasets/2n5z-4m99zdpyp?v=1.0)

ThyroidXL citation:

Duong, V. H., Vu, H., Phan, H. D., Nguyen, D. Q., Pham, D. H., Le, Q. T., Nguyen, B. S., Do, T. D., Dinh, V. S., Nguyen, T. C., Pham, H. H., & Ngo, D. H. (2025). ThyroidXL: Advancing Thyroid Nodule Diagnosis with an Expert-Labeled, Pathology-Validated Dataset. Springer-Verlag. [https://doi.org/10.1007/978-3-032-05182-0_60](https://doi.org/10.1007/978-3-032-05182-0_60)

### Data Availability

* Stanford AIMI Thyroid Ultrasound Cine-clip dataset: available through Redivis at [https://stanford.redivis.com/datasets/2n5z-4m99zdpyp?v=1.0](https://stanford.redivis.com/datasets/2n5z-4m99zdpyp?v=1.0)
* ThyroidXL dataset: cite the official ThyroidXL paper. No official dataset download URL is included in this repository.

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

## License

This repository is licensed under the MIT License. See the `LICENSE` file for details.

This license applies only to the code, documentation, scripts, and repository materials created for this project. It does not apply to the Stanford AIMI Thyroid Ultrasound Cine-clip dataset, the ThyroidXL dataset, or any third-party datasets, papers, images, or external resources cited or used in this project. Users must obtain datasets from their official sources and follow their respective terms of use.

## Disclaimer

This is a research project and not a clinical diagnostic tool.
