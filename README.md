# Evaluating the Diagnostic and Explainability Impact of Adding TI-RADS Score to Deep Learning for Thyroid Nodule Malignancy Classification

This repository contains research code, experiment scripts, documentation, and result summaries comparing image-only and hybrid image + TI-RADS models for diagnostic performance and explainability in thyroid nodule malignancy classification.

## Research Question

Does adding structured TI-RADS information to ultrasound-based deep learning improve thyroid nodule malignancy classification, and does any performance improvement correspond to more clinically meaningful model explainability?

## Datasets

- Stanford AIMI Thyroid Ultrasound Cine-clip dataset
- ThyroidXL dataset

Datasets and medical images are not included. Users must obtain access to the Stanford AIMI and ThyroidXL datasets from their official sources and place them locally using the expected folder structures in `examples/`.

## Experiments

Stanford:

- A: image-only model
- B: image + TI-RADS hybrid model
- C: explainability analysis comparing Grad-CAM attention and TI-RADS feature effects

ThyroidXL:

- A-XL: image-only EfficientNet-B0
- B1-XL: image + nodule metadata
- B2-XL: image + final TI-RADS category
- C-XL: Experiment C-XL evaluates explainability by comparing Grad-CAM attention maps from the image-only model and the image + final TI-RADS model against nodule masks.

## Main Results

| Dataset | Experiment | Input | AUC | Accuracy | F1 | Sensitivity | Specificity |
|---|---|---|---:|---:|---:|---:|---:|
| Stanford | A | Image only | 0.703704 | 0.862069 | 0.333333 | 0.500000 | 0.888889 |
| Stanford | B | Image + TI-RADS descriptors | 0.944444 | 0.931034 | 0.666667 | 1.000000 | 0.925926 |
| ThyroidXL | A-XL | Image only | 0.920900 | 0.834912 | 0.818452 | 0.779037 | 0.886010 |
| ThyroidXL | B1-XL | Image + nodule metadata | 0.928709 | 0.848444 | 0.8382 | 0.8215 | 0.8731 |
| ThyroidXL | B2-XL | Image + final TI-RADS category | 0.9648 | 0.9053 | 0.9006 | 0.8980 | 0.9119 |

The Stanford hybrid result is preliminary because the test set had only 2 malignant patients. ThyroidXL B2-XL is image + final TI-RADS category fusion, not full TI-RADS descriptor fusion.

### Explainability findings

Although the hybrid image + TI-RADS model improved diagnostic performance, the Grad-CAM mask-overlap analysis did not show stronger localization within the nodule mask. In the selected Stanford examples, the image-only model had higher mean CAM/mask overlap than the hybrid model. TI-RADS probability-change and permutation-importance analyses also did not identify one dominant individual TI-RADS descriptor. These findings suggest that improved classification performance does not automatically imply improved explainability.

| Explainability metric | Image-only A | Hybrid B |
|---|---:|---:|
| Mean CAM inside mask | 0.170 | 0.159 |
| Mean IoU | 0.146 | 0.058 |
| Frames analyzed | 6 | 6 |

These explainability results are preliminary because they are based on a small selected Stanford subset.

### ThyroidXL explainability findings

Grad-CAM was used to visualize where each model focused in the ultrasound image. The Grad-CAM maps were compared with nodule masks to evaluate whether model attention aligned with the actual nodule region. This analysis compares Experiment A, the image-only model, with Experiment B2, the image + final TI-RADS model.

| Explainability metric | Experiment A: Image only | Experiment B2: Image + final TI-RADS | Interpretation |
|---|---:|---:|---|
| Mean IoU | 0.115 | 0.183 | B2 had stronger overlap between Grad-CAM attention and the nodule mask. |
| Mean Dice | 0.170 | 0.252 | B2 showed better attention-mask alignment than A. |
| Mean inside-mask activation | 0.076 | 0.112 | B2 had stronger Grad-CAM activation within the nodule region. |
| Mean CAM area pixels | 8945 | 7609 | B2 used a smaller average attention area, suggesting more focused activation. |
| Mean mask area pixels | 3915 | 3915 | The average nodule mask area was the same because both models were evaluated on the same images. |
| Images analyzed | 2094 | 2094 | Full ThyroidXL test-image explainability set. |

Overall, the ThyroidXL Grad-CAM results suggest that Experiment B2 improved visual localization compared with Experiment A, with higher mean IoU, Dice, and inside-mask activation. However, visual inspection and label-stratified analysis suggest that this improvement is stronger for malignant nodules than for benign nodules. Therefore, the explainability improvement should be interpreted cautiously rather than as uniform improvement across all cases.

Diagnostic performance was evaluated at the patient level by averaging image-level predictions for each patient. Grad-CAM explainability was evaluated at the image level because each heatmap is generated for a single ultrasound image.

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
