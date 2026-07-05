# Stanford Experiments

Project: Evaluating the Diagnostic and Explainability Impact of Adding TI-RADS Score to Deep Learning for Thyroid Nodule Malignancy Classification

The Stanford AIMI Thyroid Ultrasound Cine-clip dataset is not included in this repository. Users must obtain access through the official Redivis source and follow its terms of use.

## Experiment A

Image-only EfficientNet-B0 using 5 representative frames per patient. Frames are selected from the central portion of each clip, patients are split at patient level, and test prediction is averaged across each patient's selected frames.

The old text-based label derivation was removed. The correct Stanford label is `histopath_diagnosis`, already numeric where `0 = benign` and `1 = malignant`.

Final reported result:

| AUC | Accuracy | F1 | Sensitivity | Specificity |
|---:|---:|---:|---:|---:|
| 0.703704 | 0.862069 | 0.333333 | 0.500000 | 0.888889 |

## Experiment B

Hybrid image + TI-RADS model. The image branch is EfficientNet-B0 and the tabular branch uses one-hot encoded TI-RADS descriptors:

- `ti_rads_composition`
- `ti_rads_echogenicity`
- `ti_rads_shape`
- `ti_rads_margin`
- `ti_rads_echogenicfoci`
- `ti_rads_level`

Experiment B uses the same patient-level train/validation/test split as Experiment A.

Final reported result:

| AUC | Accuracy | F1 | Sensitivity | Specificity |
|---:|---:|---:|---:|---:|
| 0.944444 | 0.931034 | 0.666667 | 1.000000 | 0.925926 |

This Stanford result is preliminary because the test set had only 2 malignant patients.

## Experiment C

Grad-CAM explainability for the Experiment A image-only model and the Experiment B hybrid model image branch. Heatmaps are compared with segmentation masks using:

- `cam_inside_mask_fraction`
- `mask_covered_by_cam_fraction`
- `iou`

The current selected-example analysis used 6 selected frames. Experiment C is an interpretation layer, not another accuracy model.
