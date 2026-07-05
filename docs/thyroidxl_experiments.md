# ThyroidXL Experiments

Project: Evaluating the Diagnostic and Explainability Impact of Adding TI-RADS Score to Deep Learning for Thyroid Nodule Malignancy Classification

The ThyroidXL dataset is not included in this repository. Users should cite the official ThyroidXL paper and follow the dataset's official access and use terms.

## Experiment A-XL

Image-only EfficientNet-B0 baseline using ultrasound image input only.

| AUC | Accuracy | F1 | Sensitivity | Specificity | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.920900 | 0.834912 | 0.818452 | 0.779037 | 0.886010 | 342 | 44 | 78 | 275 |

## Experiment B1-XL

Hybrid image + nodule metadata model. Inputs are ultrasound image, nodule position, nodule width, nodule height, and nodule depth if available.

| AUC | Accuracy | F1 | Sensitivity | Specificity | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.928709 | 0.848444 | 0.8382 | 0.8215 | 0.8731 | 337 | 49 | 63 | 290 |

## Experiment B2-XL

Hybrid image + final TI-RADS category model. This is image + final TI-RADS category fusion. It should not be described as full TI-RADS descriptor fusion because ThyroidXL provides the final TI-RADS category, not individual ACR TI-RADS descriptors.

| AUC | Accuracy | F1 | Sensitivity | Specificity | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9648 | 0.9053 | 0.9006 | 0.8980 | 0.9119 | 352 | 34 | 36 | 317 |

## Experiment C-XL

Grad-CAM explainability comparing heatmaps with ThyroidXL segmentation masks. The full run evaluates all 2,094 test images and calculates:

- `cam_inside_mask_fraction`
- `mask_covered_by_cam_fraction`
- `iou`

This evaluates whether the model focuses on the true thyroid nodule region.
