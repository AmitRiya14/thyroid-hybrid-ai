# Methodology Summary

This project compares image-only deep learning models with hybrid models that combine thyroid ultrasound images with structured nodule information.

Core methodological choices:

- EfficientNet-B0 is used as the image backbone.
- Binary output is benign vs malignant.
- Training uses `BCEWithLogitsLoss`.
- Class imbalance is handled through positive-class weighting from the training split.
- Splits are patient-level to reduce leakage.
- Stanford patient-level prediction averages the 5 selected frame probabilities.
- ThyroidXL patient-level prediction averages probabilities across all available patient images.
- Hybrid models concatenate image features with a small tabular branch.
- Explainability uses Grad-CAM heatmaps and segmentation mask overlap metrics.

Public repository cleanup:

- Removed Colab installs, Google Drive mounts, hardcoded `/content/drive/...` paths, exploratory print/debug cells, failed recovery attempts, duplicate training loops, duplicate evaluation functions, and duplicate Grad-CAM runs.
- Preserved core loading, splitting, EfficientNet models, hybrid tabular fusion, class weighting, patient-level aggregation, CSV result saving, Grad-CAM, and mask-overlap logic.
