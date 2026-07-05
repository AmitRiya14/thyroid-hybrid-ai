from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGE_SIZE


class GradCAM:
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.activations = None
        self.gradients = None
        self.forward_hook = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, tabular_tensor=None):
        self.model.zero_grad(set_to_none=True)
        image_tensor = image_tensor.to(self.device)
        if tabular_tensor is None:
            logits = self.model(image_tensor)
        else:
            logits = self.model(image_tensor, tabular_tensor.to(self.device))
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        logits[0].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam).squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def compute_mask_overlap(cam, mask, threshold: float = 0.5) -> dict:
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    cam_binary = (cam >= threshold).astype(np.uint8)
    intersection = np.logical_and(cam_binary == 1, mask == 1).sum()
    cam_area = cam_binary.sum()
    mask_area = mask.sum()
    union = np.logical_or(cam_binary == 1, mask == 1).sum()
    return {
        "cam_threshold": threshold,
        "cam_inside_mask_fraction": intersection / cam_area if cam_area else 0.0,
        "mask_covered_by_cam_fraction": intersection / mask_area if mask_area else 0.0,
        "iou": intersection / union if union else 0.0,
        "cam_area_pixels": int(cam_area),
        "mask_area_pixels": int(mask_area),
        "intersection_pixels": int(intersection),
    }


def save_overlap_results(records, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    return df


class ThyroidXLCAMDataset(Dataset):
    def __init__(self, dataframe, tabular_features=None, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.tabular_features = tabular_features
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        mask = Image.open(row["mask_path"]).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
        sample = {
            "image": image,
            "mask": (np.asarray(mask) > 0).astype(np.uint8),
            "label": int(row["label"]),
            "patient_id": str(row["patient_id"]),
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
        }
        if self.tabular_features is not None:
            sample["tabular"] = torch.tensor(self.tabular_features[idx], dtype=torch.float32)
        return sample
