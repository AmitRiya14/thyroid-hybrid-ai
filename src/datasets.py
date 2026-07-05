from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGE_SIZE, RANDOM_SEED, STANFORD_FRAMES_PER_PATIENT, STANFORD_TIRADS_COLUMNS


def image_to_pil(image) -> Image.Image:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        image = image - image.min()
        if image.max() > 0:
            image = image / image.max()
        image = (image * 255).astype(np.uint8)
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    if image.ndim == 3 and image.shape[-1] == 1:
        return Image.fromarray(image.squeeze()).convert("RGB")
    return Image.fromarray(image).convert("RGB")


def image_transforms(train: bool = False, image_size: int = IMAGE_SIZE):
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.extend([transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)])
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)


def patient_split(patient_ids, labels, seed: int = RANDOM_SEED, val_size: float = 0.15, test_size: float = 0.15):
    patient_df = pd.DataFrame({"patient_id": patient_ids, "label": labels}).groupby("patient_id").first().reset_index()
    temp_size = val_size + test_size
    train_patients, temp_patients = train_test_split(
        patient_df, test_size=temp_size, random_state=seed, stratify=patient_df["label"]
    )
    val_fraction_of_temp = val_size / temp_size
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=1 - val_fraction_of_temp,
        random_state=seed,
        stratify=temp_patients["label"],
    )
    return set(train_patients.patient_id), set(val_patients.patient_id), set(test_patients.patient_id)


def masks_from_patient_ids(patient_ids, train_ids, val_ids, test_ids):
    patient_ids = np.asarray(patient_ids)
    return (
        np.array([pid in train_ids for pid in patient_ids]),
        np.array([pid in val_ids for pid in patient_ids]),
        np.array([pid in test_ids for pid in patient_ids]),
    )


def _decode_array(values):
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def select_representative_frames(index_df: pd.DataFrame, frames_per_patient: int = STANFORD_FRAMES_PER_PATIENT):
    def pick(group):
        group = group.sort_values("array_idx").reset_index(drop=True)
        positions = np.linspace(0.20, 0.80, frames_per_patient)
        chosen = np.unique(np.round(positions * (len(group) - 1)).astype(int))
        selected = group.iloc[chosen].copy()
        selected["selected_frame_number"] = range(len(selected))
        selected["total_frames_in_clip"] = len(group)
        return selected

    return index_df.groupby("annot_id", group_keys=False).apply(pick).reset_index(drop=True)


def load_stanford_dataframe(data_dir, frames_per_patient: int = STANFORD_FRAMES_PER_PATIENT, image_key: str = "image"):
    data_dir = Path(data_dir)
    metadata = pd.read_csv(data_dir / "metadata.csv")
    hdf5_path = data_dir / "dataset.hdf5"
    with h5py.File(hdf5_path, "r") as f:
        if image_key not in f:
            raise KeyError(f"Expected image dataset key '{image_key}' in {hdf5_path}. Found: {list(f.keys())}")
        if "annot_id" not in f:
            raise KeyError(f"Expected 'annot_id' key in {hdf5_path}. Found: {list(f.keys())}")
        annot_ids = _decode_array(f["annot_id"][:])
        frame_num = f["frame_num"][:] if "frame_num" in f else np.arange(len(annot_ids))
        index_df = pd.DataFrame({"annot_id": annot_ids, "array_idx": np.arange(len(annot_ids)), "frame_num": frame_num})
        selected = select_representative_frames(index_df, frames_per_patient)
        array_idx = selected["array_idx"].to_numpy()
        sorted_order = np.argsort(array_idx)
        images_sorted = f[image_key][array_idx[sorted_order]]
        images = images_sorted[np.argsort(sorted_order)]
        masks = None
        if "mask" in f:
            masks_sorted = f["mask"][array_idx[sorted_order]]
            masks = masks_sorted[np.argsort(sorted_order)]
    df = selected.merge(metadata, on="annot_id", how="left")
    if df["histopath_diagnosis"].isna().any():
        raise ValueError("Some selected Stanford frames did not match metadata annot_id rows.")
    df["label"] = df["histopath_diagnosis"].astype(int)
    df["patient_id"] = df["annot_id"].astype(str)
    return df, images, masks


def encode_stanford_tirads(meta_df: pd.DataFrame, frame_df: pd.DataFrame, train_mask):
    missing = [c for c in STANFORD_TIRADS_COLUMNS if c not in meta_df.columns]
    if missing:
        raise ValueError(f"Missing Stanford TI-RADS columns: {missing}")
    tirads = pd.get_dummies(meta_df[["annot_id"] + STANFORD_TIRADS_COLUMNS].set_index("annot_id").astype(str), dummy_na=True)
    features = frame_df[["annot_id"]].join(tirads, on="annot_id")
    if features.isna().any().any():
        raise ValueError("Missing TI-RADS features after merging on annot_id.")
    feature_cols = [c for c in features.columns if c != "annot_id"]
    matrix = features[feature_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    matrix[train_mask] = scaler.fit_transform(matrix[train_mask])
    matrix[~train_mask] = scaler.transform(matrix[~train_mask])
    return matrix.astype(np.float32), feature_cols, scaler


class ImageOnlyDataset(Dataset):
    def __init__(self, images, labels, patient_ids, transform=None, source_rows=None):
        self.images = images
        self.labels = np.asarray(labels).astype(np.float32)
        self.patient_ids = np.asarray(patient_ids).astype(str)
        self.transform = transform
        self.source_rows = list(source_rows) if source_rows is not None else list(range(len(self.labels)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = image_to_pil(self.images[idx])
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.labels[idx]), self.patient_ids[idx], self.source_rows[idx]


class HybridDataset(ImageOnlyDataset):
    def __init__(self, images, tabular, labels, patient_ids, transform=None, source_rows=None):
        super().__init__(images, labels, patient_ids, transform, source_rows)
        self.tabular = np.asarray(tabular, dtype=np.float32)

    def __getitem__(self, idx):
        image, label, patient_id, source_row = super().__getitem__(idx)
        return image, torch.tensor(self.tabular[idx]), label, patient_id, source_row


def thyroidxl_label(conclusion_numeric):
    if conclusion_numeric in [1, 2]:
        return 1
    if conclusion_numeric in [3, 4]:
        return 0
    return None


def load_thyroidxl_annotations(data_dir, split: str) -> pd.DataFrame:
    data_dir = Path(data_dir)
    with (data_dir / split / f"{split}_annotations.json").open("r", encoding="utf-8") as f:
        annotations = json.load(f)
    rows = []
    images_root = data_dir / split / "images"
    for patient_id, patient_info in annotations["info"].items():
        nodule = patient_info.get("nodule_1") or {}
        label = thyroidxl_label(patient_info.get("conclusion"))
        for image_name in patient_info.get("images", []):
            rows.append(
                {
                    "patient_id": str(patient_id).zfill(8),
                    "image_name": image_name,
                    "image_path": str(images_root / image_name),
                    "mask_path": str(data_dir / split / "masks" / image_name),
                    "label": label,
                    "split": split,
                    "conclusion_numeric": patient_info.get("conclusion"),
                    "nodule_position": nodule.get("position"),
                    "nodule_width": nodule.get("width"),
                    "nodule_height": nodule.get("height"),
                    "nodule_depth": nodule.get("depth"),
                    "final_tirads": nodule.get("TIRADS"),
                }
            )
    df = pd.DataFrame(rows)
    return df[df["label"].notna()].copy()


def load_thyroidxl_splits(data_dir, seed: int = RANDOM_SEED, val_size: float = 0.15):
    train_full = load_thyroidxl_annotations(data_dir, "train")
    test_df = load_thyroidxl_annotations(data_dir, "test")
    patient_df = train_full.groupby("patient_id")["label"].first().reset_index()
    train_patients, val_patients = train_test_split(
        patient_df, test_size=val_size, random_state=seed, stratify=patient_df["label"]
    )
    train_ids = set(train_patients["patient_id"])
    val_ids = set(val_patients["patient_id"])
    train_df = train_full[train_full["patient_id"].isin(train_ids)].reset_index(drop=True)
    val_df = train_full[train_full["patient_id"].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df, test_df.reset_index(drop=True)


class ThyroidXLImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(float(row["label"])), str(row["patient_id"]), idx


class ThyroidXLHybridDataset(ThyroidXLImageDataset):
    def __init__(self, dataframe: pd.DataFrame, tabular_features, transform=None):
        super().__init__(dataframe, transform)
        self.tabular_features = np.asarray(tabular_features, dtype=np.float32)

    def __getitem__(self, idx):
        image, label, patient_id, source_row = super().__getitem__(idx)
        return image, torch.tensor(self.tabular_features[idx]), label, patient_id, source_row


def build_tabular_preprocessor(train_df, numeric_cols, categorical_cols):
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    transformer = ColumnTransformer(
        [("num", numeric, numeric_cols), ("cat", categorical, categorical_cols)],
        remainder="drop",
    )
    return transformer.fit(train_df[numeric_cols + categorical_cols])


def prepare_final_tirads(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["final_tirads"] = df["final_tirads"].astype("Int64").astype(str).replace("<NA>", "missing")
    return df
