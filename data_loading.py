"""Read the frozen inspection data splits from this project on any machine."""

import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS = ("mvtec_ad", "local_capture")
SPLITS = ("train", "val", "test")
SCHEMA_VERSION = 1


def project_path(root, relative_path):
    """Resolve a manifest path while keeping it inside the project."""
    relative = PurePosixPath(relative_path)
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in relative_path
        or ":" in relative_path
    ):
        raise ValueError(f"Expected a project-relative path, got {relative_path!r}")
    root = Path(root).resolve()
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes the project: {relative_path!r}")
    return path


def manifest_path(dataset, root=PROJECT_ROOT):
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    return Path(root) / "manifests" / f"{dataset}.json"


def load_manifest(dataset, root=PROJECT_ROOT):
    path = manifest_path(dataset, root)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("dataset") != dataset:
        raise ValueError(f"Incompatible dataset manifest: {path}")
    return manifest


def read_split(dataset, split, category=None, *, root=PROJECT_ROOT):
    """Return metadata for one split; pass category=None to include all classes."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    manifest = load_manifest(dataset, root)
    if category is not None and category not in manifest["categories"]:
        raise ValueError(f"Unknown category {category!r}; choose from {manifest['categories']}")
    samples = [
        sample for sample in manifest["samples"]
        if sample["split"] == split and (category is None or sample["category"] == category)
    ]
    if split in {"train", "val"} and any(item["label"] != 0 or item["mask_path"] for item in samples):
        raise ValueError(f"{dataset}/{split} must only contain normal images")
    return samples


class InspectionDataset:
    """Indexable RGB images and binary test masks for NumPy, PyTorch or Keras."""

    def __init__(self, dataset, category, split, *, image_size=None, root=PROJECT_ROOT):
        self.root = Path(root).resolve()
        self.split = split
        self.samples = read_split(dataset, split, category, root=self.root)
        if image_size is None:
            self.size_wh = None
        else:
            size_hw = (image_size, image_size) if type(image_size) is int else image_size
            if (
                not isinstance(size_hw, (tuple, list))
                or len(size_hw) != 2
                or any(type(value) is not int or value <= 0 for value in size_hw)
            ):
                raise ValueError("image_size must be a positive integer or a (height, width) pair")
            height, width = size_hw
            self.size_wh = (width, height)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        native_size = (row["width"], row["height"])
        with Image.open(project_path(self.root, row["image_path"])) as source:
            if source.size != native_size:
                raise ValueError(f"Image size changed for {row['sample_id']}")
            image = source.convert("RGB")
            if self.size_wh is not None:
                image = image.resize(self.size_wh, Image.Resampling.BILINEAR)
            image = np.array(image, dtype=np.uint8)
        output = {
            "sample_id": row["sample_id"],
            "category": row["category"],
            "label": row["label"],
            "image": image,
        }
        if self.split == "test":
            if row["label"] == 0:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
            else:
                mask_path = row["mask_path"]
                if not mask_path:
                    raise ValueError(f"Missing anomaly mask for {row['sample_id']}")
                with Image.open(project_path(self.root, mask_path)) as source:
                    if source.size != native_size:
                        raise ValueError(f"Image and mask sizes differ for {row['sample_id']}")
                    mask_image = source.convert("L")
                    if self.size_wh is not None:
                        mask_image = mask_image.resize(self.size_wh, Image.Resampling.NEAREST)
                    mask = (np.asarray(mask_image) > 0).astype(np.uint8)
            output["mask"] = mask
        return output
