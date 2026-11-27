"""Evaluation primitives. All AUROCs use continuous scores and half credit for ties."""
from pathlib import Path

import numpy as np
from PIL import Image


def index_by_id(rows):
    index = {}
    for row in rows:
        key = row["sample_id"]
        if key in index:
            raise ValueError(f"Duplicate sample ID: {key}")
        index[key] = row
    return index


def align(truth, predictions):
    labels = index_by_id(truth)
    scores = index_by_id(predictions)
    if labels.keys() != scores.keys():
        raise ValueError(f"Sample IDs differ, missing={sorted(labels.keys() - scores.keys())[:4]}, "
                         f"extra={sorted(scores.keys() - labels.keys())[:4]}")
    for key, row in labels.items():
        pred = scores[key]
        for field in ("dataset", "category", "split", "label"):
            if field in pred and field in row and pred[field] != row[field]:
                raise ValueError(f"Conflicting {field} for {key}: {pred[field]} vs {row[field]}")
        if not np.isfinite(float(pred["score"])):
            raise ValueError(f"Non-finite image score for {key}")
    return [(row, scores[row["sample_id"]]) for row in truth]


def auc_from_groups(positive, negative):
    """Mann–Whitney AUROC. Negative scores are sorted in place to limit peak RAM."""
    if not len(positive) or not len(negative):
        raise ValueError("AUROC needs both positive and negative observations")
    negative.sort()
    pairs = 0.0
    for start in range(0, len(positive), 250000):
        values = np.sort(positive[start:start + 250000])
        lower = np.searchsorted(negative, values, side="left")
        upper = np.searchsorted(negative, values, side="right")
        pairs += 0.5 * (lower.sum(dtype=np.float64) + upper.sum(dtype=np.float64))
    return pairs / (float(len(positive)) * len(negative))


def image_auc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Expected aligned, finite image-level scores")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Expected binary image labels")
    return auc_from_groups(scores[labels == 1], scores[labels == 0])


def frozen_threshold(validation):
    if not validation or any(t["label"] != 0 or t["split"] != "val" for t, _ in validation):
        raise ValueError("Thresholds require nonempty normal validation data")
    values = np.asarray([p["score"] for _, p in validation], dtype=np.float64)
    return float(np.quantile(values, .99, method="linear"))


def native_map(values, width, height):
    values = np.asarray(values)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError(f"Expected a finite 2-D anomaly map, got {values.shape}")
    values = values.astype(np.float32, copy=False)
    if values.shape == (height, width):
        return values
    return np.asarray(Image.fromarray(values).resize((width, height), Image.Resampling.BILINEAR))


def native_mask(root, row):
    size = (row["width"], row["height"])
    if row["label"] == 0:
        if row.get("mask_path"):
            raise ValueError(f"Normal row has a mask path: {row['sample_id']}")
        return np.zeros(size[::-1], dtype=bool)
    if row["label"] != 1 or not row.get("mask_path"):
        raise ValueError(f"Invalid anomaly label/mask for {row['sample_id']}")
    with Image.open(Path(root) / row["mask_path"]) as source:
        if source.size != size:
            raise ValueError(f"Mask dimensions differ for {row['sample_id']}")
        mask = np.asarray(source.convert("L")) > 0
    if not mask.any() or ("mask_positive_pixels" in row and int(mask.sum()) != row["mask_positive_pixels"]):
        raise ValueError(f"Foreground pixel count differs for {row['sample_id']}")
    return mask


class PixelAUROC:
    """One float32 per pixel for one category; avoids a global argsort index or disk spill."""
    def __init__(self, total_pixels):
        self.values = np.empty(total_pixels, dtype=np.float32)
        self.first_negative = total_pixels
        self.positive_count = 0

    def add(self, scores, mask):
        if scores.shape != mask.shape or mask.dtype != np.bool_:
            raise ValueError("Pixel scores and binary native mask must have identical dimensions")
        positives = int(mask.sum())
        negatives = mask.size - positives
        stop = self.positive_count + positives
        start = self.first_negative - negatives
        if stop > start:
            raise ValueError("More pixels than declared in the manifest")
        if positives:
            self.values[self.positive_count:stop] = scores[mask]
        if negatives:
            self.values[start:self.first_negative] = scores[~mask] if positives else scores.reshape(-1)
        self.positive_count = stop
        self.first_negative = start

    def compute(self):
        if self.positive_count != self.first_negative:
            raise ValueError("Not all manifest pixels were scored")
        return auc_from_groups(self.values[:self.positive_count], self.values[self.first_negative:])
