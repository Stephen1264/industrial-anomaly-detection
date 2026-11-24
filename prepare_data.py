"""Validate the original data and build the handoff manifests and local masks."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image, ImageDraw

from data_loading import DATASETS, PROJECT_ROOT, SCHEMA_VERSION, manifest_path, project_path


# Official MVTec AD CVPR 2019 paper, Table 1: train, normal test, anomalous test, side.
MVTEC_COUNTS = {
    "bottle": (209, 20, 63, 900),
    "cable": (224, 58, 92, 1024),
    "metal_nut": (220, 22, 93, 700),
    "tile": (230, 33, 84, 840),
    "wood": (247, 19, 60, 1024),
}
VALIDATION_FRACTION = 0.20
SPLIT_SEED = 42
CAPTURE_ROOT = "bottle_nut_reference"
CAPTURE_CATEGORIES = ("bottle", "nut")
CAPTURE_INDEX = f"{CAPTURE_ROOT}/capture_index.csv"
ANNOTATIONS = "annotations/local_stain_polygons.json"
CAPTURE_STATES = {
    ("train_normal", "正常"): ("train", 0, "normal"),
    ("val_normal", "正常"): ("val", 0, "normal"),
    ("test_normal", "正常"): ("test", 0, "normal"),
    ("test_defect", "污渍前"): ("test", 0, "before_stain"),
    ("test_defect", "污渍后"): ("test", 1, "after_stain"),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_record(root, dataset, category, path, split, label, defect_type):
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        mode = image.mode
    image_path = path.relative_to(root).as_posix()
    native_root = "data/mvtec_ad" if dataset == "mvtec_ad" else CAPTURE_ROOT
    source_id = path.relative_to(root / native_root).with_suffix("").as_posix()
    return {
        "sample_id": f"{dataset}/{source_id}",
        "category": category,
        "split": split,
        "label": label,
        "defect_type": defect_type,
        "image_path": image_path,
        "image_sha256": sha256(path),
        "width": width,
        "height": height,
        "mode": mode,
        "mask_path": None,
    }


def attach_mask(record, path, root):
    with Image.open(path) as source:
        if source.size != (record["width"], record["height"]):
            raise ValueError(f"Mask size mismatch: {path}")
        pixels = np.asarray(source.convert("L"))
        low, high = int(pixels.min()), int(pixels.max())
        if low != 0 or high not in (1, 255) or np.any((pixels != 0) & (pixels != high)):
            raise ValueError(f"Expected a binary mask with positive and background pixels: {path}")
    record["mask_path"] = path.relative_to(root).as_posix()
    record["mask_sha256"] = sha256(path)
    record["mask_positive_pixels"] = int(np.count_nonzero(pixels))


def validation_images(category, training):
    def rank(path):
        return hashlib.sha256(f"{SPLIT_SEED}:{category}/{path.name}".encode("utf-8")).digest()

    count = round(len(training) * VALIDATION_FRACTION)
    return set(sorted(training, key=rank)[:count])


def scan_mvtec(root):
    records = []
    for category, (train_count, normal_count, anomaly_count, side) in MVTEC_COUNTS.items():
        folder = root / "data" / "mvtec_ad" / category
        training = sorted(folder.glob("train/good/*.png"))
        testing = sorted(folder.glob("test/*/*.png"))
        official_masks = set(folder.glob("ground_truth/*/*.png"))
        normal_tests = sum(path.parent.name == "good" for path in testing)
        if (
            (len(training), normal_tests, len(testing) - normal_tests, len(official_masks))
            != (train_count, normal_count, anomaly_count, anomaly_count)
            or set(training) != set(folder.glob("train/*/*.png"))
        ):
            raise ValueError(f"{category}: data inventory differs from the official published counts")
        val_images = validation_images(category, training)
        for path in training + testing:
            raw_split = path.parent.parent.name
            split = "val" if path in val_images else raw_split
            defect = path.parent.name
            label = int(defect != "good")
            record = image_record(root, "mvtec_ad", category, path, split, label, defect)
            record["source_split"] = raw_split
            if (record["width"], record["height"]) != (side, side):
                raise ValueError(f"Image size differs from the published size: {path}")
            if label:
                mask = folder / "ground_truth" / defect / f"{path.stem}_mask.png"
                if mask not in official_masks:
                    raise ValueError(f"Missing or repeated mask: {mask}")
                official_masks.remove(mask)
                attach_mask(record, mask, root)
            records.append(record)
        if official_masks:
            raise ValueError(f"Unmatched official masks for {category}: {len(official_masks)}")
    return records


def annotation_mask(record, annotation):
    if annotation["image_sha256"] != record["image_sha256"]:
        raise ValueError(f"Annotation was made for a different image: {record['image_path']}")
    width, height = record["width"], record["height"]
    mask = Image.new("L", (width, height))
    draw = ImageDraw.Draw(mask)
    polygons = annotation["polygons"]
    if not polygons:
        raise ValueError(f"No positive region annotated: {record['image_path']}")
    for polygon in polygons:
        points = [tuple(point) for point in polygon]
        if len(points) < 3 or any(not (0 <= x < width and 0 <= y < height) for x, y in points):
            raise ValueError(f"Invalid polygon for {record['image_path']}")
        draw.polygon(points, fill=255)
    return mask


def store_mask(mask, path, check):
    if path.is_file():
        try:
            with Image.open(path) as existing:
                if existing.mode == "L" and np.array_equal(np.asarray(existing), np.asarray(mask)):
                    return
        except OSError:
            if check:
                raise
    if check:
        raise ValueError(f"Mask is missing or differs from the checked-in polygon: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        mask.save(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def scan_capture(root, check):
    path = root / CAPTURE_INDEX
    with path.open(encoding="utf-8-sig", newline="") as stream:
        index_rows = list(csv.DictReader(stream))
    annotations = json.loads((root / ANNOTATIONS).read_text(encoding="utf-8"))["images"]
    remaining_annotations = set(annotations)
    image_paths = {
        path.relative_to(root).as_posix() for path in (root / CAPTURE_ROOT).rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    }
    records = []
    image_splits = {}
    object_splits = {}
    for row in index_rows:
        category, object_id, photo_id = row["类别"], row["实物ID"], row["照片序号"]
        source_split, source_state = row["划分"], row["画面状态"]
        source_path = f"{CAPTURE_ROOT}/{row['相对路径']}"
        if source_path not in image_paths:
            raise ValueError(f"Capture index refers to a missing or repeated photo: {source_path}")
        image_paths.remove(source_path)
        path = project_path(root, source_path)
        expected_name = f"{category}_{object_id}_{photo_id}"
        if category not in CAPTURE_CATEGORIES or path.stem != expected_name:
            raise ValueError(f"Photo name disagrees with its category or specimen ID: {source_path}")
        if row["相对路径"] != f"{category}/{source_split}/{row['文件名']}" or path.name != row["文件名"]:
            raise ValueError(f"Photo path disagrees with capture index: {source_path}")
        split, label, state = CAPTURE_STATES[(source_split, source_state)]
        physical_id = f"{category}:{object_id}"
        if object_splits.setdefault(physical_id, split) != split:
            raise ValueError(f"Physical specimen assigned to multiple splits: {physical_id}")
        defect_type = "controlled_stain" if label else "good"
        record = image_record(root, "local_capture", category, path, split, label, defect_type)
        record.update(physical_id=physical_id, photo_id=photo_id, capture_state=state, source_split=source_split)
        checksum = record["image_sha256"]
        if image_splits.setdefault(checksum, (split, physical_id)) != (split, physical_id):
            raise ValueError(f"Identical photos have different splits or specimen IDs: {source_path}")
        if label:
            if source_path not in remaining_annotations:
                raise ValueError(f"Missing stain annotation: {source_path}")
            remaining_annotations.remove(source_path)
            mask = annotation_mask(record, annotations[source_path])
            mask_path = root / "data" / "local_masks" / category / f"{path.stem}_mask.png"
            store_mask(mask, mask_path, check)
            attach_mask(record, mask_path, root)
        records.append(record)
    if image_paths or remaining_annotations:
        raise ValueError(f"Unmatched photos: {len(image_paths)}; unmatched annotations: {len(remaining_annotations)}")
    return records


def summarize(records, categories):
    summary = {}
    for category in categories:
        rows = [row for row in records if row["category"] == category]
        counts = Counter((row["split"], row["label"]) for row in rows)
        for required in (("train", 0), ("val", 0), ("test", 0), ("test", 1)):
            if counts[required] == 0:
                raise ValueError(f"{category} lacks samples for {required}")
        if counts[("train", 1)] or counts[("val", 1)]:
            raise ValueError(f"Anomalous images found in {category}'s train or validation split")
        summary[category] = {
            "train_normal": counts[("train", 0)],
            "val_normal": counts[("val", 0)],
            "test_normal": counts[("test", 0)],
            "test_anomaly": counts[("test", 1)],
            "masks": sum(bool(row["mask_path"]) for row in rows),
            "images": len(rows),
        }
        objects = defaultdict(set)
        for row in rows:
            if "physical_id" in row:
                objects[row["split"]].add(row["physical_id"])
        if objects:
            summary[category]["physical_ids_by_split"] = {split: len(ids) for split, ids in sorted(objects.items())}
    return summary


def build_manifest(root, dataset, check):
    if dataset == "mvtec_ad":
        records = scan_mvtec(root)
        categories = list(MVTEC_COUNTS)
        policy = {
            "official_test_preserved": True,
            "validation_fraction": VALIDATION_FRACTION,
            "seed": SPLIT_SEED,
            "selection": "Lowest SHA-256 ranks of seed:category/filename within each official train/good folder.",
        }
    else:
        records = scan_capture(root, check)
        categories = list(CAPTURE_CATEGORIES)
        policy = {
            "source": CAPTURE_INDEX,
            "annotations": ANNOTATIONS,
            "label_rule": "Before-stain photos stay in test and are normal; after-stain photos are anomalous.",
            "specimen_ids_disjoint": True,
        }
    if len({row["sample_id"] for row in records}) != len(records):
        raise ValueError(f"Nonunique sample IDs in {dataset}")
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "categories": categories,
        "split_policy": policy,
        "summary": summarize(records, categories),
        "samples": sorted(records, key=lambda row: row["sample_id"]),
    }


def write_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, suffix=".json", delete=False) as temp:
        temporary = Path(temp.name)
        json.dump(manifest, temp, indent=2, ensure_ascii=False)
        temp.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--check", action="store_true", help="Check the originals, masks and manifest without writing")
    options = parser.parse_args()
    root = options.root.resolve()
    path = manifest_path(options.dataset, root)
    manifest = build_manifest(root, options.dataset, options.check)
    if options.check:
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise SystemExit(f"Manifest differs from validated files: {path}")
        print(f"Verified {path}")
    else:
        write_manifest(path, manifest)
        print(f"Wrote {path}")
    for category, counts in manifest["summary"].items():
        print(category, json.dumps(counts))


if __name__ == "__main__":
    main()
