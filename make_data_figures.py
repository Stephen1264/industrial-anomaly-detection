"""Build the report's example figures from the frozen data manifests."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data_loading import PROJECT_ROOT, load_manifest, project_path


REPORT = PROJECT_ROOT / "report"
INK = "#182c29"
SUBTLE = "#455b57"
OVERLAY = np.array([231, 80, 56], dtype=np.float32)


def open_image(row):
    with Image.open(project_path(PROJECT_ROOT, row["image_path"])) as image:
        return image.convert("RGB")


def open_mask(row):
    with Image.open(project_path(PROJECT_ROOT, row["mask_path"])) as mask:
        return mask.convert("L").point(lambda value: 255 if value else 0)


def mask_overlay(image, mask):
    pixels = np.array(image)
    inside = np.asarray(mask) > 0
    pixels[inside] = np.round(0.54 * pixels[inside] + 0.46 * OVERLAY).astype(np.uint8)
    return pixels


def show(axis, image):
    axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axis.set_axis_off()


def row_label(axis, text):
    axis.text(-0.09, 0.5, text, ha="right", va="center", color=INK, size=10,
              transform=axis.transAxes)


def heading(figure, title, detail):
    figure.text(0.042, 0.957, title, color=INK, size=20, weight="bold", va="top")
    figure.text(0.042, 0.907, detail, color=SUBTLE, size=10, va="top")


def figure_mvtec():
    manifest = load_manifest("mvtec_ad")
    figure, grid = plt.subplots(4, 5, figsize=(12.8, 10.1), facecolor="#f9fbfa")
    figure.subplots_adjust(left=0.135, right=0.97, top=0.835, bottom=0.065,
                           hspace=0.065, wspace=0.075)
    heading(figure, "MVTec AD benchmark examples",
            "Five categories with one official normal image and one annotated anomalous image per category")
    for column, category in enumerate(manifest["categories"]):
        examples = [item for item in manifest["samples"]
                    if item["split"] == "test" and item["category"] == category]
        normal = next(item for item in examples if item["label"] == 0)
        anomalies = sorted((item for item in examples if item["label"]),
                           key=lambda item: item["mask_positive_pixels"])
        anomaly = anomalies[round(0.8 * (len(anomalies) - 1))]
        image, mask = open_image(anomaly), open_mask(anomaly)
        for axis, payload in zip(grid[:, column], (open_image(normal), image, mask, mask_overlay(image, mask))):
            show(axis, payload)
        grid[0, column].set_title(category.replace("_", " ").title(), color=INK,
                                  fontsize=11, fontweight="bold", pad=8)
    for axis, label in zip(grid[:, 0], ("Normal", "Anomalous", "Ground truth", "Mask overlay")):
        row_label(axis, label)
    figure.text(0.042, 0.031, "MVTec AD, Bergmann et al. (2019), CC BY-NC-SA 4.0. Red marks the annotated pixels.",
                color=SUBTLE, size=9)
    figure.savefig(REPORT / "mvtec_examples.png", dpi=165, facecolor=figure.get_facecolor())
    plt.close(figure)


def figure_capture():
    manifest = load_manifest("local_capture")
    figure, grid = plt.subplots(2, 4, figsize=(11.9, 6.25), facecolor="#f9fbfa")
    figure.subplots_adjust(left=0.10, right=0.97, top=0.802, bottom=0.10,
                           hspace=0.13, wspace=0.09)
    heading(figure, "Locally captured bottle and nut examples",
            "Held-out specimens shown before and after a controlled stain")
    headers = ("Before stain", "After stain", "Stain detail", "Mask overlay")
    for axis, title in zip(grid[0], headers):
        slot = axis.get_position(original=True)
        figure.text((slot.x0 + slot.x1) / 2, 0.827, title, color=INK, size=10,
                    weight="bold", ha="center", va="bottom")
    for row, category in enumerate(manifest["categories"]):
        examples = [entry for entry in manifest["samples"] if entry["category"] == category]
        after = max((item for item in examples if item["label"]),
                    key=lambda item: item["mask_positive_pixels"])
        before = next(item for item in examples
                      if item["physical_id"] == after["physical_id"] and item["capture_state"] == "before_stain")
        image, mask = open_image(after), open_mask(after)
        left, top, right, bottom = mask.getbbox()
        padding = max(36, (right - left) // 3, (bottom - top) // 3)
        region = (max(0, left - padding), max(0, top - padding),
                  min(image.width, right + padding), min(image.height, bottom + padding))
        detail = image.crop(region)
        for column, payload in enumerate((open_image(before), image, detail, mask_overlay(detail, mask.crop(region)))):
            show(grid[row, column], payload)
        row_label(grid[row, 0], category.capitalize())
    figure.text(0.042, 0.045, "Red marks the visually traced stain region. Final masks use the full original image dimensions.",
                color=SUBTLE, size=9)
    figure.savefig(REPORT / "local_capture_examples.png", dpi=165, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    REPORT.mkdir(exist_ok=True)
    figure_mvtec()
    figure_capture()
