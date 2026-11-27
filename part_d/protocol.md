# Part D evaluation protocol

The inputs are the three releases pinned in `inputs.json`. Part A's two manifests define the sample universe. The official MVTec AD test set contains 544 images and the independent local test set contains 50 images. A physical item is the split unit for the local data.

## Choices fixed for the D comparison

* Use all seven released autoencoder checkpoints and Part C's designated 5% PatchCore configuration. Run the 1% and 10% banks as the prespecified sensitivity comparison. Keep the two datasets and their category models separate.
* For every dataset, category and model configuration, use the 99th percentile of its continuous scores on normal validation images as the image decision threshold. Use linear quantile interpolation. Save the numeric thresholds and input fingerprints before the D test evaluation. C's published release already contained its own test results.
* Join by sample ID and validate dataset, category, split and label. Reject duplicates, missing samples, non-finite scores and malformed anomaly maps. Apply the model's continuous image score directly. Upsample its continuous anomaly map to the native image dimensions with bilinear interpolation. A normal image has an all-zero ground-truth mask. An anomalous image uses its original binary mask.
* Compute image AUROC within each category. For pixel AUROC, pool all native-resolution pixels within that category, with half credit for equal positive and negative scores. Report unweighted means of the five MVTec or two local category AUROCs. Do not mix scores from separately fitted models across categories. Localization uses continuous maps and requires no pixel decision threshold.
* Use the paired autoencoder/PatchCore latency measurements taken by Part C under the same hardware and protocol. Record input dimensions, batch size, warm-up, repetitions, timed operations and units from its timing logs. Keep costs from different protocols in separate comparisons.
* Use the fixed thresholds for error counts and case review. Include normal, obvious defect, small defect and false-positive cases. Identify each by dataset, category and sample ID; read these examples only after the numerical comparison.
