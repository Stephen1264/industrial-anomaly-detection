# Part D unified results

The full handoff includes the seven-category test comparison, the three PatchCore coreset settings, all autoencoder test exports, paired cases, inference timing, an English course report and D's five presentation slides.

| Test data | Method | Image AUROC | Pixel AUROC | Mean latency, ms/image | Detected defects | False alarms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MVTec AD, five categories | Autoencoder | 0.6658 | 0.6691 | 2.33 | 67/392 | 3/152 |
| MVTec AD, five categories | PatchCore 5% | 0.9457 | 0.9440 | 22.96 | 315/392 | 10/152 |
| Local bottle and nut | Autoencoder | 0.4679 | 0.9082 | 2.62 | 4/24 | 5/26 |
| Local bottle and nut | PatchCore 5% | 0.6955 | 0.9526 | 18.80 | 1/24 | 0/26 |

AUROCs and latency are unweighted category means within the dataset. Alarms use a per-category 99th-percentile normal-validation threshold. Latency is from the shared RTX 4060 Laptop GPU benchmark, batch size one, using 128 × 128 inputs.

[Part D release](https://github.com/Stephen1264/industrial-anomaly-detection/releases/tag/part-d-v1) · [Verified evaluation run](https://github.com/Stephen1264/industrial-anomaly-detection/actions/runs/46005297941) · [Reproduction and demo instructions](HANDOFF_zh.md)
