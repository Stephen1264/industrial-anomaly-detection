D 已完成全组的统一评测、案例分析和英文报告。下载 **project-part-d-v1.zip** 并解压。入口是 `project/README.md`。英文报告和 D 的五页展示稿也可在下方单独下载，展示稿含逐页英文讲稿。

| 数据集 | 模型 | 图像 AUROC | 像素 AUROC | ms/张 | 检出缺陷 | 误报 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MVTec AD 五类 | 自编码器 | 0.6658 | 0.6691 | 2.33 | 67/392 | 3/152 |
| MVTec AD 五类 | PatchCore 5% | 0.9457 | 0.9440 | 22.96 | 315/392 | 10/152 |
| 自采瓶子与螺母 | 自编码器 | 0.4679 | 0.9082 | 2.62 | 4/24 | 5/26 |
| 自采瓶子与螺母 | PatchCore 5% | 0.6955 | 0.9526 | 18.80 | 1/24 | 0/26 |

AUROC 与延迟分别在每个数据集内按类别宏平均。延迟来自 C 在同一 RTX 4060 Laptop GPU 上的 840 次原始计时，单位为毫秒/张。阈值取各类别正常验证分数的第 99 百分位。AUROC 使用连续分数，像素评测使用原尺寸掩码。

- 594 张独立测试划分中的图像，四种配置产生 2,376 条逐样本预测。D 导出了 B 的 594 张正式测试异常图，独立复算并核对了 C 的 21 组结果。
- 1%、5%、10% 三档 coreset 均有逐类结果、成本和资源记录。每个数据集各含正常、明显缺陷、小缺陷和误报案例。
- 交付包包含 D 的七个 Demo 冻结阈值，以及网页与推理脚本对固定正常、缺陷样例的核验结果。
- A 的原始数据和 C 的异常图按原发布包引用。所有来源包、模型、阈值及正式输出可通过交付文件中的 SHA-256 记录追溯。

[正式评测运行记录](https://github.com/Stephen1264/industrial-anomaly-detection/actions/runs/46005297941) · [A 数据交接](https://github.com/Stephen1264/industrial-anomaly-detection/releases/tag/part-a-v1) · [B 自编码器](https://github.com/Stephen1264/industrial-anomaly-detection/releases/tag/part-b-v1) · [C PatchCore](https://github.com/Stephen1264/industrial-anomaly-detection/releases/tag/part-c-v1)
