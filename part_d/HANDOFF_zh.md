# D 部分交接说明

D 使用 A 的两份固定清单、B 的七个卷积自编码器权重，以及 C 的三档 PatchCore 预测和同机测速。完整交付包解压后以 `project/` 为根目录。模型比较、错误案例和展示稿使用同一套结果表。

## 从哪里看

| 路径 | 内容 |
| --- | --- |
| `report/Final_report.pdf`、`report/Final_report.docx` | 全组英文报告，含动机、相关工作、数据、模型、评测、案例和结论 |
| `presentation/Part_D_results.pptx`、`presentation/speaker_notes.md` | 供 A 合并的 D 部分英文展示页和逐页讲稿 |
| `results/per_category.csv`、`results/per_dataset.csv` | 两个数据集的逐类结果、类别宏平均、阈值判定和推理成本 |
| `results/per_sample.csv` | 2,376 条测试预测，含真实标签、分数、冻结阈值、实物 ID、热图位置 |
| `results/protocol_lock.json`、`results/thresholds.csv` | 28 组验证阈值及训练清单、模型、验证预测的校验值 |
| `predictions/ae/` | D 用 B 的正式权重生成的 594 张测试图的连续预测和 128×128 异常图 |
| `timing/latency_raw.csv`、`timing/benchmark_report.json` | C 在 RTX 4060 上记录的 840 次原始计时、计时边界与环境 |
| `figures/`、`results/case_index.csv` | 并排案例图及每个示例的样本 ID、掩码面积、两组分数和判定 |
| `demo/artifacts/`、`results/demo_verification.json` | 给 B 的七个正式阈值及网页样例与测试脚本的对照 |
| `part_d/` | 数据获取、阈值冻结、正式评测、案例绘制及交付生成代码 |

`per_sample.csv` 中以 `C:project/` 开头的热图来自 C 的原始交接包。以 `D:project/` 开头的热图在本次交付包内。图像分数越高表示越异常。判定使用严格大于阈值，AUROC 使用完整连续分数。推理延迟以毫秒/张计，各类别均权。特征向量存储量按字节记录，只统计 PatchCore 特征库。

## 在本机复跑

建议使用 Python 3.11。以下命令在已解压的 `project` 目录执行。准备数据脚本会校验 A、B、C 三个原始发布包的 SHA-256，也可先把已下载的 ZIP 放入 `inputs` 目录复用。请给复跑输出使用新目录。

```bash
python -m pip install -r part_d/requirements.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
python part_d/prepare_inputs.py inputs
python -m unittest discover -s part_d -p test_metrics.py
python part_d/evaluate.py lock --inputs inputs --out rerun
python part_d/evaluate.py test --inputs inputs --out rerun
python part_d/cases.py --inputs inputs --out rerun
```

先生成 28 组阈值和校验值，再读取正式测试预测。公开数据五类、自采两类分别训练、设阈值和求均值。自采的同一实物留在一个划分中。脚本比较 A/B 清单的完整解析内容，并分别核验训练时使用的原始文件哈希。C 的全部 21 组独立重算结果与随包诊断值逐一核对。

如需从头重新训练或建特征库，分别查阅 B、C 发布包内的原始说明。这里复现的是使用已发布模型和预测进行的正式评测。C 的原始 GPU 计时由 D 核验每轮输入、权重和汇总统计。

## 交给 B 的网页 Demo

保留原 B 目录，另建一个含正式阈值的模型目录。以下从 D 的 `project` 目录执行。

```bash
python -m pip install streamlit==1.52.2
python part_d/prepare_demo.py --b-project inputs/B/project --evaluation . --destination demo_models
python -m streamlit run inputs/B/project/demo.py -- --models demo_models
```

网页沿用 B 的类别选择、上传和热图显示。选 `mvtec_ad / bottle` 可以试用原有的正常与缺陷样例，页面阈值状态为 `locked_by_D`。原缺陷示例的判定和分数可在 `results/demo_verification.json` 查到。

## 来源

[A 数据交接](https://github.com/Stephen1264/industrial-anomaly-detection/releases/tag/part-a-v1) · [B 卷积自编码器](https://github.com/Stephen1264/industrial-anomaly-detection/releases/tag/part-b-v1) · [C PatchCore](https://github.com/Stephen1264/industrial-anomaly-detection/releases/tag/part-c-v1)。MVTec 示例图像来自 MVTec Software GmbH，按 CC BY-NC-SA 4.0 使用。自采照片及污渍掩码来自课程项目小组。
