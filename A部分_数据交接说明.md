# A 部分数据交接

接手同学复制整个 `project` 文件夹，就可以在本地读取两批数据。加载器根据代码位置定位项目，清单中的图片和掩码路径都使用项目内的相对路径。

| 数据源 | 类别 | 正常训练 | 正常验证 | 正常测试 | 异常测试 | 异常掩码 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MVTec AD | `bottle`、`cable`、`metal_nut`、`tile`、`wood` | 904 | 226 | 152 | 392 | 392 |
| 小组自采 | `bottle`、`nut` | 80 | 30 | 26 | 24 | 24 |

## 接手时检查

在 `project` 目录运行以下命令。使用 Python 3.10 或更新版本。

```powershell
python -m pip install -r requirements.txt
python -B prepare_data.py mvtec_ad --check
python -B prepare_data.py local_capture --check
```

检查会解码原图，核对掩码尺寸和二值像素，验证划分与文件校验值是否仍和交付清单一致。MVTec AD 还会核对论文公布的原训练图、正常测试图和异常测试图数量。

## 模型组怎么读

```python
from data_loading import InspectionDataset, read_split

# B、C 从同一类别的正常图训练和选配置
train = InspectionDataset("mvtec_ad", "bottle", "train", image_size=256)
val = InspectionDataset("mvtec_ad", "bottle", "val", image_size=256)
print(len(train), len(val), train[0]["image"].shape)  # 167 42 (256, 256, 3)

# D 在配置与阈值固定后读取官方测试图和原尺寸掩码
test = InspectionDataset("mvtec_ad", "bottle", "test")
sample = test[0]
print(sample["sample_id"], sample["label"], sample["mask"].shape)

# 自采实验用自采的正常训练、验证和测试数据
local_train = InspectionDataset("local_capture", "nut", "train", image_size=256)
local_test_rows = read_split("local_capture", "test", "nut")
```

- `image` 为 RGB、`uint8`、`H × W × 3`。模型各自应用所需的归一化，`image_size` 可以传正方形边长或 `(height, width)`。
- `label` 为 `0` 正常、`1` 异常。测试样本额外返回 `H × W` 的 `uint8` 掩码，取值为 0、1。正常测试图的掩码全零。
- `read_split` 返回样本 ID、类别、原始路径、尺寸、标签和掩码路径等字段。B、C 导出预测时保留 `sample_id`，D 可用它连接异常分数、异常图与真值。
- 像素级评测使用原尺寸掩码。若模型输出的是较小的连续异常图，由评测端将异常图对齐到原尺寸。加载器显式缩放时对图片用双线性插值，对掩码用最近邻。

## 固定划分

**MVTec AD** 每类从官方 `train/good` 里按种子 42 的 SHA-256 排序留出约 20% 作正常验证。官方 `test` 和 `ground_truth` 对应关系完整保留。B、C 都通过清单取训练集，以免把留出的验证图也读进训练。

**小组自采** 使用组员确认的 160 张实拍图，每类 80 张。按 `类别:实物ID` 检查训练、验证、测试三部分互斥。每类有 8 个训练、3 个验证、5 个测试实物。原表把污渍前后照片放在同一个测试子目录，标签按表内的画面状态确定。6 张污渍前的照片在测试集内标为正常，24 张污渍后照片标为异常。可见污渍按原图像素勾画，颜色渐淡处的边界为近似标注。

## 文件索引

| 文件或目录 | 用途 |
| --- | --- |
| `data/mvtec_ad/` | 五类公开数据的原始 PNG 和来源记录 |
| `bottle_nut_reference/` | 组员提供的原始 JPG 和采集清单 |
| `data/local_masks/` | 自采异常的 24 张原尺寸二值 PNG |
| `annotations/local_stain_polygons.json` | 自采污渍轮廓的原图坐标及图像校验值 |
| `manifests/` | 两批数据的正式划分、样本元数据、计数和 SHA-256 |
| `data_loading.py` | 共用读取入口 |
| `prepare_data.py` | 数据核验和清单重建入口 |
| `report/` | 可交给报告统稿人和展示负责人的数据段落、统计表和配图 |

如需从原始文件重建，运行 `python -B prepare_data.py mvtec_ad` 和 `python -B prepare_data.py local_capture`。第二条也会从轮廓重建自采掩码。配图可在安装 Matplotlib 后用 `python -B make_data_figures.py` 重出。

## MVTec AD 来源

原数据由 MVTec Software GmbH 发布，许可为 CC BY-NC-SA 4.0。随项目提供的文件从 Kaggle 上的公开数据镜像获取。五类共 2,066 个 PNG 均通过 ZIP 校验，并和另一份固定版本的公开镜像逐文件核对了字节校验值。论文和原发布页可用于报告引用，镜像及校验详情见 `data/mvtec_ad/SOURCE.json`。 [MVTec AD 发布页](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) · [Bergmann 等的论文](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_Detection_paper.html)
