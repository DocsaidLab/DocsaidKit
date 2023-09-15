# PyTorch

本模塊主要包含與 PyTorch 有關的功能，例如神經網絡結構、優化器等相關功能。

## 目錄
- [Backbone](#backbone)
    - [主要功能](#主要功能)
    - [使用示例](#使用示例)
    - [主要結構和組件](#主要結構和組件)
- [EfficientDet](#efficientdet)
    - [主要功能](#主要功能-1)
    - [主要結構和組件](#主要結構和組件-1)
    - [使用示例](#使用示例-1)
- [Neck](#neck)
    - [主要功能](#主要功能-2)
    - [主要結構和組件](#主要結構和組件-2)
    - [使用示例](#使用示例-2)
- [NN (Neural Networks)](#nn-neural-networks)
- [Optim](#optim)
- [Transformers](#transformers)
- [Utils](#utils)
    - [主要功能](#主要功能-3)
    - [主要結構和組件](#主要結構和組件-3)

---

## [Backbone](./bacobone/)

- **目的**:
  此模組的主要目的是提供基礎的網絡結構，如：ResNet, VGG 等，並透過 `timm` 庫來支持多種預訓練的模型。

- **文件**:
  - `__init__.py`

### 主要功能

1. **模型註冊**: 使用 `timm` 庫的 `list_models` 和 `create_model` 函數來列舉並創建多種支持的模型結構。這些模型將被註冊到 `BACKBONE` 字典中。

2. **建立 Backbone**:
   - `build_backbone` 函數允許使用者根據給定的名稱建立一個 Backbone 模型。
   - 如果名稱不在支持的模型列表中，它會引發一個錯誤。
   - 使用 `BACKBONE` 字典中的相應函數來實例化模型。

3. **列出支持的 Backbones**:
   - `list_backbones` 函數可以返回當前支持的所有模型的列表。
   - 通過傳遞一個過濾器參數 `filter`，可以使用 Unix shell-style 的通配符來過濾和搜尋特定的模型。

### 使用示例

1. **建立 Backbone**:
   ```python
   model = build_backbone("resnet50")
   ```

2. **列出所有支持的 Backbone**:
   ```python
   all_backbones = list_backbones()
   print(all_backbones)
   ```

3. **搜尋特定的 Backbone**:
   ```python
   resnet_models = list_backbones("resnet*")
   print(resnet_models)
   ```

---

## [EfficientDet](./efficientdet/)

- **目的**:
  此模組是為了實現 EfficientDet 物件檢測模型，一種現代的，高效的目標檢測結構。

- **文件**:
  - `efficientdet.py`

### 主要功能

#### 1. **EfficientDet 定義**:

- **類**: `EfficientDet`

- **參數**:
   - `compound_coef`: 組合縮放係數，決定模型的大小和復雜性。
   - `pretrained`: 是否需要使用 ImageNet 上預訓練的模型。

- **特點**:
   1. 使用組合縮放 (Compound scaling) 策略，這意味著模型的深度、寬度和解析度都可以按照給定的 `compound_coef` 進行調整。
   2. 依賴 `timm` 庫來建立 `efficientnet` 作為其 Backbone。
   3. 使用 BiFPNs (Bidirectional Feature Pyramids) 作為其 Neck 部分，以融合不同尺度的特徵。BiFPNs 是 EfficientDet 架構的核心部分，用於進行特徵的上下採樣。

### 主要結構和組件

- **Backbone (`self.backbone`)**:
  這部分是由 `efficientnet` 組成的，其深度由 `compound_coef` 決定。它的目的是從輸入圖像中提取基本的特徵。

- **BiFPNs (`self.bifpn`)**:
  這是 EfficientDet 的一個重要部分，它是一種特徵金字塔結構，可以融合不同層次的特徵。此結構可以進行多次重複（取決於 `compound_coef`），且每次重複都可以進行特徵的上下採樣。

### 使用示例：

```python
# 建立一個使用預訓練的 EfficientDet 模型，其組合縮放係數為 2
model = EfficientDet(compound_coef=2, pretrained=True)
```

---

## [Neck](./neck/)

- **目的**:
  此模組是為了實現不同的特徵金字塔網絡，用於目標檢測或語義分割任務中對特徵進行上下採樣和融合。

- **文件**:
  - `bifpn.py`: 包含 BiFPN 的定義
  - `fpn.py`: 包含 FPN 的定義

### 主要功能

#### 1. **特徵金字塔網絡 (Feature Pyramid Network, FPN)**:

- **類**: `FPN` 和 `FPNs`

  特徵金字塔網絡是一種用於目標檢測和語義分割的模型，它可以同時生成多尺度的特徵表示。通常由底層詳細特徵和上層語義強特徵組成。

#### 2. **雙向特徵金字塔網絡 (Bidirectional Feature Pyramid Network, BiFPN)**:

- **類**: `BiFPN` 和 `BiFPNs`

  BiFPN 是 FPN 的一個變種，它允許特徵在金字塔的各個層之間進行上下採樣，從而得到更豐富的特徵表示。它是 EfficientDet 架構的核心部分。

### 主要結構和組件

- **Neck 結構字典 (`NECK`)**:

  提供了一個查找表，以從名稱創建特定的金字塔網絡結構。

- **建立和列出功能**:

  - `build_neck()`: 根據提供的名稱和參數創建特定的特徵金字塔網絡。
  - `list_necks()`: 列出支持的所有金字塔網絡或根據特定過濾器列出模型。

### 使用示例

```python
# 建立一個 BiFPN 網絡
bifpn = build_neck('bifpn', in_channels_list=[64, 128, 256], out_channels=256)

# 列出所有支持的特徵金字塔網絡
all_necks = list_necks()
```

---

## [NN (Neural Networks)](./nn/)

**目的**:
這個模塊的目的是包含各種常用的深度學習模塊和組件。

**文件**:
- `aspp.py`: 這個文件包含了ASPP (Atrous Spatial Pyramid Pooling) 模組，這是一種常用於語義分割的模塊。
- `block.py`: 基礎模塊可能包含基礎的構建區塊，如卷積區塊或殘差區塊。
- `cnn.py`: 包含常規的卷積網絡模型，例如基礎的CNN。
- `dwcnn.py`: 深度可分離卷積網絡模型，這是一種高效的CNN變體。
- `grl.py`: 包含梯度反轉層，常用於域自適應技術中。
- `mbcnn.py`: MobileNet-like CNNs可能是一些輕量級的網絡模型。
- `positional_encoding.py`: 位置編碼，常用於Transformer架構中。
- `selayer.py`: Squeeze-and-Excitation層，增強了通道間的關聯性。
- `utils.py`: 包含一些實用的工具和助手函數。
- `vae.py`: Variational AutoEncoder，一種用於生成模型的自編碼器。

**components**:
- `activation.py`: 激活函數，如ReLU, Sigmoid等。
- `dropout.py`: dropout層，用於正則化。
- `loss.py`: 包含損失函數，如交叉熵、均方誤差等。
- `norm.py`: 正則化方法，如BatchNorm、LayerNorm等。
- `pooling.py`: 池化操作，如MaxPooling, AvgPooling等。

---

## [Optim](./optim/)

**目的**:

此模組的主要目的是為深度學習模型提供優化策略，特別是學習率的調整策略。適當的學習率策略可以幫助模型更快地收斂，並可能提高其最終性能。

**文件**:
  - `warm_up.py` - 這個文件定義了一個為學習率提供暖身策略的調度器。暖身策略是近年來在深度學習社區中變得流行的策略，它在訓練的開始階段逐步增加學習率，這對於避免訓練初期由於學習率太高而可能發生的不穩定性特別有用。

---

## [Transformers](./transformers/)

- **目的**: 此模組的主要目的是為深度學習模型提供各種基於Transformer的模型架構。這些架構在許多自然語言處理和計算機視覺的應用中都取得了卓越的成果。

- **文件**:
  - `basic.py` - 包含基礎的Transformer架構，如Encoder和相關層。
  - `efficientformer.py` - 包含EfficientFormer模型，這是一種針對效率優化的Transformer架構。
  - `metaformer.py` - 包含MetaFormer模型，這是一個具有多種可插拔功能的Transformer模型。
  - `mobilevit.py` - 包含MobileViT模型，這是專為移動裝置優化的Vision Transformer模型。
  - `poolformer.py` - 包含PoolFormer模型，這是一種將pooling策略整合到Transformer中的模型。
  - `token_mixer.py` - 包含Token Mixer模組，這是用於混合和操作輸入token的方法。
  - `utils.py` - 包含多種工具和助手函數，如計算patch大小和列出可用的transformer模型。
  - `vit.py` - 包含Vision Transformer模型，這是一種專為圖像處理設計的Transformer。

---

## [Utils](./utils/)

- **目的**: 此模組旨在提供各種工具函數，以支援上述的模塊。

- **文件**:
  - `cpuinfo.py` - 用於獲取 CPU 相關資訊。
  - `model_profile.py` - 提供模型分析和性能分析功能。
  - `replace.py` - 提供用於替換模型組件的工具。

### 主要功能

1. **CPU 資訊提取**：利用`cpuinfo.py`，用戶可以獲取詳細的CPU資訊，例如製造商、型號、核心數量等。
2. **模型分析和性能分析**：透過`model_profile.py`，用戶可以評估模型的複雜性、計算量等資訊。
3. **模型組件替換**：使用`replace.py`，開發者可以方便地替換或修改特定模型的組件。

### 主要結構和組件

1. **CPU資訊提取器**：`cpuinfo`功能是基於Pearu Peterson於2002年的著作，用於提取系統中的CPU詳細資訊。

   使用範例：
   ```python
   from cpuinfo import cpuinfo
   info = cpuinfo()
   print(list(info[0].keys()))
   ```

   返回的資訊範例如：'processor', 'vendor_id', 'cpu family', ... 等。

2. **模型複雜度和性能分析**：這部分提供了`get_model_complexity_info`、`get_cpu_gflops`和`get_meta_info`等函數，允許用戶評估模型的複雜性、計算量以及基於CPU的預估推理時間。

3. **模型組件替換工具**：這裡提供了`replace_module`和`replace_module_attr_value`兩個函數。前者允許用戶替換指定的模組，而後者則允許用戶修改指定模組的屬性值。
