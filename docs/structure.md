# Structure

`structures` 模塊是 DocsaidKit 中的一個重要模塊，用於表示和處理一般化的框架（如邊界框）的格式和操作。

## 目錄

- [Box, Boxes and BoxMode](#box-boxes-and-boxmode)
  - [BoxMode](#boxmode)
  - [Box](#box)
  - [Boxes](#boxes)
- [Polygon, Polygons](#polygon-polygons)
    - [Polygon](#polygon)
    - [Polygons](#polygons)
- [其他](#其他)
    - [pairwise_intersection](#pairwise_intersection)
    - [pairwise_iou](#pairwise_iou)
    - [pairwise_ioa](#pairwise_ioa)
    - [merge_boxes](#merge_boxes)

---

## [**Box, Boxes and BoxMode**](../docsaidkit/structures/boxes.py)

### **BoxMode**：

這是一個列舉類型，用於定義不同的邊界框表示方式。它包含以下格式：

- `XYXY`: 使用左上角和右下角的坐標。
- `XYWH`: 使用左上角的坐標和框的寬度和高度。
- `CXCYWH`: 使用框的中心坐標、寬度和高度。

此外，`BoxMode` 類還提供了一個方法 `convert`，可以用來轉換框的格式。

---

### **Box**：

這是一個類，用於表示一個邊界框。要使用 `Box` 類來表示和操作邊界框，首先需要從 `docsaidkit` 匯入相應的類型和方法。以下是如何創建、表示和操作 `Box` 的基本步驟：

1. **匯入必要的類型和方法**：

    在你的 Python 程式中，首先導入 `Box` 和 `BoxMode`：

    ```python
    from docsaidkit import Box, BoxMode
    ```

2. **創建 Box 對象**：

    使用 `Box` 類的構造函數，你可以創建一個新的邊界框。在這裡，你需要提供一個數組來表示框的坐標以及一個表示框坐標格式的 `BoxMode`：

    ```python
    box = Box([50, 50, 150, 150], box_mode=BoxMode.XYXY)
    ```

    這裡，我們創建了一個左上角坐標為 (50, 50) 且右下角坐標為 (150, 150) 的邊界框，並指定其格式為 `XYXY`。

3. **使用 Box 方法**：

   - `copy()`: 創建 `Box` 物件的拷貝。
   - `numpy()`: 將 `Box` 物件轉換為 numpy 數組。
   - `normalize(w, h)`: 正規化Box的坐標。
   - `denormalize(w, h)`: 反正規化Box的坐標。
   - `clip(xmin, ymin, xmax, ymax)`: 將Box裁剪到給定的坐標範圍。
   - `shift(shift_x, shift_y)`: 平移Box。
   - `scale(factor_x, factor_y)`: 縮放Box。
   - `area()`: 獲得Box的面積。
   - `aspect_ratio()`: 獲得Box的寬高比。
   - `center()`: 計算Box的中心點。
   - `left_top()`: 獲得Box的左上角坐標。
   - `right_bottom()`: 獲得Box的右下角坐標。
   - `left_bottom()`: 獲得Box的左下角坐標。
   - `height()`: 獲得Box的高度。
   - `width()`: 獲得Box的寬度。
   - `convert(mode)`: 將Box從一種形式轉換為另一種形式。
   - `square()`: 轉換Box為正方形，保持中心不變。
   - `to_list(flatten)`: 將Box轉換為列表。
   - `to_polygon()`: 將Box轉換為 `docsaidkit.Polygon` 物件。
   - `tolist()`: 將Box轉換為列表。

---

### **Boxes**：

`Boxes` 類代表了一組邊界框。它提供了一種高效的方式來同時處理、操作和表示多個邊界框。為了有效地使用 `Boxes` 類，了解其基本操作和功能是至關重要的。

1. **匯入必要的類型和方法**：

   要開始使用 `Boxes` 類，首先需要匯入它以及 `BoxMode`：

   ```python
   from docsaidkit import Boxes, BoxMode
   ```

2. **創建 Boxes 物件**：

   使用 `Boxes` 類，你可以輕鬆地表示多個邊界框。這個類的構造函數需要一個數組結構（如列表或 numpy 數組）來表示每個框的坐標，以及一個 `BoxMode` 來表示這些坐標的格式：

   ```python
   boxes = Boxes([[50, 50, 150, 150], [100, 100, 200, 200]], box_mode=BoxMode.XYXY)
   ```

   在上面的代碼片段中，我們創建了兩個邊界框。第一個框的左上角坐標是 (50, 50) 且右下角坐標是 (150, 150)。第二個框的定義也類似。使用的坐標格式是 `XYXY`。

3. **利用 Boxes 的方法**：

   - `copy()`: 創建 `Boxes` 物件的拷貝。
   - `numpy()`: 將 `Boxes` 物件轉換為 numpy 數組。
   - `normalize(w, h)`: 正規化Box的坐標。
   - `denormalize(w, h)`: 反正規化Box的坐標。
   - `clip(xmin, ymin, xmax, ymax)`: 將Box裁剪到給定的坐標範圍。
   - `shift(shift_x, shift_y)`: 平移Box。
   - `scale(factor_x, factor_y)`: 縮放Box。
   - `area()`: 獲得每個Box的面積。
   - `aspect_ratio()`: 獲得Box的寬高比。
   - `center()`: 計算Box的中心點。
   - `left_top()`: 獲得Box的左上角坐標。
   - `right_bottom()`: 獲得Box的右下角坐標。
   - `height()`: 獲得Box的高度。
   - `width()`: 獲得Box的寬度。
   - `convert(mode)`: 將Box從一種形式轉換為另一種形式。
   - `drop_empty()`: 移除空的Box。
   - `get_empty_index()`: 獲得空Box的索引。
   - `square()`: 轉換Box為正方形，保持中心不變。
   - `to_list(flatten)`: 將Box轉換為列表。
   - `to_polygons()`: 將Box轉換為 `docsaidkit.Polygons` 物件。
   - `tolist()`: 將Box轉換為列表。

---

## [**Polygon, Polygons**](../docsaidkit/structures/polygons.py)

### **Polygon**：

`Polygon` 類代表了一組多邊形。這個類提供了一個高效的方法來同時處理、操作和表示多個多邊形。為了更有效地使用 `Polygon` 類，了解其基本操作和功能是非常重要的。

1. **匯入必要的類型和方法**：

   要開始使用 `Polygon` 類，首先需要匯入它：

   ```python
   from docsaidkit import Polygon
   ```

2. **創建 Polygon 物件**：

   使用 `Polygon` 類，你可以方便地表示多邊形。這個類的建構函數接受一個數組（如列表或 numpy 數組）以表示多邊形的每個頂點，並可以指定它是否已經被正規化：

   ```python
   poly = Polygon([[1, 2], [2, 3], [3, 4]])
   ```

   在上述的代碼片段中，我們建立了一個多邊形，其頂點由給定的座標列表表示。

3. **利用 Polygon 的方法**：

   - `copy()`: 創建 `Polygon` 物件的拷貝。
   - `numpy()`: 將 `Polygon` 物件轉換為 numpy 數組。
   - `normalize(w, h)`: 正規化多邊形的坐標。
   - `denormalize(w, h)`: 反正規化多邊形的坐標。
   - `clip(xmin, ymin, xmax, ymax)`: 將多邊形裁剪到給定的坐標範圍。
   - `shift(shift_x, shift_y)`: 平移多邊形。
   - `scale(distance, join_style)`: 返回在給定距離內的所有點的近似表示。
   - `to_convexhull()`: 計算多邊形的凸包。
   - `to_min_boxpoints()`: 將多邊形轉換為最小面積邊界框。
   - `to_box(box_mode)`: 將多邊形轉換為邊界框，其型別為 `docsaidkit.Box`。
   - `to_list(flatten)`: 將多邊形轉換為列表。
   - `is_empty(threshold)`: 檢查多邊形是否為空。
   - `moments`: 計算多邊形的質心。
   - `area`: 獲得多邊形的面積。
   - `arclength`: 獲得多邊形的周長。
   - `centroid`: 計算多邊形的中心。
   - `boundingbox`: 獲得多邊形的邊界框。
   - `min_circle`: 計算多邊形的最小封閉圓。
   - `min_box`: 計算多邊形的最小面積矩形。
   - `orientation`: 獲得多邊形的方向。
   - `min_box_wh`: 獲得最小面積矩形的寬度和高度。
   - `extent`: 計算多邊形的面積與其邊界框面積之間的比例。
   - `solidity`: 計算多邊形的面積與其凸包面積之間的比例。

---

### **Polygons**：

`Polygons` 類代表了一組多邊形。這個類提供了一個高效的方法來同時處理、操作和表示多個多邊形。為了更有效地使用 `Polygons` 類，了解其基本操作和功能是非常重要的。

1. **匯入必要的類型和方法**：

   要開始使用 `Polygons` 類，首先需要匯入它和相關的模塊：

   ```python
   from docsaidkit import Polygons
   ```

2. **創建 Polygons 物件**：

   使用 `Polygons` 類，你可以方便地表示一組多邊形。這個類的建構函數接受一組多邊形和一個指示這些多邊形是否已被正規化的布林值：

   ```python
   polygons = Polygons([[[1, 2], [2, 3], [3, 4]], [[2, 2], [3, 3], [4, 4]]], normalized=True)
   ```

3. **利用 Polygons 的方法**：

   - `is_empty`: 檢查每個多邊形是否為空。
   - `to_min_boxpoints`: 將每個多邊形轉換為最小面積邊界框的點。
   - `to_convexhull`: 計算每個多邊形的凸包。
   - `to_boxes`: 將每個多邊形轉換為邊界框，其型別為 `docsaidkit.Boxes`。
   - `drop_empty`: 刪除空的多邊形。
   - `copy`: 複製當前 `Polygons` 物件。
   - `normalize`: 正規化所有多邊形的座標。
   - `denormalize`: 反正規化所有多邊形的座標。
   - `scale`: 縮放每個多邊形。
   - `numpy`: 將 `Polygons` 物件轉換為 numpy 數組。
   - `to_list` 和 `tolist`: 將 `Polygons` 物件轉換為列表。
   - `moments`: 計算每個多邊形的質心。
   - `min_circle`: 計算每個多邊形的最小封閉圓。
   - `min_box`: 計算每個多邊形的最小面積矩形。
   - 其他屬性，例如 `area`, `arclength`, `centroid`, `boundingbox`, `extent`, `solidity`, `orientation`, 和 `min_box_wh` 可以方便地查詢有關每個多邊形的相關信息。

---

### [**其他**](../docsaidkit/structures/functionals.py)

#### `pairwise_intersection`

**功能**:
此函數用於計算兩組Box間的每對Box的交集面積。

**使用方式**:

```python
交集面積 = pairwise_intersection(Box1, Box2)
```

**參數**:
- `Box1, Box2`: 兩個 `Boxes` 物件，代表一系列的邊界框。

**返回值**:
- `交集`: 這是一個 [N, M] 的2D numpy數組。其中，N是Box1中的Box數量，而M是Box2中的Box數量。數組中的每一個元素 (i, j) 代表Box1中的第i個Box和Box2中的第j個Box的交集面積。

---

#### `pairwise_iou`

**功能**:
用於計算兩組Box間的每一對Box的IoU值（交集對聯集的比率）。

**使用方式**:

```python
iou值 = pairwise_iou(Box1, Box2)
```

**參數**:
- `Box1, Box2`: 分別代表兩組 `Boxes` 物件中的一系列邊界框。

**返回值**:
- 這是一個形狀為 [N, M] 的ndarray，其中，每個值代表對應Box對的IoU值。

---

#### `pairwise_ioa`

**功能**:
用於計算兩組Box間的每一對Box的IoA值（交集對面積的比率）。

**使用方式**:

```python
ioa值 = pairwise_ioa(Box1, Box2)
```

**參數**:
- `Box1, Box2`: 分別代表兩組 `Boxes` 物件中的一系列邊界框。

**返回值**:
- 這是一個形狀為 [N, M] 的ndarray，其中，每個值代表對應Box對的IoA值。

---

#### `merge_boxes`

**功能**:
此函數將重疊的Box進行合併。當Box之間的IoU值超過設定的閾值時，將進行合併。此合併過程使用圖論策略，首先將重疊的Box分組，然後計算每組的組合邊界框。

**使用方式**:

```python
merged_boxes, merged_idx = merge_boxes(Boxes, threshold=0.5)
```

**參數**:
- `Boxes`: 一個 `Boxes` 物件，代表一系列的邊界框。
- `threshold`: IoU的閾值，當Box間的IoU超過此值時將進行合併。預設值為0.5。

**返回值**:
- `merged_boxes`: 一個新的`Boxes`物件，其中包含了合併後的邊界框。
- `merged_idx`: 一個索引列表，每個元素代表從原始Boxes合併到新`merged_boxes`的Box組。

**注意**:
此函數的操作需要`networkx`庫。NetworkX是一個Python套件，專為處理複雜網絡結構而設。
