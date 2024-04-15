# Vision

本模塊主要包含了影像處理、視覺幾何、視覺效果等相關功能。

## 目錄
- [Functionals (`functionals.py`)](#functionals-functionalspy)
  - [`meanblur`](#1-meanblur)
  - [`gaussianblur`](#2-gaussianblur)
  - [`medianblur`](#3-medianblur)
  - [`imcvtcolor`](#4-imcvtcolor)
  - [`imadjust`](#5-imadjust)
  - [`pad`](#6-pad)
  - [`imcropbox`](#7-imcropbox)
  - [`imcropboxes`](#8-imcropboxes)
  - [`imbinarize`](#9-imbinarize)
- [Geometric Transformations (`geometric.py`)](#geometric-transformations-geometricpy)
  - [`imresize`](#1-imresize)
  - [`imrotate90`](#2-imrotate90)
  - [`imrotate`](#3-imrotate)
  - [`imwarp_quadrangle`](#4-imwarp_quadrangle)
  - [`imwarp_quadrangles`](#5-imwarp_quadrangles)
- [Image Processing (`improc.py`)](#image-processing-improcpy)
  - [`is_numpy_img`](#1-is_numpy_img)
  - [`get_orientation_code`](#2-get_orientation_code)
  - [`jpgencode`](#3-jpgencode)
  - [`jpgdecode`](#4-jpgdecode)
  - [`jpgread`](#5-jpgread)
  - [`pngencode`](#6-pngencode)
  - [`pngdecode`](#7-pngdecode)
  - [`imencode`](#8-imencode)
  - [`imdecode`](#9-imdecode)
  - [`img_to_b64`](#10-img_to_b64)
  - [`npy_to_b64`](#11-npy_to_b64)
  - [`npy_to_b64str`](#12-npy_to_b64str)
  - [`img_to_b64str`](#13-img_to_b64str)
  - [`b64_to_img`](#14-b64_to_img)
  - [`b64str_to_img`](#15-b64str_to_img)
  - [`b64_to_npy`](#16-b64_to_npy)
  - [`b64str_to_npy`](#17-b64str_to_npy)
  - [`npyread`](#18-npyread)
  - [`imread`](#19-imread)
  - [`imwrite`](#20-imwrite)
- [Morphology (`morphology.py`)](#morphology-morphologypy)
  - [`imerode`](#1-imerode)
  - [`imdilate`](#2-imdilate)
  - [`imopen`](#3-imopen)
  - [`imclose`](#4-imclose)
  - [`imgradient`](#5-imgradient)
  - [`imtophat`](#6-imtophat)
  - [`imblackhat`](#7-imblackhat)
- [IP Camera (`ipcam`)](#ip-camera-ipcam)
  - [`WebDemo`](#1-webdemo)
  - [`IpcamCapture`](#2-ipcamcapture)
- [Video Tools (`videotools`)](#video-tools-videotools)
  - [`is_video_file`](#1-is_video_file)
  - [`get_step_inds`](#2-get_step_inds)
  - [`_extract_frames`](#3-_extract_frames)
  - [`video2frames`](#4-video2frames)
- [Visualization (`visualization`)](#visualization-visualization)
  - [`draw_box` & `draw_boxes`](#1-draw_box--draw_boxes)
  - [`draw_polygon` & `draw_polygons`](#2-draw_polygon--draw_polygons)
  - [`draw_text`](#3-draw_text)
  - [`generate_colors`](#4-generate_colors)
  - [`draw_ocr_infos`](#5-draw_ocr_infos)

## [Functionals (`functionals.py`)](../docsaidkit/vision/functionals.py)

這個模塊提供了一套針對影像處理的基本功能性函數。

以下是每個功能的簡短說明：

### 1. `meanblur`

- **功能**: 對影像進行平均模糊。
- **參數**:
  - `img`: 輸入的 Numpy 影像。
  - `ksize`: 模糊的核大小，預設為 3。
  - `**kwargs`: 其他參數。

### 2. `gaussianblur`

- **功能**: 對影像進行高斯模糊。
- **參數**:
  - `img`: 輸入的 Numpy 影像。
  - `ksize`: 模糊的核大小，預設為 3。
  - `sigmaX`: 標準差，預設為 0。
  - `**kwargs`: 其他參數。

### 3. `medianblur`

- **功能**: 對影像進行中值模糊。
- **參數**:
  - `img`: 輸入的 Numpy 影像。
  - `ksize`: 模糊的核大小，預設為 3。
  - `**kwargs`: 其他參數。

### 4. `imcvtcolor`

- **功能**: 改變影像的色彩空間。
- **參數**:
  - `img`: 輸入的 Numpy 影像。
  - `cvt_mode`: 轉換模式，可以是整數或字符串。

### 5. `imadjust`

- **功能**: 調整影像的亮度和對比度。
- **參數**:
  - `img`: 輸入的 Numpy 影像。
  - `rng_out`: 輸出範圍，預設為 (0, 255)。
  - `gamma`: 伽馬值，預設為 1.0。
  - `color_base`: 色彩基礎，預設為 'BGR'。

### 6. `pad`

- **功能**: 為影像增加邊框。
- **參數**:
  - `img`: 輸入的 Numpy 影像。
  - `pad_size`: 邊框的大小。
  - `fill_value`: 邊框的顏色或值，預設為 0。
  - `pad_mode`: 邊框的模式，預設為 BORDER.CONSTANT。

### 7. `imcropbox`

- **功能**: 根據給定的邊界框裁剪影像。
- **參數**:
  - `img`: 輸入的 Numpy 影像。
  - `box`: 裁剪的邊界框。
  - `use_pad`: 是否使用填充，預設為 False。

### 8. `imcropboxes`

- **功能**: 根據給定的多個邊界框裁剪影像。
- **參數**:
  - `img`: 輸入的 Numpy 影像。
  - `boxes`: 裁剪的邊界框列表。
  - `use_pad`: 是否使用填充，預設為 False。

### 9. `imbinarize`

- **功能**: 將影像二值化。
- **參數**:
  - `img`: 輸入的 Numpy 影像。
  - `threth`: 二值化閾值，預設為 cv2.THRESH_BINARY。
  - `color_base`: 色彩基礎，預設為 'BGR'。

## [Geometric Transformations (`geometric.py`)](../docsaidkit/vision/geometric.py)

這個模塊提供了一套針對影像進行幾何變換的功能性函數。

以下是每個功能的簡短說明：

### 1. `imresize`

- **功能**: 重新調整影像的大小。
- **參數**:
  - `img`: 輸入的 Numpy 影像。
  - `size`: 目標大小，格式為 `(height, width)`。
  - `interpolation`: 插值方式，預設為 `INTER.BILINEAR`。
  - `return_scale`: 是否返回縮放因子，預設為 `False`。

### 2. `imrotate90`

- **功能**: 將影像旋轉 90 度。
- **參數**:
  - `img`: 輸入的 Numpy 影像。
  - `rotate_code`: 旋轉方向和次數，取自 `ROTATE` 列舉類型。

### 3. `imrotate`

- **功能**: 根據指定的角度旋轉影像。
- **參數**:
  - `img`: 輸入的 Numpy 影像。
  - `angle`: 旋轉的角度。
  - `scale`: 縮放因子，預設為 1。
  - `interpolation`: 插值方式，預設為 `INTER.BILINEAR`。
  - `bordertype`: 邊界處理方式，預設為 `BORDER.CONSTANT`。
  - `bordervalue`: 邊界值，預設為 `None`。
  - `expand`: 是否擴展影像以包含完整的旋轉，預設為 `True`。
  - `center`: 旋轉中心，預設為 `None`。

### 4. `imwarp_quadrangle`

- **功能**: 根據指定的四邊形進行透視變換。
- **參數**:
  - `img`: 輸入的 Numpy 影像。
  - `polygon`: 透視變換的四邊形，可以是 `Polygon` 或 `np.ndarray`。

### 5. `imwarp_quadrangles`

- **功能**: 根據指定的多個四邊形進行透視變換。
- **參數**:
  - `img`: 輸入的 Numpy 影像。
  - `polygons`: 透視變換的四邊形列表。

## [Image Processing (`improc.py`)](../docsaidkit/vision/improc.py)

這個模塊專注於影像的讀取、寫入、編碼、解碼以及與其他格式之間的轉換。

以下是每個功能的簡短說明：

### 1. `is_numpy_img`
- **功能**: 檢查給定的物件是否是一個 Numpy 影像。

### 2. `get_orientation_code`
- **功能**: 從提供的流中取得方向代碼。

### 3. `jpgencode`
- **功能**: 將 Numpy 影像編碼為 JPEG 格式的 bytes。
- **參數**:
  - `img`: 輸入的 Numpy 影像。
  - `quality`: JPEG 壓縮品質。

### 4. `jpgdecode`
- **功能**: 將 JPEG 格式的 bytes 解碼為 Numpy 影像。

### 5. `jpgread`
- **功能**: 從給定的路徑讀取 JPEG 影像。

### 6. `pngencode`
- **功能**: 將 Numpy 影像編碼為 PNG 格式的 bytes。

### 7. `pngdecode`
- **功能**: 將 PNG 格式的 bytes 解碼為 Numpy 影像。

### 8. `imencode`
- **功能**: 將 Numpy 影像編碼為指定的圖像格式的 bytes。

### 9. `imdecode`
- **功能**: 將圖像格式的 bytes 解碼為 Numpy 影像。

### 10. `img_to_b64`
- **功能**: 將 Numpy 影像轉換為 base64 編碼。

### 11. `npy_to_b64`
- **功能**: 將 Numpy 陣列轉換為 base64 編碼。

### 12. `npy_to_b64str`
- **功能**: 將 Numpy 陣列轉換為 base64 字串。

### 13. `img_to_b64str`
- **功能**: 將 Numpy 影像轉換為 base64 字串。

### 14. `b64_to_img`
- **功能**: 從 base64 編碼轉換為 Numpy 影像。

### 15. `b64str_to_img`
- **功能**: 從 base64 字串轉換為 Numpy 影像。

### 16. `b64_to_npy`
- **功能**: 從 base64 編碼轉換為 Numpy 陣列。

### 17. `b64str_to_npy`
- **功能**: 從 base64 字串轉換為 Numpy 陣列。

### 18. `npyread`
- **功能**: 從指定的路徑讀取 Numpy 陣列。

### 19. `imread`
- **功能**: 從指定的路徑讀取圖像。

### 20. `imwrite`
- **功能**: 將 Numpy 影像寫入指定的路徑。

## [Morphology (`morphology.py`)](../docsaidkit/vision/morphology.py)

這個模塊專注於形態學操作，它是影像處理中的一個重要部分，主要用於提取影像組件、修復和分隔等。

以下是這些功能的詳細說明：

### 1. `imerode`：
- **功能**: 對影像進行腐蝕操作。
- **參數**:
    - `img`: 需要處理的影像。
    - `ksize`: 核的大小。
    - `kstruct`: 核的結構形式。

### 2. `imdilate`：
- **功能**: 對影像進行膨脹操作。
- **參數**:
    - 同上。

### 3. `imopen`：
- **功能**: 先腐蝕後膨脹，常用於移除噪音。
- **參數**:
    - 同上。

### 4. `imclose`：
- **功能**: 先膨脹後腐蝕，常用於關閉前景物體內部的小孔或小黑點。
- **參數**:
    - 同上。

### 5. `imgradient`：
- **功能**: 計算影像的形態學梯度，即膨脹圖像和腐蝕圖像的差。
- **參數**:
    - 同上。

### 6. `imtophat`：
- **功能**: 計算原始影像和開操作後影像的差，用於突顯比鄰近的像素亮度更高的細節。
- **參數**:
    - 同上。

### 7. `imblackhat`：
- **功能**: 計算閉操作後影像和原始影像的差，用於突顯比鄰近的像素亮度更低的細節。
- **參數**:
    - 同上。

### 使用方式：

這些函數都非常簡單易用。只需將你的影像、所需的核大小和核的結構形式作為參數傳入對應的函數，然後進行相應的形態學操作。

例如：如果你想對一個影像 `img` 進行膨脹操作，只需調用 `imdilate(img)`。如果你想使用其他核大小或形狀，只需修改 `ksize` 和 `kstruct` 參數。

## [IP Camera (`ipcam`)](../docsaidkit/vision/ipcam/)

這個模塊提供了從 IP 攝像頭獲取視頻流並將其顯示在 web 應用中的功能。

- **App (`app.py`)**: 定義IP攝像頭的主要應用程序。
- **Camera (`camera.py`)**: 處理與攝像頭的連接和數據流。
- **Video Streaming (`video_streaming.html`)**: 提供了攝像頭的視頻流視覺化界面。

以下是主要功能的說明：

### 1. `WebDemo`：

這是一個簡單的 web 應用，用於顯示 IP 攝像頭的視頻流。

- **`__init__`**:
  - **功能**: 初始化類並設定相關配置。
  - **參數**:
    - `camera_ip`: IP 攝像頭的 IP 地址。
    - `color_base`: 影像的顏色空間。
    - `route`: Web 服務的路由。

- **`_index`**: 返回用於顯示視頻流的 HTML 頁面。

- **`_video_feed`**: 返回視頻流的 Response 對象。

- **`gen`**: 獲取來自 IP 攝像頭的每一幀並將其編碼為 JPEG 格式。

- **`run`**: 啟動 Flask web 服務。

### 2. `IpcamCapture`：

這個類用於從 IP 攝像頭獲取視頻流。

- **`__init__`**:
  - **功能**: 初始化類並打開視頻流。
  - **參數**:
    - `url`: 視頻流的源地址，可以是設備索引或網絡地址。
    - `color_base`: 影像的顏色空間。

- **`_queryframe`**: 這是一個私有方法，用於連續獲取視頻流中的幀。

- **`get_frame`**: 返回當前的幀。如果沒有可用的幀，則返回一個黑色的幀。

- **`__iter__`**: 使該類成為可迭代對象，返回當前的幀。

### 使用方式：

1. 初始化 `IpcamCapture` 類，指定 IP 攝像頭的網絡地址。
2. 初始化 `WebDemo` 類，指定 IP 攝像頭的 IP 地址。
3. 調用 `WebDemo` 類的 `run` 方法，啟動 Flask web 服務。

## [Video Tools (`videotools`)](../docsaidkit/vision/videotools/)

這個模塊專注於處理視頻文件，主要是將視頻轉換成帧。

以下是該功能的詳細說明：

### 1. `is_video_file`
- **功能**: 檢查指定的路徑是否存在且是支援的視頻格式。

### 2. `get_step_inds`
- **功能**: 生成所需的帧索引列表。
- **輸出**: 視頻的幀索引列表。

### 3. `_extract_frames`
- **功能**: 從視頻中提取指定的帧。
- **輸出**: 視頻的幀列表。

### 4. `video2frames`
- **功能**: 使用多線程從視頻中提取帧。
- **輸出**: 視頻的幀列表。

### 使用方法

如果您想從一個視頻中提取帧，只需調用 `video2frames` 函數並傳入視頻的路徑以及其他參數。

例如：要從 `my_video.mp4` 提取帧，每秒2帧，從第1秒開始到第10秒結束，可以這樣做：
```python
frames = video2frames('my_video.mp4', frame_per_sec=2, start_sec=1, end_sec=10)
```

您可以選擇使用所有的預設參數，這樣將提取視頻中的所有帧：
```python
frames = video2frames('my_video.mp4')
```

### 注意
- 確保視頻路徑存在，並且其格式支援。
- 使用多線程可以加快提取速度，但需要注意不要過度使用，以免耗盡系統資源。
- 如果選擇的開始和結束時間超出了視頻的長度，則會引發異常。

## [Visualization (`visualization`)](../docsaidkit/vision/visualization/)

這個模塊提供了一套專門針對圖像視覺化的功能，適合用於物體檢測或 OCR 的情境。

以下是每個功能的簡短說明：

### 1. `draw_box` & `draw_boxes`

- **功能**: 在圖像上繪製邊界框。
  - `draw_box`: 繪製單一的邊界框。
  - `draw_boxes`: 同時繪製多個邊界框。

- **參數**:
  - `img`: 輸入的 Numpy 影像。
  - `coordinates`: 邊界框的坐標。
  - `color`: 線條顏色，預設為白色。
  - `thickness`: 線條厚度，預設為2。
  - `**kwargs`: 其他參數。

### 2. `draw_polygon` & `draw_polygons`

- **功能**: 在圖像上繪製多邊形。
  - `draw_polygon`: 繪製單一的多邊形。
  - `draw_polygons`: 同時繪製多個多邊形。

- **參數**:
  - `img`: 輸入的 Numpy 影像。
  - `vertices`: 多邊形的頂點坐標。
  - `color`: 線條顏色，預設為白色。
  - `thickness`: 線條厚度，預設為2。
  - `**kwargs`: 其他參數。

### 3. `draw_text`

- **功能**: 在指定的圖像位置上繪製指定的文字。

- **參數**:
  - `img`: 輸入的 Numpy 影像。
  - `text`: 要顯示的文字。
  - `position`: 文字的起始位置。
  - `font`: 字體，預設為 OpenCV 的 Hershey 字體。
  - `color`: 文字顏色。
  - `size`: 文字大小。
  - `**kwargs`: 其他參數。

### 4. `generate_colors`

- **功能**: 生成指定數量的顏色，基於選定的色彩方案。

- **參數**:
  - `num_colors`: 顏色的數量。
  - `color_scheme`: 色彩方案，如`hsv`、`triadic`、`analogous`或`square`。
  - `**kwargs`: 其他參數。

### 5. `draw_ocr_infos`

- **功能**: 在圖像上顯示OCR的結果。

- **參數**:
  - `img`: 輸入的 Numpy 影像。
  - `ocr_data`: 從OCR系統獲得的資料。
  - `display_mode`: 顯示模式，例如'bbox'或'polygon'。
  - `color`: 線條或文字的顏色。
  - `thickness`: 線條厚度。
  - `**kwargs`: 其他參數。
