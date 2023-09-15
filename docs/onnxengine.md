# ONNXEngine

`onnxengine` 模塊提供了一個方便的接口，用於加載和使用 ONNX 格式的模型。它支持 CPU 和 CUDA 兩種後端，並提供了一些工具函數，用於獲取和寫入 ONNX 模型的元數據。

## 目錄
- [ONNXEngine (`engine.py`)](#onnxengine-enginepy)
  - [說明](#說明)
  - [使用範例](#使用範例)
  - [ONNXEngine 初始化流程說明](#onnxengine-初始化流程說明)
    - [1. 輸入參數](#1-輸入參數)
    - [2. 設定裝置資訊](#2-設定裝置資訊)
    - [3. 設定提供者選項](#3-設定提供者選項)
    - [4. 設定會話選項](#4-設定會話選項)
    - [5. 初始化 onnxruntime 會話](#5-初始化-onnxruntime-會話)
    - [6. 設定 onnxruntime 會話資訊](#6-設定-onnxruntime-會話資訊)
    - [7. 蒐集模型的輸入和輸出資訊](#7-蒐集模型的輸入和輸出資訊)
- [metadata (`metadata.py`)](#metadata-metadatapy)
    - [說明](#說明-1)
    - [使用範例](#使用範例)
---

## [ONNXEngine (`engine.py`)](../docsaidkit/onnxengine/engine.py)

這個模塊是為了方便使用 `onnxruntime` 進行模型推理，並支持 CUDA 和 CPU 兩種後端。

### 說明

- `Backend`: 此列舉類別定義了可用的後端選項，可以選擇 `cpu` 或 `cuda`。

- `ONNXEngine`:
  - 是主要的模型推理引擎，可以方便地加載 ONNX 模型並進行推理。
  - 使用 `_get_session_info` 和 `_get_provider_info` 來獲取會話和提供者的相關配置。

- `get_onnx_metadata`: 這是一個方便的工具函數，可以快速獲取 ONNX 模型的元數據。

- `write_metadata_into_onnx`: 允許用戶簡單地向 ONNX 模型添加或更新元數據。

### 使用範例

1. **初始化 ONNXEngine**

    ```python
    from docsaidkit import ONNXEngine, Backend

    engine = ONNXEngine(model_path="your_model_path.onnx", backend=Backend.cuda, gpu_id=0)
    ```

2. **執行模型推理**

    假設您的模型需要一個名為 'input' 的輸入:

    ```python
    import numpy as np

    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    result = engine(input=input_data)
    ```

### ONNXEngine 初始化流程說明

當您創建一個 `ONNXEngine` 對象時，以下的初始化流程會被觸發：

#### 1. 輸入參數

- `model_path`: 指定 ONNX 或 ORT 格式的模型文件的路徑。
- `gpu_id`: 若使用 GPU 進行模型推理，此參數指定所使用的 GPU ID。預設為0。
- `backend`: 用於指定模型運行的後端，可以是 `cpu` 或 `cuda`。如果提供了字符串或整數，它將被轉換為 `Backend` 枚舉對象。
- `session_option`: 用於定義 ONNX runtime 會話的選項。
- `provider_option`: 用於定義執行提供者的配置選項。

#### 2. 設定裝置資訊

- 根據提供的 `backend` 參數確定運行模型的裝置。如果 `backend` 是 `cpu`，則 `device_id` 將設為 0。

#### 3. 設定提供者選項

- 通過 `_get_provider_info` 方法確定適當的執行提供者和其相關選項。

#### 4. 設定會話選項

- 使用 `_get_session_info` 方法來配置會話選項，如圖優化級別和日誌嚴重性級別。

#### 5. 初始化 onnxruntime 會話

- 使用 `ort.InferenceSession` 創建一個新的 ONNX runtime 會話。
- 這個會話將負責模型的實際推理過程。

#### 6. 設定 onnxruntime 會話資訊

- 存儲模型路徑至 `model_path`。
- 使用 `get_onnx_metadata` 獲取模型的元數據。
- 獲取會話使用的執行提供者列表。
- 獲取提供者的配置選項。

#### 7. 蒐集模型的輸入和輸出資訊

- 從 ONNX 會話中獲取模型的輸入和輸出資訊，並分別存儲在 `input_infos` 和 `output_infos` 中。

---

## [metadata (`metadata.py`)](../docsaidkit/onnxengine/metadata.py)

這個模塊提供了操作 ONNX 模型的元數據的工具，允許用戶讀取和寫入自定義元數據。它使用 `onnxruntime` 和 `onnx` 庫來處理模型的讀取和存儲。

### 說明

- **`get_onnx_metadata(onnx_path: Union[str, Path]) -> dict`**:
  - 參數:
    - `onnx_path`: 指定 ONNX 模型的路徑。
  - 該函數用於從指定的 ONNX 模型中提取元數據。它首先將模型路徑轉換為字符串格式，然後使用 `onnxruntime` 的 `InferenceSession` 加載模型，最後從模型中提取和返回元數據。

- **`write_metadata_into_onnx(onnx_path: Union[str, Path], out_path: Union[str, Path], drop_old_meta: bool = False, **kwargs)`**:
  - 參數:
    - `onnx_path`: 指定 ONNX 模型的路徑。
    - `out_path`: 指定存儲更新後的 ONNX 模型的路徑。
    - `drop_old_meta`: 如果設為 `True`，則在寫入新元數據之前會刪除現有的元數據。
    - `**kwargs`: 允許用戶提供其他要寫入模型的自定義元數據。
  - 該函數首先讀取指定的 ONNX 模型，然後檢查是否保留現有的元數據。接著，它更新元數據，包括當前的日期和時間，以及其他任何用戶指定的元數據。最後，它將更新後的元數據寫入 ONNX 模型並保存到指定的路徑。

### 使用範例

1. **提取 ONNX 模型的元數據**:

   ```python
   from metadata import get_onnx_metadata

   model_path = "path_to_your_model.onnx"
   metadata = get_onnx_metadata(model_path)
   print(metadata)
   ```

2. **寫入自定義元數據到 ONNX 模型**:

   ```python
   from metadata import write_metadata_into_onnx

   model_path = "path_to_your_model.onnx"
   output_path = "path_for_updated_model.onnx"
   custom_metadata = {
       "version": "1.0.1",
       "description": "Updated model with new training data"
   }

   write_metadata_into_onnx(model_path, output_path, **custom_metadata)
   ```
