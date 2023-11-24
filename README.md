# DocsaidKit

## 目錄

- [介紹](#介紹)
- [安裝說明](#安裝說明)
  - [先決條件](#先決條件)
  - [安裝方式](#安裝方式)
    - [通用方式](#通用方式)
    - [MacOS](#macos)
- [使用方式](#使用方式)
  - [概述](#概述)
  - [Structure](#structure)
  - [Vision](#vision)
  - [ONNXEngine](#onnxengine)
  - [Pytorch](#pytorch)
  - [Others](#others)
- [Pytest 測試](#pytest-測試)
- [CI/CD](#cicd)
  - [持續集成 (CI)](#持續集成-ci)
  - [持續交付 (CD)](#持續交付-cd)

## 介紹

**DocsaidKit** 是一個 Python 深度學習工具箱，是我們內部開發用的核心 Python 套件。

其內容是我們內部的開發者共同設計撰寫，以支持深度學習和計算機視覺相關的項目和研究。我們希望這套工具能為組織內的開發者提供一個統一的工作流程，使模型的開發和部署變得更為直接。

DocsaidKit 的結構分為以下主要模組：

- **vision**：此模組包含計算機視覺相關的功能，例如圖像處理、影片處理等。

- **structures**：此模組是用於處理 Bounding Box 等結構化資料的模組。

- **onnxengine**：此模組提供 ONNX 推論相關的功能，支援 ONNX 格式的模型。

- **torch**：此模組與 PyTorch 有關，內含神經網絡結構、優化器等相關功能。

- **utils**：此模組提供各種工具函數，包括文件處理、時間處理等常用功能。

- **tests**：此目錄包含測試文件，目的是驗證 DocsaidKit 的各個功能。


DocsaidKit 的目標是簡化深度學習和計算機視覺的開發過程，並提供清晰的結構和模組化的設計，以方便使用者查找和擴展功能。

## 安裝說明

在開始安裝 DocsaidKit 之前，請確保您已完成以下先決條件：

### 先決條件

1. **Python版本**:

    - 確保您的系統已安裝Python 3.8或以上版本。

2. **依賴套件**:

    - **Ubuntu**:

      ```bash
      sudo apt install libturbojpeg exiftool ffmpeg
      ```

    - **MacOS**:

      ```bash
      # 使用 brew 進行安裝
      brew install jpeg-turbo exiftool ffmpeg
      ```

3. **pdf2image 的安裝**:

    - 建議參考此[頁面](https://github.com/Belval/pdf2image)以完成安裝。

### 安裝方式

完成先決條件後，您可以選擇以下的安裝方式：

#### 通用方式：

補充：本套件僅提供內部的 Pypi 服務，若您是外部使用者，請直接使用 git clone 並使用 setup.py 安裝。

1. **通過 git clone 並使用 setup.py 安裝 (推薦)**:

    - 基本安裝:

      ```bash
      pip install wheel
      python setup.py bdist_wheel
      pip install dist/DocsaidKit-${version}-none-any.whl
      ```

    - 針對 **訓練環境** 的安裝：

      ```bash
      pip install wheel
      python setup.py bdist_wheel
      pip install "dist/DocsaidKit-${version}-none-any.whl[torch]"
      ```

3. **通過 git clone 並使用 docker 安裝 (穩定環境需求)**:

    ```bash
    bash docker/build.bash
    ```

    使用方式請參考：[Docker](./docker/README.md)

3. **通過 PyPi (內部開發者)**:

    - 基本安裝：

      ```bash
      tgt_ip=192.168.xxx.xxx
      pip install --trusted-host $tgt_ip \
        --index-url http://$tgt_ip:8080/simple/ \
        DocsaidKit==$version
      ```

    - 針對 **開發環境** 需要安裝的 pytorch 及相關套件：

      ```bash
      tgt_ip=192.168.xxx.xxx
      pip install --trusted-host $tgt_ip \
        --index-url http://$tgt_ip:8080/simple/ \
        "DocsaidKit[torch]==$version"
      ```

#### MacOS-Arm64

由於 MacOS-Arm64 的限制，我們建議使用 conda 進行安裝。

1. **安裝環境管理器 (例如: conda)**:

    - 下載 conda 安裝腳本：

      ```bash
      wget https://github.com/conda-forge/miniforge#download
      ```

    - 運行安裝腳本：

      ```bash
      bash Miniforge3-MacOSX-arm64.sh
      ```

2. **創建並啟動環境**:

    - 創建一個新的環境：

      ```bash
      conda create -n DocsaidKit python=3.8
      ```

    - 啟動該環境：

      ```bash
      conda activate DocsaidKit
      ```

3. **安裝 DocsaidKit**:

    - 在啟動環境後，使用以下命令安裝 DocsaidKit：

      ```bash
      pip install wheel setuptools
      python setup.py bdist_wheel
      pip install dist/*.whl
      ```

如有任何疑問或需要進一步的說明，請查看官方文檔或與我們聯絡。

## 使用方式

### 概述

DocsaidKit 的使用方式旨在為研究人員和開發者提供一個簡單的接口，用於深度學習和計算機視覺領域的項目和研究。

在這裡，我們會介紹如何使用 DocsaidKit，並提供一些基本的使用示例。

### Structure

`structures` 模塊是 DocsaidKit 中的一個重要模塊，用於表示和處理一般化的框架（如邊界框）的格式和操作。

詳細說明請參考 [Structure](./docs/structure.md)。

### Vision

`vision` 模塊包含計算機視覺相關的功能，例如影像處理、視覺幾何、視覺效果等相關功能。。

詳細說明請參考 [Vision](./docs/vision.md)。

### ONNXEngine

`onnxengine` 模塊提供 ONNX 推論相關的功能，支援 ONNX 格式的模型。

詳細說明請參考 [ONNXEngine](./docs/onnxengine.md)。

### Pytorch

`torch` 模塊與 PyTorch 有關，內含神經網絡結構、優化器等相關功能。

詳細說明請參考 [Pytorch](./docs/pytorch.md)。

### Others

`utils` 模塊提供各種工具函數，包括文件處理、時間處理等常用功能。

詳細說明請參考 [Utils](./docs/utils.md)。

## Pytest 測試

為了確保 DocsaidKit 的功能穩定性和正確性，我們採用了 `pytest` 進行單元測試。

使用者可以自行運行測試，以驗證所使用功能的正確性。

運行測試方法如下：

```bash
python -m pytest tests
```

## CI/CD

**DocsaidKit** 專案嚴格遵循持續整合（CI）與持續交付（CD）的最佳實務，以確保程式碼的品質和專案的穩定性。

我們利用 [GitHub Actions](https://github.com/features/actions) 作為主要的 CI/CD 工具，自動化地執行程式碼檢查、測試和發佈流程。

### 持續整合 (CI)

我們的持續整合流程目的是確保所有程式碼提交和合併請求都能達到預設的品質標準。

以下是 CI 流程中的主要步驟：

1. **程式碼檢出**：
   - 取得最新的程式碼，確保測試和建構是基於最新的程式碼更改。

2. **環境設置**：
   - 根據需要的 Python 版本和操作系統設置運行環境。

3. **依賴安裝**：
   - 安裝必要的系統依賴和 Python 依賴套件，為後續的測試和建構步驟做好準備。

4. **程式碼品質檢查**：
   - 使用 `pylint` 對程式碼進行靜態分析，檢查程式碼的品質和風格。

5. **單元測試和整合測試**：
   - 通過 `pytest` 執行單元測試和整合測試，確保程式碼的功能正確性和穩定性。
   - 生成測試覆蓋率報告，並通過 GitHub Actions 將覆蓋率資訊回傳給開發團隊。

### 持續交付 (CD)

我們的持續交付流程旨在自動化建構和發佈過程，確保高效、準確地將新版本交付給使用者。

以下是 CD 流程中的主要步驟：

1. **版本號更新**：
   - 根據輸入的版本標籤自動更新程式碼中的版本號。

2. **建構和打包**：
   - 自動化建構 Python wheel 包，方便分發和安裝。
   - 使用 `twine` 工具將建構好的 wheel 包上傳到私有倉庫。

3. **建立發佈**：
   - 在 GitHub 上建立一個新的 release，填寫版本資訊和發佈說明。
   - 上傳建構好的 wheel 包作為 release 的附件，方便使用者下載。

4. **清理建構環境**：
   - 清理建構產生的臨時檔案和目錄，保持建構環境的整潔。

---

希望以上的說明能讓您對 DocsaidKit 有更深的了解。

如果遇到任何問題或需要進一步的支援，歡迎瀏覽我們的 [GitHub](https://github.com/DocsaidLab/DocsaidKit)。
