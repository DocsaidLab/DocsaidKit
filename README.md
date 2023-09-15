# DocsaidKit

DocsaidKit 是我們設計的核心 Python 套件。

DocsaidKit 整合了 [PyTorch-Lightning](https://www.pytorchlightning.ai/)，從而簡化模型的訓練和推理流程。它支持資料加載、模型訓練及評估，使我們能有效利用 PyTorch-Lightning 的功能，如分佈式訓練、混合精度訓練及模型儲存技術。此外，DocsaidKit 集成了 [ONNXRuntime](https://onnxruntime.ai/)，一款專為 ONNX 模型設計的運行時引擎。這使得技術團隊可以進行模型優化和部署，確保模型在不同硬體平台上的性能。在電腦視覺領域，DocsaidKit 整合了 [MMCV](https://github.com/open-mmlab/mmcv) 的功能。通過 MMCV，工程師能接觸到多種視覺模型和資料集，輔助電腦視覺的研究和開發。DocsaidKit 也融入了 [timm](https://github.com/rwightman/pytorch-image-models)、[hugging face](https://huggingface.co/) 等套件的部分功能，擴展了模型的應用範疇，涵蓋從自然語言處理到圖像識別的需求。


總之，我們希望這套工具能為開發者提供一個統一的工作流程，使模型的開發和部署變得更為直接，使他們能充分利用 PyTorch Lightning、ONNX Runtime、MMCV 以及其他套件的功能。對於希望深入了解 DocsaidKit 功能的開發者，我們提供了一系列的文檔和資源，供開發者查閱。

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

**DocsaidKit** 是一個 Python 深度學習工具箱，專為研究人員和開發者設計，以支持深度學習和計算機視覺相關的項目和研究。

DocsaidKit 的結構分為以下主要模組：

- **docsaidkit**：專案的根目錄，內含核心庫的代碼和功能。

- **tests**：此目錄包含測試代碼和資源文件，目的是驗證 DocsaidKit 的各個功能。

- **utils**：此模組提供各種工具函數，包括文件處理、時間處理等常用功能。

- **vision**：此模組包含計算機視覺相關的功能，例如圖像處理、視頻處理等。

- **torch**：此模組與 PyTorch 有關，內含神經網絡結構、優化器等相關功能。

- **onnxengine**：此模組提供 ONNX 推論相關的功能，支援 ONNX 格式的模型。

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

1. **通過 PyPi (推薦)**:

    - 基本安裝：

      ```bash
      pip install --trusted-host 192.168.0.105 \
        --index-url http://192.168.0.105:8080/simple/ \
        DocsaidKit==$version
      ```

    - 針對 **開發環境** 需要安裝的 pytorch 及相關套件：

      ```bash
      pip install --trusted-host 192.168.0.105 \
        --index-url http://192.168.0.105:8080/simple/ \
        "DocsaidKit[torch]==$version"
      ```

2. **通過 git clone 並使用 setup.py 安裝**:

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
      pip install --trusted-host 192.168.0.105 \
        --index-url http://192.168.0.105:8080/simple/ \
        DocsaidKit==$version
      ```

      或通過 git clone:

      ```bash
      pip install wheel setuptools
      python setup.py bdist_wheel
      pip install dist/*.whl
      ```

4. **安裝額外的功能相關依賴**:

    - 如果您打算使用機器學習相關的工具，可以安裝 PyTorch 及其他相關庫：

      ```bash
      pip install torch torchvision
      ```

    - 若需要 OpenMMLab 支持：

      ```bash
      pip install openmim
      mim install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html
      ```

希望這份指南可以協助您順利地完成 DocsaidKit 的安裝！如有任何疑問或需要進一步的說明，請查看官方文檔或與我們聯絡。

## 使用方式

### 概述

DocsaidKit 的使用方式旨在為研究人員和開發者提供一個簡單的接口，用於深度學習和計算機視覺領域的項目和研究。在這裡，我們會介紹如何使用 DocsaidKit，並提供一些基本的使用示例。

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

為了確保 DocsaidKit 的功能穩定性和正確性，我們採用了 `pytest` 進行單元測試。使用者可以自行運行測試，以驗證所使用功能的正確性。

運行測試方法如下：

```bash
python -m pytest tests
```

## CI/CD

DocsaidKit 遵循持續集成（CI）和持續交付（CD）的最佳實踐。我們在每次代碼提交時，都會自動運行測試和構建流程，確保代碼的質量和穩定性。

目前，我們使用 [GitHub Actions](https://github.com/features/actions) 作為 CI/CD 工具。每當有新的代碼提交或合併請求時，GitHub Actions 會自動觸發，執行預定義的工作流程。

### 持續集成 (CI)

- 自動代碼風格檢查
- 單元測試
- 整合測試

### 持續交付 (CD)

- 自動構建和打包
- 自動發布到 PyPI 和其他平台

---

希望以上說明可以幫助使用者更加熟悉 DocsaidKit。如果遇到任何問題或需要進一步的支持，請查看我們的 [GitHub 存儲庫](https://github.com/DocsaidLab/DocsaidKit) 。
