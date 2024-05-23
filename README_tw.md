[English](./README.md) | **[中文](./README_tw.md)**

# DocsaidKit

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/DocsaidLab/DocsaidKit/releases"><img src="https://img.shields.io/github/v/release/DocsaidLab/DocsaidKit?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
</p>

## 介紹

本專案的定位是影像處理和深度學習工具箱，主要包括以下幾個部分：

- **Vision**：包括與電腦視覺相關的功能，如圖像和影片處理。
- **Structures**：用於處理結構化數據的模塊，例如 BoundingBox 和 Polygon。
- **ONNXEngine**：提供 ONNX 推理的功能，支持 ONNX 格式模型。
- **Torch**：與 PyTorch 相關，包含神經網絡架構、優化器等。
- **Utils**：不知道該怎麼歸類的，就放在這裡。
- **Tests**：測試文件，用於驗證各類函數的功能。

## 技術文件

套件安裝和使用的方式，請參閱 [**DocsaidKit Documents**](https://docsaid.org/docs/category/docsaidkit)。

在那裡你可以找到所有有關本專案的詳細資訊。

## 安裝

在開始安裝 DocsaidKit 之前，請確保你的系統符合以下要求：

### Python 版本

- 確保系統已安裝 Python 3.8 或以上版本。

### 依賴套件

根據你的作業系統，安裝所需的依賴套件。

- **Ubuntu**

  開啟終端，執行以下命令安裝依賴：

  ```bash
  sudo apt install libturbojpeg exiftool ffmpeg libheif-dev
  ```

- **MacOS**

  使用 brew 安裝相依性：

  ```bash
  brew install jpeg-turbo exiftool ffmpeg libheif
  ```

### pdf2image 依賴套件

pdf2image 是一個 Python 模組，用於將 PDF 文件轉換為圖片。

根據你的作業系統，請遵循以下指示進行安裝：

- 或參考開源專案 [**pdf2image**](https://github.com/Belval/pdf2image) 相關頁面以取得安裝指南。

- MacOS：Mac 使用者需要安裝 poppler。透過 Brew 進行安裝：

  ```bash
  brew install poppler
  ```

- Linux：大多數 Linux 發行版已預裝 `pdftoppm` 和 `pdftocairo`。

  如果未安裝，請透過你的套件管理器安裝 poppler-utils。

  ```bash
  sudo apt install poppler-utils
  ```

### 透過 git clone 安裝

1. 下載本套件：

   ```bash
   git clone https://github.com/DocsaidLab/DocsaidKit.git
   ```

2. 安裝 wheel 套件：

   ```bash
   pip install wheel
   ```

3. 建構 wheel 檔案：

   ```bash
   cd DocsaidKit
   python setup.py bdist_wheel
   ```

4. 安裝建置的 wheel 套件：

   ```bash
   pip install dist/docsaidkit-*-py3-none-any.whl
   ```

   如果需要安裝支援 PyTorch 的版本：

   ```bash
   pip install "dist/docsaidKit-${version}-none-any.whl[torch]"
   ```

### 透過 docker 安裝（建議）

透過 docker 進行安裝，確保環境的一致性。

使用以下指令：

```bash
cd DocsaidKit
bash docker/build.bash
```

完成後，每次使用的時候就把指令包在 docker 裡面執行：

```bash
docker run -v ${PWD}:/code -it docsaid_training_base_image your_scripts.py
```

建置檔案的具體內容，請參考：[**Dockerfile**](https://github.com/DocsaidLab/DocsaidKit/blob/main/docker/Dockerfile)

## 測試

為了確保 DocsaidKit 功能的穩定性和正確性，我們使用 `pytest` 進行單元測試。

用戶可以自行運行測試以驗證所使用功能的準確性。

運行測試的方法如下：

```bash
python -m pytest tests
```
