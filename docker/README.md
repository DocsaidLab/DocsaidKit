# Docsaid Training Base Image

## Overview

此 Docker 映像專為機器學習和深度學習模型訓練環境設計，提供 NVIDIA PyTorch 映像為基礎，並集成了多種音訊、視訊和圖像處理工具。映像中包含了豐富的 Python 套件，適用於各種數據處理和模型訓練任務。

相關參考資料：

- 每個版本的細節，請查閱：[PyTorch Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)

- NVIDIA runtime 前準備，請參考：[Installation (Native GPU Support)](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support)#usage)

- NVIDIA Toolkit 安裝方式，請參考：[Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

---

選擇 pytorch 映像時，必須同時參考 onnxruntime 的版本，以確保兩者相容。

相關內容可參考：[ONNX Runtime Release Notes](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)

舉例來說：

當我們選擇使用 pytorch:23.11 版本時，其對應的 cuda 版本為 12.3.0。因此我們將無法在此映像中使用 onnxruntime-gpu 版本，因為即使是目前最新的 v1.16 版，它需要的 cuda 版本為 11.8。若希望能使用 onnxruntime-gpu，則必須選擇 pytorch:22.12 版本，其對應的 cuda 版本為 11.8.0。

## 建置 Docker 映像

### 前置要求

- 確保您的系統已安裝 Docker。
- 確保您的系統支持 NVIDIA Docker，並已安裝 NVIDIA Container Toolkit。
- 確保您有穩定的網絡連接以便於下載和安裝必要的套件。

### 基礎映像

我們使用 NVIDIA 官方提供的 PyTorch 映像 `nvcr.io/nvidia/pytorch:22.12-py3` 作為基礎，它提供了一個高效、靈活的深度學習環境。

### Dockerfile 說明

- **環境變數設置**：設置了多個環境變數以優化映像的運行。
- **安裝套件**：包括音訊、視訊和圖像處理相關的庫和工具，以及必要的 Python 套件。
- **Python 套件**：包括用於訓練的工具和庫，如 `tqdm`, `Pillow`, `tensorboard` 等。
- **工作目錄**：設置 `/code` 為預設工作目錄。

### 建置指令

在 DocsaidKit 目錄中，執行以下命令來建置 Docker 映像：

```bash
cd DocsaidKit
bash docker/build.bash
```

## 運行 Docker 映像

建置成功後，您可以使用以下指令來運行映像：

### 基本運行指令

```bash
#!/bin/bash
docker run \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -it --rm docsaid_training_base_image bash
```

### 腳本說明

- `--gpus all`：分配所有可用 GPU 給 Docker 容器。
- `--shm-size=64g`：設定共享內存大小，適用於大型深度學習任務。
- `--ipc=host --net=host`：容器將使用主機的 IPC 和網絡設置。
- `--cpuset-cpus="0-31"`：限制 CPU 使用，可根據需求調整。

### 注意事項

- 請確保 Docker 映像運行時，主機具有足夠的資源（如內存和儲存空間）。
- 若有版本衝突或特定需求，可自行調整 Dockerfile 中的安裝套件和版本。
