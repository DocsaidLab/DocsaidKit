# Training environment

我們推薦使用 Docker 來建置模型訓練環境。

相關參考資料：


- 每個版本的細節，請查閱：[PyTorch Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)

- NVIDIA runtime 前準備，請參考：[Installation (Native GPU Support)](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support)#usage)

- NVIDIA Toolkit 安裝方式，請參考：[Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Build Docker Image

我們提供了一個 Dockerfile 來建置模型訓練環境，請參考：[Dockerfile](./Dockerfile)

使用方式如下：

```bash
cd DocsaidKit
bash docker/build.bash
```

### 文件說明

1. **基礎映像**：使用 NVIDIA 官方提供的 PyTorch 映像 "nvcr.io/nvidia/pytorch:22.12-py3" 作為基礎。

2. **環境變數設置**：
    - `DEBIAN_FRONTEND=noninteractive`：確保 apt 命令運行時不會有任何用戶交互。
    - `PYTHONDONTWRITEBYTECODE=1`：防止 Python 創建 `.pyc` 字節碼文件。
    - `MPLCONFIGDIR` 和 `TRANSFORMERS_CACHE`：為 Matplotlib 和 Transformers 建立配置和緩存目錄。

3. **安裝套件**：安裝了各種音頻、視頻和圖像處理相關的庫和工具，例如 ffmpeg、exiftool、libjpeg、opencv 等。

4. **Python 套件**：
    - 升級 setuptools、pip 和 wheel。
    - 安裝其他 Python 套件，如 tqdm、colored、ipython、tensorboard 等。
    - 使用指定的版本安裝 opencv-python。

5. **安裝 docsaidkit**：這是一個內部使用的 Python 套件。請注意，安裝此套件時使用了一個特定的 URL 和密碼，這需要您洽詢管理員。

6. **工作目錄**：預設的工作目錄被設定為 `/code`。

### 注意事項

1. 請確保在運行 Docker 映像建構過程時擁有網絡連接，以確保所有套件和工具能夠正確下載和安裝。

2. 使用此 Dockerfile 之前，請確認您有 NVIDIA Docker 支持，因為此 Dockerfile 使用了 NVIDIA 官方的 PyTorch 映像。

3. 在使用 docsaidkit 的安裝命令時，您需要確保能夠訪問提供的 IP 地址（例如 192.168.0.105）且該 IP 提供 PYPI 服務。

4. 若您需要使用其他版本的 Python 套件或工具，請在 Dockerfile 中進行適當修改。

5. 確保有足夠的磁盤空間來下載和建構 Docker 映像。

6. 執行時，需要確認基底映像內容：`FROM nvcr.io/nvidia/pytorch:xx.yy-py3` 所採用的 OpenCV 的版本，由於我們的套件會自動安裝 opencv-python 的最新版本，因此，若基底映像內容的 OpenCV 版本較舊，則會造成版本衝突，因此，必須在最後根據基底版本重新安裝：`RUN pip install opencv-python==a.b.c.xx`。

## Run Docker Image

若您已經成功建構了 Docker 映像，接下來您可以運行該映像：

### 基本運行指令

我們提供了一個 bash 腳本以便於您運行 Docker 映像，內容如下：

```bash
#!/bin/bash
docker run -u $(id -u):$(id -g) \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -it --rm docsaid_training_base_image bash
```

### 腳本說明

- `-u $(id -u):$(id -g)`：以當前用戶的 UID 和 GID 運行容器，這確保在容器內部創建的任何文件在原本主機上具有可以操作的權限。
- `--gpus all`：使用所有可用的 GPU。
- `--shm-size=64g`：設定共享內存大小為 64 GB，對於某些深度學習工作負載可能是必需的。
- `--ipc=host --net=host`：容器使用主機的 IPC 命名空間和網絡堆棧。
- `--cpuset-cpus="0-31"`：限制容器只使用 CPU 0 到 31。
