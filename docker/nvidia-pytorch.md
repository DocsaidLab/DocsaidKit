# **NVIDIA PyTorch**:

1. **內容**: 容器鏡像中包含完整的 PyTorch 源代碼版本，位置於 `/opt/pytorch`。

2. **安裝與環境**: PyTorch 已預先構建並在容器的 Conda 預設環境 (`/opt/conda/lib/python3.8/site-packages/torch/`) 中安裝。

3. **優化與兼容性**: 容器專為 NVIDIA GPU 優化。內含的軟件已經過驗證，確保彼此之間的兼容性。

4. **主要 GPU 加速軟件**:
   - **基礎**: CUDA, cuBLAS
   - **深度學習專用**: NVIDIA cuDNN, NVIDIA NCCL (為 NVLink 優化)
   - **數據處理**: RAPIDS, NVIDIA Data Loading Library (DALI)
   - **模型優化與推理**: TensorRT, Torch-TensorRT

5. **使用便利性**: 使用者無需進行額外的安裝或編譯，可直接利用此容器加速深度學習工作流程。

## **運行NGC深度學習框架容器前的準備**

**一、簡介**：

在運行NGC深度學習框架容器之前，您的Docker®環境必須支援NVIDIA GPU。要運行容器，請根據“運行容器”部分發出適當的命令，並指定相應的註冊表、存儲庫和標籤。

**二、任務概述**：

在具有GPU支援的系統上，當您運行一個容器時，會發生以下情況：
1. Docker引擎將映像加載到運行該軟件的容器中。
2. 您可以通過包括額外的標誌和設置來定義容器的運行時資源。這些標誌和設置都描述在“運行容器”部分。
3. GPUs會被明確地為Docker容器定義（默認為所有GPUs，但可以使用NVIDIA_VISIBLE_DEVICES環境變數進行指定）。

**三、操作步驟**：

1. 發出您要的容器版本的命令。以下的命令假定您要拉取最新的容器：

    ```
    docker pull nvcr.io/nvidia/pytorch:23.08-py3
    ```

2. 打開命令提示符並粘貼上述命令。確保拉取成功後再繼續下一步。

3. 要運行容器映像，選擇以下模式之一：

   - **交互式**:

     - 若您使用Docker 19.03或更高版本，啟動容器的典型命令為：

        ```
        docker run --gpus all -it --rm -v local_dir:container_dir nvcr.io/nvidia/pytorch:<xx.xx>-py3
        ```

     - 若您使用Docker 19.02或更早版本，啟動容器的典型命令為：

        ```
        nvidia-docker run -it --rm -v local_dir:container_dir nvcr.io/nvidia/pytorch:<xx.xx>-py3
        ```

   - **非交互式**:

     - 若您使用Docker 19.03或更高版本：

        ```
        docker run --gpus all -it --rm -v local_dir:container_dir nvcr.io/nvidia/pytorch:<xx.xx>-py3 <command>
        ```

     - 若您使用Docker 19.02或更早版本：

        ```
        nvidia-docker run -it --rm -v local_dir:container_dir nvcr.io/nvidia/pytorch:<xx.xx>-py3 <command>
        ```

    **注意**：
    若您使用多進程的多線程數據加載器，則容器運行的默認共享內存段大小可能不夠。因此，您應該通過在命令行中增加以下命令之一來增加共享內存大小：`--ipc=host` 或 `--shm-size=<請求的內存大小>`。

**四、先決條件清單**：

1. 使用 GNU/Linux x86_64，且內核版本需 > 3.10
2. 安裝 Docker >= 19.03 版本
3. 配備 NVIDIA GPU，其架構需 > Fermi (2.1)
4. NVIDIA 驅動版本 ~= 361.93（在較早版本上未經測試）
5. 請注意，您的驅動版本可能會限制您的CUDA能力（請參閱CUDA要求）。

**五、安裝GPU支援**：

1. 確保已安裝NVIDIA驅動程序和您的發行版本所支援的Docker（參見先決條件）。
2. 根據此處的指示安裝您的發行版本的存儲庫。
3. 安裝 `nvidia-container-toolkit` 包：

   ```
   $ sudo apt-get install -y nvidia-container-toolkit
   $ sudo yum install -y nvidia-container-toolkit
   ```

**六、使用方法**：

NVIDIA運行時已與Docker CLI整合，GPU可以透過Docker CLI選項無縫訪問容器。

以下是一些示例：

- 啟動一個支援GPU的容器：

  ```
  $ docker run --gpus all nvidia/cuda nvidia-smi
  ```

- 在兩個GPU上啟動一個支援GPU的容器：

  ```
  $ docker run --gpus 2 nvidia/cuda nvidia-smi
  ```

- 在指定的GPU上啟動一個支援GPU的容器：

  ```
  $ docker run --gpus device=1,2 nvidia/cuda nvidia-smi
  $ docker run --gpus device=UUID-ABCDEF,1 nvidia/cuda nvidia-smi
  ```

- 為我的容器指定一個能力（例如：圖形、計算等）：

  ```
  $ docker run --gpus all,capabilities=utilities nvidia/cuda nvidia-smi
  ```

**七、非CUDA映像**：

設置NVIDIA_VISIBLE_DEVICES將為任何容器映像啟用GPU支援：

```
docker run --gpus all,capabilities=utilities --rm debian:stretch nvidia-smi
```

**八、Docker文件**：

如果在Dockerfile內設置了環境變數，則不需要在docker run命令行上設置它們。

例如，如果您正在創建自己的自定義CUDA容器，您應該使用以下內容：

```
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
```

這些環境變數已在我們推送到Docker Hub的官方映像中設置。

對於使用NVIDIA Video Codec SDK的Dockerfile，您應該使用：

```
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility
```
