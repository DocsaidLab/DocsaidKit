# PyTorch Release 23.08

## 容器資訊

NVIDIA 的 PyTorch 容器映像，版本 23.08，現已在 NVIDIA GPU Cloud (NGC) 上提供。

### 內容

- **PyTorch**: 在預設的 Python 環境中預先構建和安裝的版本 (`/usr/local/lib/python3.10/dist-packages/torch`)。完整原始碼位於 `/opt/pytorch`。
- **操作系統**: Ubuntu 22.04
- **Python 版本**: Python 3.10
- **相依性**:
  - NVIDIA CUDA 12.2.1
  - NVIDIA cuBLAS 12.2.5.1
  - NVIDIA cuDNN 8.9.4
  - NVIDIA NCCL 2.18.3
  - NVIDIA RAPIDS 23.06
  - Apex
  - rdma-core 39.0
  - NVIDIA HPC-X 2.15
  - OpenMPI 4.1.4+
  - GDRCopy 2.3
  - TensorBoard 2.9.0
  - Nsight Compute 2023.2.1.3
  - Nsight Systems 2023.2.3.1001
  - NVIDIA TensorRT 8.6.1.6
  - Torch-TensorRT 2.0.0.dev0
  - NVIDIA DALI 1.28.0
  - MAGMA 2.6.2
  - JupyterLab 2.3.2 (包含 Jupyter-TensorBoard)
  - TransformerEngine 0.11.0++3f01b4f
  - PyTorch 量化輪 2.1.2

## 驅動程式需求

- **基本需求**: NVIDIA 驅動程式發布 535 或更新版本。
- **資料中心 GPU 例外**: 發布 450.51 (或 R450 更新版本), 470.57 (或 R470 更新版本), 510.47 (或 R510 更新版本), 或 515.65 (或 R515 更新版本)。

## GPU 需求

支援 CUDA 計算能力 6.0 及更新版本，涵蓋 NVIDIA Pascal, Volta, Turing, Ampere 及 Hopper 架構。查看完整列表，請參考官方 CUDA GPU 列表和深度學習框架支援矩陣。

## 主要功能和增強功能

- PyTorch 容器映像版本基於 `2.1.0a0+29c30b1`。
- 從 23.06 版本開始，NVIDIA 優化的深度學習框架容器不再在 Pascal GPU 架構上進行測試。
- Transformer Engine 是一個用於加速 NVIDIA GPU 上的 Transformer 模型的庫。它支持 Hopper GPU上的8位浮點數（FP8）精度，提供了更好的訓練和推理性能，以及更低的內存使用率。
- 現在包括 Torch-TensorRT (1.4.0dev0) 的預覽。
- Torch-TRT 是 PyTorch 的 TensorRT 集成，並將 TensorRT 的功能直接帶到 Python 和 C++的 API 中。
- 從 22.05 版本開始，PyTorch 容器適用於 Arm SBSA 平台。
- 19.11 及更高版本的深度學習框架容器包括對 Singularity v3.0 的實驗性支持。
- 從 22.11 的 PyTorch NGC 容器開始，miniforge 被移除，所有 Python 包都安裝在默認的 Python 環境中。

---

如需進一步的資訊或查詢，請參考官方文檔或聯繫支援團隊。
