# PyTorch Release 22.12

## 容器資訊

NVIDIA 的 PyTorch 容器映像，版本 22.12，現已在 NVIDIA GPU Cloud (NGC) 上提供。

### 內容

- **PyTorch**: 在預設的 Python 環境中預先構建和安裝的版本 (`/usr/local/lib/python3.8/dist-packages/torch`)。完整原始碼位於 `/opt/pytorch`。
- **操作系統**: Ubuntu 20.04
- **Python 版本**: Python 3.8
- **相依性**:
  - NVIDIA CUDA 11.8.0
  - NVIDIA cuBLAS 11.11.3.6
  - NVIDIA cuDNN 8.7.0.84
  - NVIDIA NCCL 2.15.5 (針對 NVIDIA NVLink 優化)
  - NVIDIA RAPIDS 22.10.01 (x86 函式庫: cudf, xgboost, rmm, cuml, cugraph)
  - Apex
  - rdma-core 36.0
  - NVIDIA HPC-X 2.13
  - OpenMPI 4.1.4+
  - GDRCopy 2.3
  - TensorBoard 2.9.0
  - Nsight Compute 2022.3.0.0
  - Nsight Systems 2022.4.2.1
  - NVIDIA TensorRT 8.5.1
  - Torch-TensorRT 1.1.0a0 (預覽)
  - NVIDIA DALI 1.20.0
  - MAGMA 2.6.2
  - JupyterLab 2.3.2 (包含 Jupyter-TensorBoard)
  - TransformerEngine 0.3.0

## 驅動程式需求

- **基本需求**: NVIDIA 驅動程式發布 520 或更新版本。
- **資料中心 GPU 例外**: 發布 450.51 (或 R450 更新版本), 470.57 (或 R470 更新版本), 510.47 (或 R510 更新版本), 或 515.65 (或 R515 更新版本)。
- **相容性提示**: 使用者應升級所有的 R418, R440 和 R460 驅動程式，以確保與 CUDA 11.8 的向前相容性。

## GPU 需求

支援 CUDA 計算能力 6.0 及更新版本，涵蓋 NVIDIA Pascal, Volta, Turing, Ampere 及 Hopper 架構。查看完整列表，請參考官方 CUDA GPU 列表和深度學習框架支援矩陣。

## 主要功能和增強功能

- PyTorch 容器映像版本基於 `1.14.0a0+410ce96`。
- **Transformer Engine**: 在 NVIDIA GPU 上加速 Transformer 模型，包括對 Hopper GPU 的 8 位浮點數 (FP8) 精度的支援。
- **DLProf**: 從版本 21.12 開始，DLProf 不再包含在內。使用者可以手動從 nvidia-pyindex 上的 pip wheel 安裝它。
- **Torch-TensorRT (1.1.0a0) 預覽**: 將 TensorRT 與 PyTorch 整合，使 TensorRT 的功能可以直接用 Python 和 C++ APIs 存取。
- **Arm SBSA 平台支援**: 從 22.05 發布開始提供。
- **PyProf 移除**: 從 21.06 版本開始，PyProf 將不再包含在 NVIDIA PyTorch 容器中。使用 DLProf 進行性能分析。
- **TensorCore 範例模型**: 不再在核心容器中提供。他們可以從 GitHub 或 NGC 獲得。
- **Python 環境**: 從 22.11 版本開始，miniforge 已被移除，所有 Python 套件都安裝在預設的 Python 環境中。對於依賴 Conda 特定套件的用戶，可能需要調整他們的設定或單獨安裝 Conda。

---

如需進一步的資訊或查詢，請參考官方文檔或聯繫支援團隊。
