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

## 快速開始

詳細功能和使用方法請參考 [**DocsaidKit Documents**](https://docsaid.org/docs/docsaidkit/intro/)。

## 測試

為了確保 DocsaidKit 功能的穩定性和正確性，我們使用 `pytest` 進行單元測試。

用戶可以自行運行測試以驗證所使用功能的準確性。

運行測試的方法如下：

```bash
python -m pytest tests
```
