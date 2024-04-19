[English](./README.md) | **[中文](./README_tw.md)**

# DocsaidKit

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/DocsaidLab/DocsaidKit/releases"><img src="https://img.shields.io/github/v/release/DocsaidLab/DocsaidKit?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
</p>

## Introduction

The positioning of this project is an image processing and deep learning toolbox, mainly including the following parts:

- **Vision**: Includes functionalities related to computer vision, such as image and video processing.
- **Structures**: Modules for handling structured data, such as BoundingBox and Polygon.
- **ONNXEngine**: Provides functionality for ONNX inference, supporting ONNX format models.
- **Torch**: Related to PyTorch, including neural network architectures, optimizers, etc.
- **Utils**: For things that don't fit elsewhere.
- **Tests**: Test files used to validate the functionality of various functions.

## Quick Start

For detailed functionalities and usage methods, please refer to the [**DocsaidKit Documents**](https://docsaid.org/en/docsaidkit/intro).

## Testing

To ensure the stability and correctness of DocsaidKit functionalities, we use `pytest` for unit testing.

Users can run tests themselves to verify the accuracy of the functionalities they are using.

Here's how to run the tests:

```bash
python -m pytest tests
```
