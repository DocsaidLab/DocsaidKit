[English](./README.md) | **[中文](./README_tw.md)**

# DocsaidKit

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/DocsaidLab/DocsaidKit/releases"><img src="https://img.shields.io/github/v/release/DocsaidLab/DocsaidKit?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
</p>

## Introduction

This project is a toolbox for image processing and deep learning, primarily consisting of the following components:

- **Vision**: Functions related to computer vision, such as image and video processing.
- **Structures**: Modules for handling structured data, such as BoundingBox and Polygon.
- **ONNXEngine**: Provides ONNX inference capabilities, supporting ONNX format models.
- **Torch**: Related to PyTorch, including neural network architectures, optimizers, etc.
- **Utils**: Miscellaneous utilities that do not fit into other categories.
- **Tests**: Test files for verifying the functionality of various functions.

## Documentation

For installation and usage instructions, please refer to the [**DocsaidKit Documents**](https://docsaid.org/docs/docsaidkit/intro/).

Here, you will find all the detailed information about this project.

## Installation

Before installing DocsaidKit, ensure your system meets the following requirements:

### Python Version

- Ensure Python 3.8 or higher is installed on your system.

### Dependencies

Install the required dependencies based on your operating system.

- **Ubuntu**

  Open the terminal and run the following commands to install dependencies:

  ```bash
  sudo apt install libturbojpeg exiftool ffmpeg libheif-dev
  ```

- **MacOS**

  Use brew to install dependencies:

  ```bash
  brew install jpeg-turbo exiftool ffmpeg libheif
  ```

### pdf2image Dependencies

pdf2image is a Python module for converting PDF documents into images.

Follow these instructions to install it based on your operating system:

- For detailed installation instructions, refer to the [**pdf2image**](https://github.com/Belval/pdf2image) project page.

- MacOS: Mac users need to install poppler. Install it via Brew:

  ```bash
  brew install poppler
  ```

- Linux: Most Linux distributions come with `pdftoppm` and `pdftocairo` pre-installed.

  If not, install poppler-utils via your package manager:

  ```bash
  sudo apt install poppler-utils
  ```

### Installation via git clone

1. Clone the repository:

   ```bash
   git clone https://github.com/DocsaidLab/DocsaidKit.git
   ```

2. Install the wheel package:

   ```bash
   pip install wheel
   ```

3. Build the wheel file:

   ```bash
   cd DocsaidKit
   python setup.py bdist_wheel
   ```

4. Install the built wheel package:

   ```bash
   pip install dist/docsaidkit-*-py3-none-any.whl
   ```

   To install the version that supports PyTorch:

   ```bash
   pip install "dist/docsaidKit-${version}-none-any.whl[torch]"
   ```

### Installation via Docker (Recommended)

Install via Docker to ensure environment consistency.

Use the following commands:

```bash
cd DocsaidKit
bash docker/build.bash
```

Once completed, run your commands within Docker:

```bash
docker run -v ${PWD}:/code -it docsaid_training_base_image your_scripts.py
```

For the specifics of the build file, refer to: [**Dockerfile**](https://github.com/DocsaidLab/DocsaidKit/blob/main/docker/Dockerfile)

## Testing

To ensure the stability and accuracy of DocsaidKit, we use `pytest` for unit testing.

Users can run the tests themselves to verify the accuracy of the functionalities they are using.

To run the tests:

```bash
python -m pytest tests
```
