# DocsaidKit

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/DocsaidLab/DocsaidKit/releases"><img src="https://img.shields.io/github/v/release/DocsaidLab/DocsaidKit?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
    <a href="https://doi.org/10.5281/zenodo.10438676"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.10438676.svg" alt="DOI"></a>
</p>

## Introduction

**DocsaidKit** is a Python deep learning toolbox, serving as a core Python package developed in-house.

Designed and written collaboratively by our internal developers, it supports deep learning and computer vision projects and research. The goal of this toolkit is to provide a unified workflow for developers within our organization, simplifying model development and deployment.

DocsaidKit is structured into the following main modules:

- **Vision**: This module includes computer vision-related functions, like image and video processing.

- **Structures**: A module for handling structured data such as Bounding Boxes.

- **ONNXEngine**: Offers functionality for ONNX inference, supporting ONNX format models.

- **Torch**: Related to PyTorch, containing neural network architectures, optimizers, etc.

- **Utils**: Provides various utility functions, including file processing and time handling.

- **Tests**: This directory contains test files to verify the functionality of DocsaidKit.

Our aim with DocsaidKit is to streamline the development process in deep learning and computer vision, providing a clear structure and modular design for ease of finding and extending functionalities.

## Installation Instructions

Before starting the installation of DocsaidKit, ensure you have met the following prerequisites:

### Prerequisites

1. **Python Version**:

   - Ensure Python 3.8 or above is installed on your system.

2. **Dependency Packages**:

   - **Ubuntu**:

     ```bash
     sudo apt install libturbojpeg exiftool ffmpeg libheif-dev
     ```

   - **MacOS**:

     ```bash
     # Install using brew
     brew install jpeg-turbo exiftool ffmpeg libheif
     ```

3. **Installation of pdf2image**:

   - Refer to this [page](https://github.com/Belval/pdf2image) for installation guidance.

### Installation Methods

After meeting the prerequisites, you can choose from the following installation methods:

#### General Method:

Note: This package is only available through our internal Pypi service. External users should clone via git and install using setup.py.

1. **Install via git clone and setup.py (Recommended)**:

   - Basic installation:

     ```bash
     pip install wheel
     python setup.py bdist_wheel
     pip install dist/DocsaidKit-${version}-none-any.whl
     ```

   - Installation for **training environments**:

     ```bash
     pip install wheel
     python setup.py bdist_wheel
     pip install "dist/DocsaidKit-${version}-none-any.whl[torch]"
     ```

2. **Install via git clone and docker (For stable environment needs)**:

   ```bash
   bash docker/build.bash
   ```

   For usage, refer to: [Docker](./docker/README.md)

3. **Through PyPi (Internal Developers)**:

   - Basic installation:

     ```bash
     tgt_ip=192.168.xxx.xxx
     pip install --trusted-host $tgt_ip \
       --index-url http://$tgt_ip:8080/simple/ \
       DocsaidKit==$version
     ```

   - For **development environments** requiring PyTorch and related packages:

     ```bash
     tgt_ip=192.168.xxx.xxx
     pip install --trusted-host $tgt_ip \
       --index-url http://$tgt_ip:8080/simple/ \
       "DocsaidKit[torch]==$version"
     ```

**Note:**

Set up a local Pypi server and install from there. In case you need to modify your PIP configuration, please be aware that some configuration files may have a priority order. Here are the following files that may exists in your machine by order of priority:

   - [Priority 1] Site level configuration files

      1. `/home/user/.pyenv/versions/3.x.x/envs/envs_name/pip.conf`

   - [Priority 2] User level configuration files

      1. `/home/user/.config/pip/pip.conf`
      2. `/home/user/.pip/pip.conf`

   - [Priority 3] Global level configuration files

      1. `/etc/pip.conf`
      2. `/etc/xdg/pip/pip.conf`

And the content of the configuration file should be like this:

```bash
[global]
no-cache-dir = true
index-url = your_pypi_server_url
trusted-host = your_pypi_server_ip
```

#### MacOS-Arm64

Due to constraints with MacOS-Arm64, we recommend installation via conda.

1. **Install Environment Manager (e.g., conda)**:

   - Download the conda installation script:

     ```bash
     wget https://github.com/conda-forge/miniforge#download
     ```

   - Run the installation script:

     ```bash
     bash Miniforge3-MacOSX-arm64.sh
     ```

2. **Create and Activate Environment**:

   - Create a new environment:

     ```bash
     conda create -n DocsaidKit python=3.8
     ```

   - Activate the environment:

     ```bash
     conda activate DocsaidKit
     ```

3. **Install DocsaidKit**:

   - After activating the environment, install DocsaidKit with:

     ```bash
     pip install wheel setuptools
     python setup.py bdist_wheel
     pip install dist/*.whl
     ```

For any questions or further clarifications, please consult our official documentation or contact us.

## Usage

### Overview

The use of DocsaidKit is aimed at providing researchers and developers with a simple interface for projects and research in the fields of deep learning and computer vision.

Here, we introduce how to use DocsaidKit, along with some basic usage examples.

### Structure

The `structures` module is a crucial component of DocsaidKit, used for representing and handling standardized frameworks like Bounding Boxes.

For detailed information, refer to [Structure](./docs/structure.md).

### Vision

The `vision` module encompasses computer vision-related functionalities, such as image processing, visual geometry, and visual effects.

For more details, see [Vision](./docs/vision.md).

### ONNXEngine

The `onnxengine` module provides functionality for ONNX inference, supporting ONNX format models.

For more information, refer to [ONNXEngine](./docs/onnxengine.md).

### Pytorch

The `torch` module is related to PyTorch, containing neural network structures, optimizers, and more.

For a detailed description, see [Pytorch](./docs/pytorch.md).

### Others

The `utils` module offers various utility functions, including file and time processing.

For more details, refer to [Utils](./docs/utils.md).

## Pytest Testing

To ensure the stability and correctness of DocsaidKit's functionalities, we use `pytest` for unit testing.

Users can run tests themselves to verify the accuracy of the functionalities used.

The method to run tests is as follows:

```bash
python -m pytest tests
```

## CI/CD

The **DocsaidKit** project strictly adheres to best practices in Continuous Integration (CI) and Continuous Delivery (CD) to ensure code quality and project stability.

We utilize [GitHub Actions](https://github.com/features/actions) as our primary CI/CD tool, automating the code inspection, testing, and release processes.

### Continuous Integration (CI)

Our CI process ensures that all code submissions and merge requests meet preset quality standards.

Key steps in the CI process include:

1. **Code Checkout**:
   - Retrieve the latest code, ensuring testing and building are based on the most recent changes.

2. **Environment Setup**:
   - Set up the running environment based on required Python versions and operating systems.

3. **Dependency Installation**:
   - Install necessary system dependencies and Python packages in preparation for subsequent testing and building steps.

4. **Code Quality Checks**:
   - Conduct static analysis of the code using `pylint` for quality and style checks.

5. **Unit and Integration Testing**:
   - Run unit and integration tests using `pytest` to ensure code functionality and stability.
   - Generate test coverage reports and feedback coverage information to the development team via GitHub Actions.

### Continuous Delivery (CD)

Our CD process aims to automate the build and release process, ensuring efficient and accurate delivery of new versions to users.

Key steps in the CD process include:

1. **Version Number Update**:
   - Automatically update the code's version number based on the input version tag.

2. **Build and Packaging**:
   - Automatically build Python wheel packages for easy distribution and installation.
   - Use `twine` to upload the built wheel packages to a private repository.

3. **Create Release**:
   - Create a new release on GitHub, filling in version information and release notes.
   - Upload the built wheel package as an attachment to the release for user download.

4. **Clean Build Environment**:
   - Clean up temporary files and directories generated during the build, maintaining a tidy build environment.

---

We hope this explanation provides a deeper understanding of DocsaidKit.

For any issues or further support, feel free to visit our [GitHub](https://github.com/DocsaidLab/DocsaidKit).
