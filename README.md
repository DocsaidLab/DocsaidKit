# DocsaidKit

Welcome to the installation guide for DocsaidKit, a comprehensive Python library with a wide range of computational modules including structure processing, visualization tools, video processing, and more. This package is organized into several key modules: engine, structures, utils, and vision.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [MacOS](#MacOS)

## Prerequisites

Before installing DocsaidKit, please ensure that you have the following software already installed on your system:

- Python 3.8 or above

Note: Depending on the specific functionalities of DocsaidKit you intend to use, additional libraries may be required. For instance, if you plan to use image or video processing tools, you might need to install libraries like OpenCV.

## Installation

To install DocsaidKit, follow the steps below:

### MacOS

If you don't have an environment manager installed on your system, you can use conda as follows:

1. Install Environment Manager (e.g., conda)

- Download the conda installer script:

  ```bash
  wget https://github.com/conda-forge/miniforge#download
  ```

- Run the installer script:

  ```bash
  bash Miniforge3-MacOSX-arm64.sh
  ```

2. Create and activate environment

- Create a new environment:

  ```bash
  conda create -n DocsaidKit python=3.8
  ```

- Activate the environment:

  ```bash
  conda activate DocsaidKit
  ```

3. Install DocsaidKit

After activating the environment, you can install DocsaidKit using the following commands:

```bash
pip install wheel setuptools
python setup.py bdist_wheel
pip install dist/*.whl
```

If you need specific functionalities that require additional dependencies, you can install them now. For example, if you plan to work with machine learning-related tools, you can install PyTorch and other related libraries as follows:

```bash
pip install torch torchvision
```

For OpenMMLab support:

```bash
pip install openmim
mim install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html
```

Install:

```bash
pip install wheel setuptools
python setup.py bdist_wheel
pip install dist/*.whl
```
