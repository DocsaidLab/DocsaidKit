# Docsaid Training Base Image

## Overview

This Docker image is specifically designed for machine learning and deep learning model training environments. It is based on the NVIDIA PyTorch image and integrates a variety of audio, video, and image processing tools. The image contains a rich set of Python packages suitable for various data processing and model training tasks.

Related Reference Material:

- For details of each version, please refer to: [PyTorch Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)

- For NVIDIA runtime preparation, please refer to: [Installation (Native GPU Support)](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support)#usage)

- For NVIDIA Toolkit installation methods, please refer to: [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

---

When selecting a PyTorch image, it is necessary to also consider the version of onnxruntime to ensure compatibility.

For more information, refer to: [ONNX Runtime Release Notes](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)

For example:

When we choose to use the pytorch:23.11 version, its corresponding CUDA version is 12.3.0. Therefore, we cannot use the onnxruntime-gpu version in this image, as even the latest version, v1.16, requires CUDA version 11.8. If you wish to use onnxruntime-gpu, you must choose the pytorch:22.12 version, which corresponds to CUDA version 11.8.0.

## Building the Docker Image

### Prerequisites

- Ensure your system has Docker installed.
- Ensure your system supports NVIDIA Docker and has [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.
- Ensure you have a stable internet connection for downloading and installing necessary packages.

### Base Image

We use the NVIDIA official PyTorch image `nvcr.io/nvidia/pytorch:22.12-py3` as the base, which provides an efficient and flexible deep learning environment.

### Dockerfile Description

- **Environment Variable Setup**: Multiple environment variables are set to optimize the image's performance.
- **Package Installation**: Includes libraries and tools for audio, video, and image processing, as well as necessary Python packages.
- **Python Packages**: Includes tools and libraries for training, such as `tqdm`, `Pillow`, `tensorboard`, etc.
- **Working Directory**: Sets `/code` as the default working directory.

### Build Instructions

In the DocsaidKit directory, execute the following command to build the Docker image:

```bash
cd DocsaidKit
bash docker/build.bash
```

## Running the Docker Image

After successful build, you can use the following command to run the image:

### Basic Run Command

```bash
#!/bin/bash
docker run \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -it --rm docsaid_training_base_image bash
```

### Script Description

- `--gpus all`: Allocates all available GPUs to the Docker container.
- `--shm-size=64g`: Sets the shared memory size, suitable for large-scale deep learning tasks.
- `--ipc=host --net=host`: The container will use the host's IPC and network settings.
- `--cpuset-cpus="0-31"`: Restricts CPU usage, can be adjusted according to needs.

### Notes

- Ensure that the host has sufficient resources (such as memory and storage space) when running the Docker image.
- If there are version conflicts or specific requirements, you can adjust the installation packages and versions in the Dockerfile as needed.
