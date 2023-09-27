import platform
import socket
import subprocess

import psutil
import requests

__all__ = [
    "get_package_versions", "get_gpu_cuda_versions", "get_system_info",
    "get_cpu_info", "get_external_ip"
]


def get_package_versions():
    """
    Get versions of commonly used packages in deep learning and data science.

    Returns:
        dict: Dictionary containing versions of installed packages.
    """
    versions_info = {}

    # PyTorch
    try:
        import torch
        versions_info["PyTorch Version"] = torch.__version__
    except Exception as e:
        versions_info["PyTorch Error"] = str(e)

    # PyTorch Lightning
    try:
        import pytorch_lightning as pl
        versions_info["PyTorch Lightning Version"] = pl.__version__
    except Exception as e:
        versions_info["PyTorch Lightning Error"] = str(e)

    # TensorFlow
    try:
        import tensorflow as tf
        versions_info["TensorFlow Version"] = tf.__version__
    except Exception as e:
        versions_info["TensorFlow Error"] = str(e)

    # Keras
    try:
        import keras
        versions_info["Keras Version"] = keras.__version__
    except Exception as e:
        versions_info["Keras Error"] = str(e)

    # NumPy
    try:
        import numpy as np
        versions_info["NumPy Version"] = np.__version__
    except Exception as e:
        versions_info["NumPy Error"] = str(e)

    # Pandas
    try:
        import pandas as pd
        versions_info["Pandas Version"] = pd.__version__
    except Exception as e:
        versions_info["Pandas Error"] = str(e)

    # Scikit-learn
    try:
        import sklearn
        versions_info["Scikit-learn Version"] = sklearn.__version__
    except Exception as e:
        versions_info["Scikit-learn Error"] = str(e)

    # OpenCV
    try:
        import cv2
        versions_info["OpenCV Version"] = cv2.__version__
    except Exception as e:
        versions_info["OpenCV Error"] = str(e)

    # ... and so on for any other packages you're interested in

    return versions_info


def get_gpu_cuda_versions():
    """
    Get GPU and CUDA versions using popular Python libraries.

    Returns:
        dict: Dictionary containing CUDA and GPU driver versions.
    """

    cuda_version = None

    # Attempt to retrieve CUDA version using PyTorch
    try:
        import torch
        cuda_version = torch.version.cuda
    except ImportError:
        pass

    # If not retrieved via PyTorch, try using TensorFlow
    if not cuda_version:
        try:
            import tensorflow as tf
            cuda_version = tf.version.COMPILER_VERSION
        except ImportError:
            pass

    # If still not retrieved, try using CuPy
    if not cuda_version:
        try:
            import cupy
            cuda_version = cupy.cuda.runtime.runtimeGetVersion()
        except ImportError:
            cuda_version = "Error: None of PyTorch, TensorFlow, or CuPy are installed."

    # Try to get Nvidia driver version using nvidia-smi command
    try:
        smi_output = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=driver_version',
            '--format=csv,noheader,nounits'
        ]).decode('utf-8').strip()
        nvidia_driver_version = smi_output.split('\n')[0]
    except Exception as e:
        nvidia_driver_version = f"Error getting NVIDIA driver version: {e}"

    return {
        "CUDA Version": cuda_version,
        "NVIDIA Driver Version": nvidia_driver_version
    }


def get_cpu_info():
    """
    Retrieve the CPU model name based on the platform.

    Returns:
        str: CPU model name or "N/A" if not found.
    """
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        # For macOS
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command, shell=True).strip().decode()
    elif platform.system() == "Linux":
        # For Linux
        command = "cat /proc/cpuinfo | grep 'model name' | uniq"
        return subprocess.check_output(command, shell=True).strip().decode().split(":")[1].strip()
    else:
        return "N/A"


def get_external_ip():
    try:
        response = requests.get('https://httpbin.org/ip')
        return response.json()['origin']
    except Exception as e:
        return f"Error obtaining IP: {e}"


def get_system_info():
    """
    Fetch system information like OS version, CPU info, RAM, Disk usage, etc.

    Returns:
        dict: Dictionary containing system information.
    """
    info = {
        "OS Version": platform.platform(),
        "CPU Model": get_cpu_info(),
        "Physical CPU Cores": psutil.cpu_count(logical=False),
        "Logical CPU Cores (incl. hyper-threading)": psutil.cpu_count(logical=True),
        "Total RAM (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "Available RAM (GB)": round(psutil.virtual_memory().available / (1024 ** 3), 2),
        "Disk Total (GB)": round(psutil.disk_usage('/').total / (1024 ** 3), 2),
        "Disk Used (GB)": round(psutil.disk_usage('/').used / (1024 ** 3), 2),
        "Disk Free (GB)": round(psutil.disk_usage('/').free / (1024 ** 3), 2)
    }

    # Try to fetch GPU information using nvidia-smi command
    try:
        gpu_info = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits']
        ).decode('utf-8').strip()
        info["GPU Info"] = gpu_info
    except Exception:
        info["GPU Info"] = "N/A or Error"

    # Get network information
    addrs = psutil.net_if_addrs()
    info["IPV4 Address"] = [
        addr.address for addr in addrs.get('enp5s0', []) if addr.family == socket.AF_INET
    ]

    info["IPV4 Address (External)"] = get_external_ip()

    # Determine platform and choose correct address family for MAC
    if hasattr(socket, 'AF_LINK'):
        AF_LINK = socket.AF_LINK
    elif hasattr(psutil, 'AF_LINK'):
        AF_LINK = psutil.AF_LINK
    else:
        raise Exception(
            "Cannot determine the correct AF_LINK value for this platform.")

    info["MAC Address"] = [
        addr.address for addr in addrs.get('enp5s0', []) if addr.family == AF_LINK
    ]

    return info
