from typing import Dict, Union

import torch
from calflops import calculate_flops
from ptflops import get_model_complexity_info

from .cpuinfo import cpuinfo

__all__ = ['get_model_complexity_info',
           'get_cpu_gflops', 'get_meta_info', 'calculate_flops']


def get_cpu_gflops(one_cpu_core: bool = True) -> float:
    _cpuinfo = cpuinfo()
    ghz = float(_cpuinfo.info[0]['cpu MHz']) * 10e-3
    core = 1 if one_cpu_core else int(_cpuinfo.info[0]['cpu cores'])
    gflops = ghz * core * 10e9
    return gflops


def get_meta_info(macs: float, params: int, one_cpu_core: bool = True) -> dict:
    return {
        'Params(M)': f"{params/1e6:.3f}",
        'MACs(G)': f"{macs/1e9:.3f}",
        'FLOPs(G)': f"{(macs * 2)/1e9:.3f}",
        'ModelSize_FP32 (MB)': f"{params * 4 / 1e6:.3f}",
        'CPU infos': {
            'cpu_model_name': cpuinfo().info[0]['model name'],
            'cpu_cores': cpuinfo().info[0]['cpu cores'],
            'infer_time (ms) (*rough estimate*)': f"{(macs * 2) * 1000 / get_cpu_gflops(one_cpu_core):.3f}",
        }
    }


def profile_model(
    model: Union[torch.nn.Module, str],
    input_shape: tuple = (1, 3, 224, 224),
    output_as_string: bool = False,
    output_precision: int = 4,
    print_detailed: bool = False,
    features_only: bool = True,
    one_cpu_core: bool = True
) -> Dict[str, str]:
    """
    Profile a model to get its meta data.

    Args:
        model (Union[torch.nn.Module, str]): Model to be profiled. If a string is given, it will be treated as the model name by timm library.
        input_shape (tuple): Input shape of the model. Default: (1, 3, 224, 224).
        output_as_string (bool): Whether to output the results as string. Default: False.
        output_precision (int): Precision of the output. Default: 4.
        print_detailed (bool): Whether to print detailed information. Default: False.
        features_only (bool): Whether to calculate only the features. Default: True.
        one_cpu_core (bool): Whether to use only one CPU core. Default: True.

    Returns:
        Dict[str, str]: Meta data of the model.
    """

    if isinstance(model, str):

        import timm

        model = timm.create_model(
            model,
            pretrained=False,
            features_only=features_only
        )

    _, macs, params = calculate_flops(
        model,
        input_shape=input_shape,
        output_as_string=output_as_string,
        output_precision=output_precision,
        print_detailed=print_detailed
    )

    meta_data = get_meta_info(macs, params, one_cpu_core=one_cpu_core)

    return meta_data
