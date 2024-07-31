# Ref from: https://github.com/sovrasov/flops-counter.pytorch
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
