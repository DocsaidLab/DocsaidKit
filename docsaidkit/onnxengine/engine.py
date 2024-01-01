from enum import Enum
from pathlib import Path
from typing import Any, Dict, Union

import colored
import numpy as np
import onnxruntime as ort

from ..enums import EnumCheckMixin
from .metadata import get_onnx_metadata


class Backend(EnumCheckMixin, Enum):
    cpu = 0
    cuda = 1


class ONNXEngine:

    def __init__(
        self,
        model_path: Union[str, Path],
        gpu_id: int = 0,
        backend: Union[str, int, Backend] = Backend.cpu,
        session_option: Dict[str, Any] = {},
        provider_option: Dict[str, Any] = {},
    ):
        """
        Initialize an ONNX model inference engine.

        Args:
            model_path (Union[str, Path]):
                Filename or serialized ONNX or ORT format model in a byte string.
            gpu_id (int, optional):
                GPU ID. Defaults to 0.
            backend (Union[str, int, Backend], optional):
                Backend. Defaults to Backend.cuda.
            session_option (Dict[str, Any], optional):
                Session options. Defaults to {}.
            provider_option (Dict[str, Any], optional):
                Provider options. Defaults to {}.
        """
        # setting device info
        backend = Backend.obj_to_enum(backend)
        self.device_id = 0 if backend.name == 'cpu' else gpu_id

        # setting provider options
        providers, provider_options = self._get_provider_info(
            backend, provider_option)

        # setting session options
        sess_options = self._get_session_info(session_option)

        # setting onnxruntime session
        model_path = str(model_path) if isinstance(
            model_path, Path) else model_path
        self.sess = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options,
        )

        # setting onnxruntime session info
        self.model_path = model_path
        self.metadata = get_onnx_metadata(model_path)
        self.providers = self.sess.get_providers()
        self.provider_options = self.sess.get_provider_options()

        self.input_infos = {
            x.name: {'shape': x.shape, 'dtype': x.type}
            for x in self.sess.get_inputs()
        }

        self.output_infos = {
            x.name: {'shape': x.shape, 'dtype': x.type}
            for x in self.sess.get_outputs()
        }

    def __call__(self, **xs) -> Dict[str, np.ndarray]:
        output_names = list(self.output_infos.keys())
        outs = self.sess.run(output_names, {k: v for k, v in xs.items()})
        outs = {k: v for k, v in zip(output_names, outs)}
        return outs

    def _get_session_info(
        self,
        session_option: Dict[str, Any] = {},
    ) -> ort.SessionOptions:
        """
        Ref: https://onnxruntime.ai/docs/api/python/api_summary.html#sessionoptions
        """
        sess_opt = ort.SessionOptions()
        session_option_default = {
            'graph_optimization_level': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            'log_severity_level': 2,
        }
        session_option_default.update(session_option)
        for k, v in session_option_default.items():
            setattr(sess_opt, k, v)
        return sess_opt

    def _get_provider_info(
        self,
        backend: Union[str, int, Backend],
        provider_option: Dict[str, Any] = {},
    ) -> Backend:
        """
        Ref: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options
        """
        if backend == Backend.cuda:
            providers = ['CUDAExecutionProvider']
            provider_option = [{
                'device_id': self.device_id,
                'cudnn_conv_use_max_workspace': '1',
                **provider_option,
            }]
        elif backend == Backend.cpu:
            providers = ['CPUExecutionProvider']
            # "CPUExecutionProvider" is different from everything else.
            provider_option = None
        else:
            raise ValueError(f'backend={backend} is not supported.')
        return providers, provider_option

    def __repr__(self) -> str:
        def format_nested_dict(dict_data, indent=0):
            info = ""
            for k, v in dict_data.items():
                prefix = "  " * indent
                if isinstance(v, dict):
                    info += f"{prefix}{k}:\n" + \
                        format_nested_dict(v, indent + 1)
                elif isinstance(v, str) and v.startswith('{') and v.endswith('}'):
                    try:
                        nested_dict = eval(v)
                        if isinstance(nested_dict, dict):
                            info += f"{prefix}{k}:\n" + \
                                format_nested_dict(nested_dict, indent + 1)
                        else:
                            info += f"{prefix}{k}: {v}\n"
                    except:
                        info += f"{prefix}{k}: {v}\n"
                else:
                    info += f"{prefix}{k}: {v}\n"
            return info

        title = 'DOCSAID X ONNXRUNTIME'
        styled_title = colored.stylize(
            title, [colored.fg('blue'), colored.attr('bold')])
        divider_length = 50
        title_length = len(title)
        left_padding = (divider_length - title_length) // 2
        right_padding = divider_length - title_length - left_padding

        path = f'Model Path: {self.model_path}'
        input_info = format_nested_dict(self.input_infos)
        output_info = format_nested_dict(self.output_infos)
        metadata = format_nested_dict(self.metadata)
        providers = f'Provider: {", ".join(self.providers)}'
        provider_options = format_nested_dict(self.provider_options)

        divider = colored.stylize(
            f"+{'-' * divider_length}+", [colored.fg('blue'), colored.attr('bold')])
        infos = f'\n\n{divider}\n|{" " * left_padding}{styled_title}{" " * right_padding}|\n{divider}\n\n{path}\n\n{input_info}\n{output_info}\n\n{metadata}\n\n{providers}\n\n{provider_options}\n{divider}'
        return infos
