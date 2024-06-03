# from enum import Enum
# from typing import List, Union

# import onnx
# from onnxruntime.quantization import CalibrationDataReader, quantize_static
# from onnxruntime.quantization.calibrate import CalibrationMethod
# from onnxruntime.quantization.quant_utils import QuantFormat

# from ..enums import Enum, EnumCheckMixin
# from ..utils import Path


# class DstDevice(EnumCheckMixin, Enum):
#     mobile = 0
#     x86 = 1


# def _get_exclude_names_from_op_type(onnx_fpath, op_type):
#     onnx_model = onnx.load(str(onnx_fpath))
#     return [node.name for node in onnx_model.graph.node if node.op_type == op_type]


# def quantize(
#     onnx_fpath: Union[str, Path],
#     calibration_data_reader: CalibrationDataReader,
#     dst_device: Union[str, DstDevice] = DstDevice.mobile,
#     op_type_to_exclude: List[str] = [],
#     **quant_cfg,
# ) -> str:
#     dst_device = DstDevice.obj_to_enum(dst_device)
#     onnx_fpath = Path(onnx_fpath)

#     print(f'\nStart to quantize onnx model to {dst_device.name}')

#     quant_fpath = str(onnx_fpath).replace('fp32', '').replace(
#         '.onnx', f'_int8_{dst_device.name}.onnx')

#     quant_format = QuantFormat.QOperator \
#         if dst_device == DstDevice.mobile else QuantFormat.QDQ
#     quant_format = quant_cfg.get('quant_format', quant_format)
#     calibrate_method = quant_cfg.get(
#         'calibrate_method', CalibrationMethod.MinMax)
#     per_channel = quant_cfg.get('per_channel', True)
#     reduce_range = quant_cfg.get('reduce_range', True)
#     nodes_to_exclude = quant_cfg.get('nodes_to_exclude', [])

#     for op_type in op_type_to_exclude:
#         nodes_to_exclude.extend(
#             _get_exclude_names_from_op_type(onnx_fpath, op_type))

#     quantize_static(
#         onnx_fpath,
#         quant_fpath,
#         calibration_data_reader,
#         quant_format=quant_format,
#         calibrate_method=calibrate_method,
#         per_channel=per_channel,
#         reduce_range=reduce_range,
#         nodes_to_exclude=nodes_to_exclude
#     )

#     return quant_fpath
