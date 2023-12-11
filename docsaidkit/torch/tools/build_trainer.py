from pathlib import Path
from typing import Any, Dict, Tuple, Union

import lightning.pytorch.callbacks as pl_callbacks
import lightning.pytorch.loggers as pl_loggers
import natsort
import torch.nn as nn
from lightning import Trainer
from torch.utils.data import DataLoader

from ...utils import (PowerDict, colorstr, dump_json, get_curdir, get_files,
                      get_gpu_cuda_versions, get_package_versions,
                      get_system_info, now)
from .custom_callbacks import CustomTQDMProgressBar

__all__ = [
    'load_model_from_config',
    'build_callback',
    'build_logger',
    'build_dataset',
    'build_trainer'
]

pl_callbacks.CustomTQDMProgressBar = CustomTQDMProgressBar


def load_model_from_config(
    cfg_name: Union[str, Path],
    root: Union[str, Path] = None,
    stem: Union[str, Path] = None,
    network: Dict[str, Any] = {},
) -> Tuple[nn.Module, PowerDict]:

    T = now(fmt='%Y-%m-%d-%H-%M-%S')

    if root is None:
        DIR = get_curdir(__file__)
    else:
        DIR = Path(root)

    cfg = PowerDict.load_yaml(DIR / Path(stem) / f'{cfg_name}.yaml')
    ind = cfg.common.restore_ind if cfg.common.restore_ind != "" else T
    cfg.update({'name': str(cfg_name), 'name_ind': ind})

    # check key
    if 'common' not in cfg:
        raise KeyError('Key "common" is not in config file.')

    # check model
    if 'model' not in cfg:
        raise KeyError('Key "model" is not in config file.')

    # check model name
    if 'name' not in cfg.model:
        raise KeyError('Key "name" is not in config file.')

    # load model
    net_name = cfg.model.name
    if cfg.common.is_restore:
        _ckpt = cfg.common.restore_ckpt
        _path = Path().joinpath(str(DIR), cfg_name, ind, 'checkpoint', 'model')
        if _ckpt is None or _ckpt == '':
            _candi_model = [i for i in get_files(
                _path, suffix=['.ckpt']) if 'last' in i.stem]
            _ckpt = natsort.os_sorted(_candi_model)[-1]
        checkpoint_path = str(DIR / _path / _ckpt)
        model = getattr(network, net_name).load_from_checkpoint(
            checkpoint_path, cfg=cfg, strict=False)
        print(
            f'MODEL Load from checkpoint {colorstr(checkpoint_path)}... Done.',
            flush=True
        )
    else:
        model = getattr(network, net_name)(cfg=cfg)

    return model, cfg


def build_callback(cfg: PowerDict):
    callbacks = []
    for callback in cfg.callbacks:
        if callback.name == 'ModelCheckpoint':
            dirpath = Path().joinpath(cfg.root_dir, 'checkpoint', 'model')
            callback.options.update({'dirpath': str(dirpath)})
        elif callback.name == 'CustomTQDMProgressBar':
            callback.options.update({'unit_scale': cfg.common.batch_size})
        options = getattr(callback, 'options', {})
        callbacks.append(getattr(pl_callbacks, callback.name)(**options))
    return callbacks


def build_logger(cfg: PowerDict):
    logger = getattr(pl_loggers, cfg.logger.name)(**cfg.logger.options)
    dump_json(cfg, Path(cfg.logger.options.save_dir) / 'config.json')
    return logger


def build_dataset(cfg: PowerDict, ds: Dict[str, Any] = {}):
    ds_loader_train_opts = cfg.dataloader.train_options
    ds_loader_valid_opts = cfg.dataloader.valid_options
    ds_loader_train_opts.update({'batch_size': cfg.common.batch_size})
    ds_loader_valid_opts.update({'batch_size': cfg.common.batch_size})
    ds_train_name, ds_train_opts = cfg.dataset.train_options.values()
    ds_valid_name, ds_valid_opts = cfg.dataset.valid_options.values()
    if 'global_settings' in cfg:
        ds_train_opts.update(cfg.global_settings)
        ds_valid_opts.update(cfg.global_settings)
    ds_train = getattr(ds, ds_train_name)(**ds_train_opts)
    ds_valid = getattr(ds, ds_valid_name)(**ds_valid_opts)
    train_data = DataLoader(dataset=ds_train, **ds_loader_train_opts)
    valid_data = DataLoader(dataset=ds_valid, **ds_loader_valid_opts)
    return train_data, valid_data


def build_trainer(
    cfg: Dict[str, Any],
    root: Union[str, Path] = None,
):

    if root is None:
        DIR = get_curdir(__file__)
    else:
        DIR = Path(root)

    # root dir
    root_dir = DIR / cfg.name / cfg.name_ind

    # Add the directory to .gitignore if it exists
    if (DIR.parent / '.gitignore').is_file():
        with open(str(DIR.parent / '.gitignore'), 'r') as file:
            content = file.read()

        # 檢查是否已包含所需的文本
        entry = f'\n{cfg.name}/\n'
        if entry not in content:
            # 如果没有包含，则將其添加到 .gitignore 文件
            with open(str(DIR.parent / '.gitignore'), 'a') as f:
                f.write(f'\n{cfg.name}/\n')

    cfg.update({'root_dir': str(root_dir)})
    cfg.logger.options.update({
        'save_dir': str(root_dir / cfg.logger.options.save_dir)
    })
    if not (log_dir := Path(cfg.logger.options.save_dir)).is_dir():
        log_dir.mkdir(parents=True)

    callbacks = build_callback(cfg)
    logger = build_logger(cfg)

    # Log infos
    pkg_ver = get_package_versions()
    dump_json(pkg_ver, Path(cfg.logger.options.save_dir) /
              'PackageVersions.json')

    gpu_cuda_ver = get_gpu_cuda_versions()
    dump_json(gpu_cuda_ver, Path(cfg.logger.options.save_dir) /
              'GPU-Cuda-Versions.json')

    sys_info = get_system_info()
    dump_json(sys_info, Path(cfg.logger.options.save_dir) /
              'SystemInfos.json')

    return Trainer(**cfg.trainer, logger=logger, callbacks=callbacks)
