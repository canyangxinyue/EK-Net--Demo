# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:52
# @Author  : zhoujun
import copy

import PIL
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def get_dataset(data_path, module_name, transform, dataset_args):
    """
    获取训练dataset
    :param data_path: dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :param module_name: 所使用的自定义dataset名称，目前只支持data_loaders.ImageDataset
    :param transform: 该数据集使用的transforms
    :param dataset_args: module_name的参数
    :return: 如果data_path列表不为空，返回对于的ConcatDataset对象，否则None
    """
    from . import dataset
    s_dataset = getattr(dataset, module_name)(transform=transform, data_path=data_path,
                                              **dataset_args)
    return s_dataset


def get_transforms(transforms_config):
    tr_list = []
    for item in transforms_config:
        if 'args' not in item:
            args = {}
        else:
            args = item['args']
        cls = getattr(transforms, item['type'])(**args)
        tr_list.append(cls)
    tr_list = transforms.Compose(tr_list)
    return tr_list


class ICDARCollectFN:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        data_dict = {}
        to_tensor_keys = []
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, (np.ndarray, torch.Tensor, PIL.Image.Image)):
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                data_dict[k].append(v)
        for k in to_tensor_keys:
            data_dict[k] = torch.stack(data_dict[k], 0)
        return data_dict

class WildReceiptCollectFN:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        data_dict = {}
        to_tensor_keys = []
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, (np.ndarray, torch.Tensor, PIL.Image.Image)):
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                    if not isinstance(v, torch.Tensor):
                        v=torch.tensor(v)
                data_dict[k].append(v)
        for k in to_tensor_keys:
            data_dict[k] = torch.stack(data_dict[k], 0)
        return data_dict


class ICDAR2019RECCollectFN:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        data_dict = {}
        to_tensor_keys = []
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, (np.ndarray, torch.Tensor, PIL.Image.Image)):
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                    if not isinstance(v, torch.Tensor):
                        v=torch.tensor(v)
                data_dict[k].append(v)
        for k in to_tensor_keys:
            data_list=data_dict[k]
            data_len=max([t.shape[-1] for t in data_list])
            data_list=[torch.nn.functional.pad(data,(0,data_len-data.shape[-1])) for data in data_list]
            data_dict[k] = torch.stack(data_list, 0)
        return data_dict

def get_dataloader(module_config, distributed=False):
    if module_config is None:
        return None
    config = copy.deepcopy(module_config)
    dataset_args = config['dataset']['args']
    if 'transforms' in dataset_args:
        img_transfroms = get_transforms(dataset_args.pop('transforms'))
    else:
        img_transfroms = None
    # 创建数据集
    dataset_name = config['dataset']['type']
    data_path = dataset_args.pop('data_path')
    if data_path == None:
        return None

    data_path = [x for x in data_path if x is not None]
    if len(data_path) == 0:
        return None
    if 'collate_fn' not in config['loader'] or config['loader']['collate_fn'] is None or len(config['loader']['collate_fn']) == 0:
        config['loader']['collate_fn'] = None
    else:
        config['loader']['collate_fn'] = eval(config['loader']['collate_fn'])()

    _dataset = get_dataset(data_path=data_path, module_name=dataset_name, transform=img_transfroms, dataset_args=dataset_args)
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        # 3）使用DistributedSampler
        sampler = DistributedSampler(_dataset)
        config['loader']['shuffle'] = False
        config['loader']['pin_memory'] = True
    loader = DataLoader(dataset=_dataset, sampler=sampler, **config['loader'])
    return loader
