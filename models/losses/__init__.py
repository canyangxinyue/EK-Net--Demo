# -*- coding: utf-8 -*-
# @Time    : 2020/6/5 11:36
# @Author  : zhoujun
import copy

from .EK_loss import EK_loss

__all__ = ['build_loss']
support_loss = [ "EK_loss"]

def build_loss(config):
    copy_config = copy.deepcopy(config)
    loss_type = copy_config.pop('type')
    assert loss_type in support_loss, f'all support loss is {support_loss}'
    criterion = eval(loss_type)(**copy_config)
    return criterion
