# -*- coding: utf-8 -*-
# @Time    : 2020/6/5 11:35
# @Author  : zhoujun

from .DB_CT_Head import DB_CT_Head


__all__ = ['build_head']
support_head = ['DB_CT_Head']


def build_head(head_name, **kwargs):
    assert head_name in support_head, f'all support head is {support_head}'
    head = eval(head_name)(**kwargs)
    return head