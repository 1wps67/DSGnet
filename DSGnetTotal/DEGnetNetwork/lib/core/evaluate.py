# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        # K值 dim = 1
        # 得 值 和 索引
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        #tensor.expand_as()这个函数就是把一个tensor变成和函数括号内一样形状的tensor，用法与expand（）类似。
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #torch.sum(input, dim, out=None) dim=0，对列求和；dim=1，对行求和。
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
