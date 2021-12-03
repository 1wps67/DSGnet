
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

#此文件是所有可配置选项的一站式参考点。 它应该有很好的文档记录，并为所有选项提供合理的默认值。
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


#创建一个配置节点_C
_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''

_C.GPUS = (0,)
_C.WORKERS = 4
#20组batchsize一输出
_C.PRINT_FREQ = 20

_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0


#在_C下创建新的配置节点_C.CUDNN
#给_C.CUDNN的属性配置默认值

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True


# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'cls_hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 2
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.LOSS = CN()
_C.LOSS.USE_OHKM = False
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''

#数据集总名字 ，下面有 images //train / valid
_C.DATASET.DATASET = 'mpii'

_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'


_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.HYBRID_JOINTS_TYPE = ''
_C.DATASET.SELECT_DATA = False

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.PROB_HALF_BODY = 0.0
_C.DATASET.NUM_JOINTS_HALF_BODY = 8
_C.DATASET.COLOR_RGB = False

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1

#epoch轮次更换学习率
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001

#优化器
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0


_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140



_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

#更改了batchsize
#_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.BATCH_SIZE_PER_GPU = 8

_C.TRAIN.SHUFFLE = True


# testing
_C.TEST = CN()
# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.POST_PROCESS = False
_C.TEST.SHIFT_HEATMAP = False
_C.TEST.USE_GT_BBOX = False

# nms
_C.TEST.IMAGE_THRE = 0.1
_C.TEST.NMS_THRE = 0.6
_C.TEST.SOFT_NMS = False
_C.TEST.OKS_THRE = 0.5
_C.TEST.IN_VIS_THRE = 0.0
_C.TEST.COCO_BBOX_FILE = ''
_C.TEST.BBOX_THRE = 1.0

_C.TEST.MODEL_FILE = ''

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False

#update_config(config, args)
#--cfg experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yam

def update_config(cfg, args):

    cfg.defrost()

    #通过merge_from_file这个函数会将yaml文件中指定的超参数对默认值进行覆盖。
    cfg.merge_from_file(args.cfg)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir

    if args.testModel:
        cfg.TEST.MODEL_FILE = args.testModel

    cfg.DATASET.ROOT = os.path.join(
        cfg.DATA_DIR, cfg.DATASET.DATASET, 'images')

    cfg.freeze()


if __name__ == '__main__':
    import sys
    #运行代码时的参数
    with open(sys.argv[1], 'w') as f:
        #输入数据到f
        print(_C, file=f)

