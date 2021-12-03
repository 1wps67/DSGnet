# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import torchvision
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

        '''
        for layers in self.parameters():
            layers.requires_grad=False
        '''

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#  64   64   1   downsample 64 -> 256
# block(inplanes, planes, stride, downsample)
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        #默认 padding = 0 , stride = 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        '''
        for layers in self.parameters():
            layers.requires_grad=False
        '''

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        '''
        调用：
        # 调用高低分辨率交互模块， stage2 为例
        HighResolutionModule(num_branches,     # 2
                             block,            # 'BASIC'
                             num_blocks,       # [4, 4]
                             num_inchannels,   # 上个transition 的out channel[18,36]
                             num_channels,     # [18, 36]
                             fuse_method,      # SUM
                             reset_multi_scale_output) # True
        '''

        #检查分支数是否合理
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels

        #融合选用相加方式
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        #核心部分 branches构建  layers构建
        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    ## 分别检查参数是否符合要求,看models.py中的参数，blocks参数冗余了
    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):

        # 构建一个分支，一个分支重复num_blocks个block
        downsample = None
        # 这里判断，如果通道变大(分辨率变小)，则使用下采样
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        # 通过循环构建多分支，每个分支属于不同的分辨率
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches      # 2
        num_inchannels = self.num_inchannels  # [18,36]
        fuse_layers = []
        # i代表枚举所有分支
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i], 
                                       momentum=BN_MOMENTUM),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    # 本层不做处理
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    # 进行strided 3x3 conv下采样,如果跨两层，就使用两次strided 3x3 conv
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        # 将fuse以后的多个分支结果保存到list中
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse



# 前面的两段

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(HighResolutionNet, self).__init__()
        # 3 -> 4

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        #self.convx = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1,
        #                       bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        #待处理的数据的通道数, 而这个1 - momentum就指定了保存前一batch mean 和 var 的比例.
        # BN_MOMENTUM = 0.1
        # 新batch的比例是 = 0.9
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # 直接修改 inplace
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        self.stage1_cfg = cfg['MODEL']['EXTRA']['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0] #  64  通道数
        block = blocks_dict[self.stage1_cfg['BLOCK']] #      BOTTLENECK Bottleneck 有1 * 1 和 3 * 3 bn relu
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0] #      4   块数
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels


        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']  #[18 , 36]
        block = blocks_dict[self.stage2_cfg['BLOCK']]  # BasicBlock
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]  # [18 , 36]
        # transition 过渡
        self.transition1 = self._make_transition_layer(# 64 * 4 , [18,36]
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)


        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']  #[18,36,72]
        block = blocks_dict[self.stage3_cfg['BLOCK']]   #basic
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))] #[18,36,72]
        self.transition2 = self._make_transition_layer(    #[18,36] -> #[18,36,72]
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS'] #[18,36,72,144]
        block = blocks_dict[self.stage4_cfg['BLOCK']]  #[basic]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))] #[18,36,72,144]
        self.transition3 = self._make_transition_layer( #[18,36,72] -> [18,36,72,144]
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        # Classification Head
        self.incre_modules, self.downsamp_modules, \
            self.final_layer = self._make_head(pre_stage_channels)

        #hrnet 后面几层的全连接
        self.classifier1 = nn.Linear(2048,1024)
        self.classifier2 = nn.Linear(1024,256)
        self.classifier3 = nn.Linear(256, 128)

        #支路resnet
        self.resnet = torchvision.models.resnet50(pretrained=True)
        #self.resnet = torchvision.models.resnet50(pretrained=True) 下面的#加载模型会错
        channel_in = self.resnet.fc.in_features
        class_num = 128
        self.resnet.fc = nn.Sequential(
            nn.Linear(channel_in, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, class_num),
            nn.ReLU(),
        )
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        #self.resnet.load_state_dict(torch.load(r'/home/wh/lwp/hrnet-class/HRNet-Image-Classification-master/tools/CLean_second_first_model_best.pth.tar'))
        self.resnet = self.resnet.cuda()

        #concate 后的支路
        self.Catclassifier4 = nn.Linear(256,128)
        self.Catclassifier5 = nn.Linear(128,64)
        self.Catclassifier6 = nn.Linear(64,32)
        self.Catclassifier7 = nn.Linear(32,2)

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution 
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels  in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,       #Bottleneck
                                            channels,         #[18,36,72,144]
                                            head_channels[i], #[32,64,128,256]
                                            1,
                                            stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
            
        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels)-1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i+1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    #   64 * 4    , [18,36]   /  [18,36] -> [18,36,72]
    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)  # 2 / 3
        num_branches_pre = len(num_channels_pre_layer)  # 1 / 2

        transition_layers = []
        for i in range(num_branches_cur): #(0,1)
            if i < num_branches_pre: #(1)
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(  #kernal_size , stride , padding
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        #如果完全直接用
        #nn.Sequential，确实是可以的，但这么做的代价就是失去了部分灵活性，不能自己去定制 forward 函数里面的内容了。
        return nn.ModuleList(transition_layers)

    # self.layer1 = self._make_layer(block, 64, num_channels(64), num_blocks(4)
    # 名字 输入channel 输出channel  blocks数量  步长
    '''
                    incre_module = self._make_layer(head_block,       #Bottleneck
                                            channels,                 #[18,36,72,144]
                                            head_channels[i],         #[32,64,128,256]
                                            1,                        # 1
                                            stride=1)                 # 1
    '''
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        # 实例化网络
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)
    # config [18,36]
    def _make_stage(self, layer_config, num_inchannels,      #[18,36,72]
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']   #1       / 4             /3
        num_branches = layer_config['NUM_BRANCHES'] #2       / 3             /4
        num_blocks = layer_config['NUM_BLOCKS']     #[4,4]   / [4,4,4]       /[4,4,4,4]
        num_channels = layer_config['NUM_CHANNELS'] #[18,36] / [18,36,72]    /[18,36,72,144]
        block = blocks_dict[layer_config['BLOCK']]  #Basic   / Basic         /Basic
        fuse_method = layer_config['FUSE_METHOD']   #SUM     / SUM           /SUM

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x,z):
        #first hrnet
        x = x.type(torch.cuda.FloatTensor)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # Classification Head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i+1](y_list[i+1]) + \
                        self.downsamp_modules[i](y)

        y = self.final_layer(y)

        # hrnet full connect
        y = y.flatten(start_dim=2).mean(dim=2)
        y = self.relu(self.classifier1(y))
        y = self.dropout(y)
        y = self.relu(self.classifier2(y))
        y = self.dropout(y)
        y = self.relu(self.classifier3(y))

        # hrnet full connect
        z = z.type(torch.cuda.FloatTensor)  # z -> depth map
        z = self.resnet(z)
        d= torch.cat((y,z) ,dim = 1)    # concate
        y = self.relu(self.Catclassifier4(d))
        y = self.relu(self.Catclassifier5(y))
        y = self.relu(self.Catclassifier6(y))
        y = self.Catclassifier7(y)

        return y


    def init_weights(self, pretrained='',nextpretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        #first 支路
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            #可以修改存在的键对应的值，也可以添加新的键 / 值对到字典中。
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print('load model paiameter success : {}'.format(pretrained.split('/')[-1]))

        #second支路
        if os.path.isfile(nextpretrained):
            nextpretrained_dict = torch.load(nextpretrained)
            model_dict = self.state_dict()
            nextpretrained_dict = {k: v for k, v in nextpretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(nextpretrained_dict)
            self.load_state_dict(model_dict)
            print('load model parameter success : {}'.format(nextpretrained.split('/')[-1]))

def get_cls_net(config,**kwargs):
    model = HighResolutionNet(config, **kwargs)
    #可以通过这句加载预训练好的模型 传入checkpoints的绝对路径
    #model.init_weights(r'/home/wh/lwp/hrnet-class/HRNet-Image-Classification-master/tools/model_best.pth.tar')
    #model.init_weights(r'/home/wh/lwp/hrnet-class/HRNet-Image-Classification-master/tools/model_best_addkeypoints.pth.tar')
    #model.init_weights(r'/home/wh/lwp/hrnet-class/HRNet-Image-Classification-master/tools/new_first_model_best.pth.tar')
    #model.init_weights(r'/home/wh/lwp/hrnet-class/HRNet-Image-Classification-master/tools/depth_model_best.pth.tar')
    #model.init_weights(r'/home/wh/lwp/hrnet-class/HRNet-Image-Classification-master/tools/CLean_new_first_model_best.pth.tar',r'/home/wh/lwp/hrnet-class/HRNet-Image-Classification-master/tools/CLean_second_first_model_best.pth.tar')
    model.init_weights(r'/home/lwp/deepc/mainCode/HRNet-Image-Classification-master/tools/Together_model_best.pth.tar')
    print('---------------------- over init and load ----------------------')
    return model
