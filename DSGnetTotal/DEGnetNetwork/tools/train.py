# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
"/“操作符执行的是截断除法(Truncating Division),当我们导入精确除法之后，”/"执行的是精确除法，如下所示：
>>> 3/4
0
>>> from __future__ import division
>>> 3/4
0.75

 First, the four-resolution feature maps are fed into a bottleneck 
 the number of output channels are increased to 128, 256, 512, and 1024,
 Then, we downsample the high-resolution representations by a 2-strided 3x3 convolution outputting 256 channels 
 and add them to the representations of the second-high-resolution representations
 This process is repeated two times to get 1024 channels over the small resolution.
 we transform 1024 channels to 2048 channels through a 1x1 convolution, followed by a global average pooling operation
 The output 2048-dimensional representation is fed into the classifier

'''
import argparse
import os
import pprint

'''
    print()和pprint()都是python的打印模块，功能基本一样，唯一的区别就是pprint()模块打印出来的数据结构更加完整，每行为一个数据结构，更加方便阅读打印输出结果。
    特别是对于特别长的数据打印，print()输出结果都在一行，不方便查看，而pprint()采用分行打印输出，所以对于数据结构比较复杂、数据长度较长的数据，适合采用pprint()打印方式。
    当然，一般情况多数采用print()。
'''
import shutil
import sys
import torch
import cv2
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from tensorboardX import SummaryWriter
import _init_paths
import models

from config import config
from config import update_config

from core.function import train
from core.function import validate
from utils.modelsummary import get_model_summary
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from torchviz import make_dot


class trainData(Dataset):
    def __init__(self, img_path,txt_path,depthMap_path,what_do, data_transforms=None):
        with open(os.path.join(txt_path, 'label.txt')) as input_file:
            # 以list形式保存txt文件的每一行
            lines = input_file.readlines()
            self.img_path = img_path
            self.img_label = txt_path
            self.img_depthMap_path = depthMap_path

            if what_do == 'train':
                self.img_name = [os.path.join(img_path, line.strip().split('\t')[0][0:-2][0:-4] + '_masked.jpg') for
                                 line in lines if line != '\n']
                self.img_label = [int(line.strip().split('\t')[0][-1:]) for line in lines if line != '\n']
            elif what_do == 'val':
                self.img_name = [os.path.join(img_path, line.strip().split('\t')[0][0:-2][0:-4] + '_masked.jpg') for
                                 line in lines if line != '\n']
                self.img_label = [int(line.strip().split('\t')[0][-1:]) for line in lines if line != '\n']
            #print(self.img_name,self.img_label)
            print(len(self.img_name))
        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        img_label = self.img_label[item]
        img = Image.open(img_name).convert('RGB')
        img = np.array(img)

        img_depthMap_path = os.path.join(self.img_depthMap_path,img_name.split('/')[-1][0:-11] + '.jpg')
        vis_depthMap_pic = cv2.imread(img_depthMap_path,cv2.IMREAD_GRAYSCALE)
        vis_depthMap_pic = np.array(vis_depthMap_pic)
        vis_depthMap_pic = np.expand_dims(vis_depthMap_pic,axis = -1)

        '''
        #print('01',vis_keypoint_pic.shape)
        img = np.concatenate((img, vis_keypoint_pic), axis = 2)
        #print(img.shape)
        '''

        if self.data_transforms is not None:
            for transforms in self.data_transforms:
                img = transforms(img)
        if self.data_transforms is not None:
            for transforms in self.data_transforms:
                vis_depthMap_pic = transforms(vis_depthMap_pic)

        return img, vis_depthMap_pic ,img_label


# python tools/train.py --cfg experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')

    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')

    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')

    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')

    args = parser.parse_args()

    update_config(config, args)

    return args


def main():
    args = parse_args()
    # logger , root_output_dir / dataset / cfg_name ,tensorboard_log_dir : Path(cfg.LOG_DIR) / dataset / model /
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')

    # 假定你有一个字典，保存在一个变量中，你希望保存这个变量和它的内容，以便将来使用。
    # pprint.pformat()函数将提供一个字符串，你可以将它写入.py 文件。该文件将成为你自己的模块，如果你需要使用存储在其中的变量，就可以导入它。

    # logger.info(pprint.pformat(args))
    # logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK  # True
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC  # False
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED  # True

    # 'models.cls_hrnet.get_cls_net'
    model = eval('models.' + config.MODEL.NAME + '.get_cls_net')(config)

    # B,C,H,W
    dump_input = torch.rand(
        (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
    # logger.info(get_model_summary(model, dump_input))

    # copy model file
    # this_dir = os.path.dirname(__file__)
    # models_dst_dir = os.path.join(final_output_dir, 'models')
    # if os.path.exists(models_dst_dir):
    #    shutil.rmtree(models_dst_dir)
    # shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    '''
    checkpoints_path = r'/home/wh/lwp/hrnet-class/HRNet-Image-Classification-master/hrnetv2_w18_imagenet_pretrained.pth'
    if os.path.isfile(checkpoints_path):
        pretrained_dict = torch.load(checkpoints_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
        # 可以修改存在的键对应的值，也可以添加新的键 / 值对到字典中。
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    '''

    # model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    model = model.cuda()
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = get_optimizer(config, model)

    last_epoch = config.TRAIN.BEGIN_EPOCH

    '''
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')

        #os.path.isdir()用于判断某一对象(需提供绝对路径) 是否为目录
        #os.path.isfile(）用于判断某一对象(需提供绝对路径) 是否为文件
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
            best_model = True

    '''

    # torch.optim
    # 每次遇到 config.TRAIN.LR_STEP 中的epoch 做一次更新
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            # _C.TRAIN.LR_FACTOR = 0.1
            # _C.TRAIN.LR_STEP = [90, 110]
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            last_epoch - 1
        )
    else:
        # 每过step_size个epoch，做一次更新：
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            # optimizer step_size  gamma  last_epoch
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            # 最后一个epoch的index
            last_epoch - 1
        )

    # Data loading code
    # traindir = os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET)
    # valdir = os.path.join(config.DATASET.ROOT, config.DATASET.TEST_SET)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    # import torchvision.datasets as datasets
    '''
    root：图片存储的根目录，即各类别文件夹所在目录的上一级目录。
    transform：对图片进行预处理的操作（函数），原始图片作为输入，返回一个转换后的图片。
    target_transform：对图片类别进行预处理的操作，输入为 target，输出对其的转换。如果不传该参数，即对 target 不做任何转换，返回的顺序索引 0,1, 2…
    loader：表示数据集加载方式，通常默认加载方式即可。
    is_valid_file：获取图像文件的路径并检查该文件是否为有效文件的函数(用于检查损坏文件)
    '''

    '''
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.RandomHorizontalFlip(),
            #Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.
            # FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            transforms.ToTensor(),
            normalize,
        ])
    )
    '''

    train_pic_path = r'/mnt/lwp/mmdetection_final_clean/maskedpic'
    val_pic_path = r'/mnt/lwp/mmdetection_final_clean_test/maskedpic'

    train_label_path = r'/mnt/lwp/mmdetection_final_clean/label'
    val_label_path = r'/mnt/lwp/mmdetection_final_clean_test/label'

    train_depthMap_path = r'/mnt/lwp/mmdetection_final_clean/final_depth_map'
    test_depthMap_path = r'/mnt/lwp/mmdetection_final_clean_test/final_depth_map'

    writer = SummaryWriter(r'/home/wh/lwp/hrnet-class/HRNet-Image-Classification-master/tools/')

    # 对每张图片的操作
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms_list = [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ]

    # 构造训练集
    train_dataset = trainData(train_pic_path, train_label_path,train_depthMap_path, 'train', transforms_list)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=True,
        # num_workers (int, optional) – 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # 构造测试集
    val_dataset = trainData(val_pic_path, val_label_path,test_depthMap_path,'val', transforms_list)
    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
        # drop_last (bool, optional) – 如果数据集大小不能被batch size整除，
        # 则设置为True后可删除最后一个不完整的batch。
        # 如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)
    )

    # last_epoch             开始训练的轮次
    # config.TRAIN.END_EPOCH 结束的轮次
    for param in model.parameters():
        param.requires_grad = False
    for param in model.Catclassifier4.parameters():
        param.requires_grad = True
    for param in model.Catclassifier5.parameters():
        param.requires_grad = True
    for param in model.Catclassifier6.parameters():
        param.requires_grad = True
    for param in model.Catclassifier7.parameters():
        param.requires_grad = True

    best_perf = 1000
    best_acc = 0
    best_model = False
    save_checkpoint_path = r'/home/wh/lwp/hrnet-class/HRNet-Image-Classification-master/tools'

    # temp = model(dump_input)
    # g = make_dot(temp)
    # g.render('espnet_model', view=False)

    print('------ start train ------')
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        # train for one epoch
        lr_scheduler.step()
        
        train(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict, writer)

        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, model, criterion,final_output_dir, tb_log_dir,writer,epoch, writer_dict)
        if perf_indicator < best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False
        '''
        if perf_indicator > best_acc :
            best_acc = perf_indicator
            best_model = True
        else :
            best_model = False
        '''
        save_checkpoint({
            'epoch': epoch + 1,
            'model': config.MODEL.NAME,
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, save_checkpoint_path, filename='checkpoint.pth.tar')


        #validate(config, valid_loader, model, criterion, final_output_dir, tb_log_dir, writer, epoch, writer_dict)

    writer.close()

    # logger.info('=> saving checkpoint to {}'.format(final_output_dir))

    # final_model_state_file = os.path.join(final_output_dir,'final_state.pth.tar')
    # logger.info('saving final model state to {}'.format(final_model_state_file))

    # 在pytorch中，torch.nn.Module模块中的state_dict变量存放训练过程中需要学习的权重和偏置系数
    # torch.save(model.module.state_dict(), final_model_state_file)
    # writer_dict['writer'].close()


if __name__ == '__main__':
    main()
