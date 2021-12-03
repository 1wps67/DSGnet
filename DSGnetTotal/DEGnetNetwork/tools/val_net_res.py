from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import json
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pprint
import shutil
import sys
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset,DataLoader
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import runDepth
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
import os
import glob
import torch
import utilsDepth
import cv2
import argparse
import sys
from torchvision.transforms import Compose
sys.path.append('../..')
from MiDaS.midas.midas_net import MidasNet
from MiDaS.midas.midas_net_custom import MidasNet_small
from MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet

val_picture_path = r'./picture'
val_json_path = r'./json'
val_masked_path = r'./masked_pic'
val_res_path = r'./our_res'
normalize = transforms.Normalize((0.5,), (0.5,))
transforms_list = [
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
]
Image.MAX_IMAGE_PIXELS = 1000000000

def select(img,dic_it,json_data) :
    res_lst =[]
    rect = dic_it['rect']
    x1 = int(rect['tl']['x'] * img.shape[1])
    x2 = int(rect['br']['x'] * img.shape[1])
    y1 = int(rect['tl']['y'] * img.shape[0])
    y2 = int(rect['br']['y'] * img.shape[0])
    r = x2 - x1
    #取bbox中心�?
    dic_lst = get_midpoint(x1,y1,x2,y2)
    dic_lst = np.array(dic_lst)

    for data in json_data :
        if data == dic_it :
            continue
        else :
            data_rect = data['rect']
            dx1 = int(data_rect['tl']['x'] * img.shape[1])
            dx2 = int(data_rect['br']['x'] * img.shape[1])
            dy1 = int(data_rect['tl']['y'] * img.shape[0])
            dy2 = int(data_rect['br']['y'] * img.shape[0])
            data_lst = get_midpoint(dx1,dy1,dx2,dy2)
            #list to numpy
            data_lst = np.array(data_lst)
            #l2 dis
            d1 = np.sqrt(np.sum(np.square(data_lst - dic_lst)))
            if d1 < r :
                res_lst.append(data)
    return res_lst

def get_midpoint(x1,y1,x2,y2) :
    return (y2 - y1) / 2 + y1 , (x2 - x1) / 2 + x1

def get_pic_mask(img,small_img,data,dic_it,up,down,left,right) :

    #1024 * 2048 , 0 is black 255 write
    mask = np.zeros(img.shape[0:2], dtype='uint8')
    firstx1 = int(dic_it['rect']['tl']['x'] * img.shape[1])
    firsty1 = int(dic_it['rect']['tl']['y'] * img.shape[0])
    firstx2 = int(dic_it['rect']['br']['x'] * img.shape[1])
    firsty2 = int(dic_it['rect']['br']['y'] * img.shape[0])
    cv2.rectangle(mask, (firstx1, firsty1), (firstx2, firsty2), 255, -1)
    secondx1 = int(data['rect']['tl']['x'] * img.shape[1])
    secondy1 = int(data['rect']['tl']['y'] * img.shape[0])
    secondx2 = int(data['rect']['br']['x'] * img.shape[1])
    secondy2 = int(data['rect']['br']['y'] * img.shape[0])
    cv2.rectangle(mask, (secondx1, secondy1), (secondx2, secondy2), 255, -1)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    masked_img_name = 'masked.jpg'
    store_path = os.path.join(val_masked_path, masked_img_name)
    cv2.imwrite(store_path, masked_img[up:down + 1, left:right + 1])
    return store_path

def create_depth_mask(img,newfirstx1,newfirsty1,newfirstx2,newfirsty2,newsecondx1,newsecondy1,newsecondx2,newsecondy2):
    mask = np.zeros(img.shape[0:2], dtype='uint8')
    # 如果为负值，如CV_FILLED，则表示填充整个矩形。
    cv2.rectangle(mask, (newfirstx1, newfirsty1), (newfirstx2, newfirsty2), 255, -1)
    cv2.rectangle(mask, (newsecondx1, newsecondy1), (newsecondx2, newsecondy2), 255, -1)
    mask_pic = cv2.bitwise_and(img, img, mask=mask)
    return mask_pic

def change(x1,y1,x2,y2, up, down, left, right) :
    xmin = max(left,x1)
    ymin = max(up,y1)
    xmax = min(right,x2)
    ymax = min(down,y2)
    return (xmin - left) / (right - left),(ymin - up) / (down - up),(xmax - left) / (right - left),(ymax - up) / (down - up)

def run(need_depth_img,depthModel,transform,device,optimize = True) :
    # img
    if need_depth_img.ndim == 2:
        need_depth_img = cv2.cvtColor(need_depth_img, cv2.COLOR_GRAY2BGR)
    need_depth_img = cv2.cvtColor(need_depth_img, cv2.COLOR_BGR2RGB) / 255.0

    img_input = transform({"image": need_depth_img})["image"]

    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        if optimize == True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()
        prediction = depthModel.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=need_depth_img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
                .squeeze()
                .cpu()
                .numpy()
        )
    # output
    filename = os.path.join(r'/home/lwp/deepc/mainCode/tempfpm', 'tempPic')
    depth_map = utilsDepth.write_depth(filename, prediction, bits=2)
    return depth_map


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


def main() :
    print('init Depth map')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depthModel = MidasNet(r'/home/lwp/deepc/mainCode/MiDaS/model-f6b98070.pt', non_negative=True)

    depth_transform = Compose(
        [
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    depthModel.eval()
    rand_example = torch.rand(1, 3, 384, 384)
    depthModel(rand_example)
    traced_script_module = torch.jit.trace(depthModel, rand_example)
    model = traced_script_module
    if device == torch.device("cuda"):
        depthModel = depthModel.to(memory_format=torch.channels_last)
        depthModel = depthModel.half()
    depthModel.to(device)
    print('Finish init Depthmodel')

    args = parse_args()
    model = eval('models.' + config.MODEL.NAME + '.get_cls_net')(config)
    model = model.cuda()
    model.eval()


    picture_list = os.listdir(val_picture_path)
    for picture in picture_list :
        #find picture
        picture_name = picture
        picture_path = os.path.join(val_picture_path, picture_name)

        '''
        #panda
        #find json
        able_name_list = picture_name.split('_')
        length = len(able_name_list)
        locate = begin = 0
        for it in able_name_list:
            if it[:] == 'patch':
                locate = begin
                break
            else:
                begin += 1

        json_name = ""
        for i in range(begin + 1):
            json_name += able_name_list[i] + '_'
        json_name += 'json{}'.format(able_name_list[begin + 1][3:]) + '_'

        for i in range(begin + 2, length - 1):
            json_name += able_name_list[i] + '_'
        json_name += '.json'
        json_path = os.path.join(val_json_path,json_name)
        '''
     
        #detect
        json_name = picture.split('.')[0] +'.json'
        json_path = os.path.join(val_json_path,json_name)
        

        img = cv2.imread(picture_path)
        pre_img = img
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
            print('------ begin process {} : {}  ------'.format(picture_name, json_name))
            # init three label
            # pan duan shi fou chu li guo
            sum_lst = []
            # qu chong yong
            point_label = {}
            mid_point_label = []

            if len(json_data) > 1:
                # single person process
                for dic_it in json_data:
                    firstx1 = int(dic_it['rect']['tl']['x'] * img.shape[1])
                    firsty1 = int(dic_it['rect']['tl']['y'] * img.shape[0])
                    firstx2 = int(dic_it['rect']['br']['x'] * img.shape[1])
                    firsty2 = int(dic_it['rect']['br']['y'] * img.shape[0])
                    choice_pair = select(img,dic_it, json_data)
                    mid_main = get_midpoint(firstx1,firsty1,firstx2,firsty2)
                    if len(choice_pair) != 0:
                        for data in choice_pair:
                            cur_lst_first = str(data['idx']) + '.' + str(dic_it['idx'])
                            cur_lst_second = str(dic_it['idx']) + '.' + str(data['idx'])

                            if cur_lst_first not in sum_lst and cur_lst_second not in sum_lst:
                                secondx1 = int(data['rect']['tl']['x'] * img.shape[1])
                                secondy1 = int(data['rect']['tl']['y'] * img.shape[0])
                                secondx2 = int(data['rect']['br']['x'] * img.shape[1])
                                secondy2 = int(data['rect']['br']['y'] * img.shape[0])
                                mid_target = get_midpoint(secondx1,secondy1,secondx2,secondy2)
                                #print(mid_target,mid_main)
                                up = min(firsty1, secondy1)
                                down = max(firsty2, secondy2)
                                left = min(firstx1, secondx1)
                                right = max(firstx2, secondx2)

                                # small image
                                small_img = copy.deepcopy(pre_img[up:down + 1, left:right + 1])
                                #cv2.imshow('small img' , small_img)
                                #cv2.waitKey(100)
                                masked_pic_path = get_pic_mask(pre_img, small_img, data, dic_it, up, down, left, right)
                                net_img = Image.open(masked_pic_path).convert('RGB')
                                net_img = np.array(net_img)

                                #change json
                                newfirstx1,newfirsty1,newfirstx2,newfirsty2 = change(firstx1,firsty1,firstx2,firsty2,up,down,left,right)
                                newsecondx1,newsecondy1,newsecondx2,newsecondy2 = change(secondx1,secondy1,secondx2,secondy2,up,down,left,right)
                                newfirstx1 = int(newfirstx1 * net_img.shape[1])
                                newfirsty1 = int(newfirsty1 * net_img.shape[0])
                                newfirstx2 = int(newfirstx2 * net_img.shape[1])
                                newfirsty2 = int(newfirsty2 * net_img.shape[0])
                                newsecondx1 = int(newsecondx1 * net_img.shape[1])
                                newsecondy1 = int(newsecondy1 * net_img.shape[0])
                                newsecondx2 = int(newsecondx2  * net_img.shape[1])
                                newsecondy2 = int(newsecondy2 * net_img.shape[0])

                                #create depth
                                need_depth_img = small_img
                                depthMap = run(need_depth_img,depthModel,depth_transform,device,True)
                                depthMap = create_depth_mask(depthMap,newfirstx1,newfirsty1,newfirstx2,newfirsty2,newsecondx1,newsecondy1,newsecondx2,newsecondy2)
                                img_depthMap_path = r'/home/lwp/deepc/mainCode/HRNet-Image-Classification-master/tools/depth_temp_pic.png'
                                cv2.imwrite(img_depthMap_path,depthMap)
                                depthMap = cv2.imread(img_depthMap_path, cv2.IMREAD_GRAYSCALE)
                                #cv2.imshow('depthMap' ,depthMap)
                                #cv2.waitKey(10)
                                depthMap = np.array(depthMap)
                                depthMap = np.expand_dims(depthMap,axis = -1)
                                #print(net_img.shape)

                                if transforms_list is not None:
                                    for transforms in transforms_list:
                                        net_img = transforms(net_img)

                                if transforms_list is not None:
                                    for transforms in transforms_list:
                                        depthMap = transforms(depthMap)

                                net_img = net_img.unsqueeze(0)
                                depthMap = depthMap.unsqueeze(0)
                                output = model(net_img,depthMap)
                                _,output = torch.max(output, dim=1)
                                pair_label = 1 if dic_it['group'] == data['group'] else 0
                                if output == 1 :
                                    cv2.line(img, (int(mid_main[1]), int(mid_main[0])), (int(mid_target[1]), int(mid_target[0])), (54,54,205),5)
                                    cv2.circle(img, (int(mid_main[1]), int(mid_main[0])), 5, (51,51,204), 0)
                                    cv2.circle(img, (int(mid_target[1]), int(mid_target[0])), 5, (51, 51, 204), 0)
                                    point_label[dic_it['idx']] = True
                                    point_label[data['idx']] = True
                                    mid_point_label.append([int(mid_main[1]), int(mid_main[0])])
                                    mid_point_label.append([int(mid_target[1]), int(mid_target[0])])
                for dic_it in json_data:
                    if dic_it['idx'] not in point_label.keys():
                        cv2.rectangle(img, (int(dic_it['rect']['tl']['x'] * img.shape[1]),int(dic_it['rect']['tl']['y'] * img.shape[0])), (int(dic_it['rect']['br']['x'] * img.shape[1]),int(dic_it['rect']['br']['y'] * img.shape[0])),(130,51,51), 4)
                        point_label[dic_it['idx']] = True
        cv2.imwrite(os.path.join(val_res_path,'Together' + picture_name),img)
        print('over process : ', picture_name)

if __name__ == '__main__':
    main()