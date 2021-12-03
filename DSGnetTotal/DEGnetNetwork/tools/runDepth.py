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


def run(img, model_path, model_type="large", optimize=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load network
    if model_type == "large":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
    elif model_type == "small":
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True,
                               non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
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

    model.eval()
    if optimize == True:
        rand_example = torch.rand(1, 3, net_h, net_w)
        model(rand_example)
        traced_script_module = torch.jit.trace(model, rand_example)
        model = traced_script_module

        if device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()

    model.to(device)

    #img
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    img_input = transform({"image": img})["image"]

    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        if optimize == True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
                .squeeze()
                .cpu()
                .numpy()
        )
    #output
    filename = os.path.join(r'/home/wh/lwp/hrnet-class/tempfpm', 'tempPic')
    depth_map = utilsDepth.write_depth(filename,prediction, bits=2)
    return depth_map