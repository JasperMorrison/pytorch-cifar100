# -*- coding: utf-8 -*-
#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader
from utils import *

import os, shutil
from PIL import Image
from pathlib import Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, help='not used')
    parser.add_argument('-warm', type=int, help='not used')
    parser.add_argument('-lr', type=float, help='not used')
    parser.add_argument('-fine', type=float, help='not used')
    parser.add_argument('-resume', action='store_true', default=False, help='not used')
    
    parser.add_argument('-out', type=str, required=True, help='output dir')
    parser.add_argument('-classes', type=int, required=True, help='class number, for example 5')
    parser.add_argument('-size', type=int, default=64, help='image size, for example 64')
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-test', type=str, required=True, help='test data dir')
    parser.add_argument('-normal', action='store_true', default=True, help='use ImageFolder as dataset')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-pretrained', action='store_true', default=False, help='using pytorch pretrained model')
    args = parser.parse_args()

    if args.pretrained:
        net = get_pretrained_network(args)
    else:
        net = get_network(args)

    if not args.normal:
        dataloader_func = get_custom_test_dataloader
    else:
        dataloader_func = get_normal_test_dataloader

    cifar100_test_loader = dataloader_func(
        args.test,
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=1,
        img_size=args.size,
        batch_size=args.b,
    )

    net.load_state_dict(torch.load(args.weights))
    net.eval()

    total_list = list(0. for i in range(args.classes))

    with torch.no_grad():
        for n_iter, sample in enumerate(cifar100_test_loader):
            image = sample['image']
            path = sample['img_path']

            if args.gpu:
                image = image.cuda()

            output = net(image)
            prediction = torch.argmax(output, 1)
            print(prediction, path)
            for i,pred in enumerate(prediction):
                pred = pred.cpu().numpy()
                img_path = path[i]
                out_path = os.path.join(args.out, str(pred))
                if not os.path.exists(out_path):
                    os.makedirs(out_path, exist_ok=True)
                shutil.copy(img_path, out_path + "/" + os.path.basename(img_path))

    print("Done")
