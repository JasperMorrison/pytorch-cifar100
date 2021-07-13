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

import os
from PIL import Image
from pathlib import Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, help='not used')
    parser.add_argument('-warm', type=int, help='not used')
    parser.add_argument('-lr', type=float, help='not used')
    parser.add_argument('-fine', type=float, help='not used')
    parser.add_argument('-resume', action='store_true', default=False, help='not used')
    
    parser.add_argument('-classes', type=int, required=True, help='class number, for example 5')
    parser.add_argument('-size', type=int, default=64, help='image size, for example 64')
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-test', type=str, required=True, help='test data dir')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-pretrained', action='store_true', default=False, help='using pytorch pretrained model')
    args = parser.parse_args()

    if args.pretrained:
        net = get_pretrained_network(args)
    else:
        net = get_network(args)

    cifar100_test_loader = get_custom_test_dataloader(
        args.test,
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=1,
        img_size=args.size,
        batch_size=args.b,
    )

    net.load_state_dict(torch.load(args.weights))
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    train_class_correct = list( i for i in range(args.classes) )
    correct_list = list(0. for i in range(args.classes))
    total_list = list(0. for i in range(args.classes))

    with torch.no_grad():
        for n_iter, sample in enumerate(cifar100_test_loader):
            image = sample['image']
            labels = sample['label']
            path = sample['img_path']

            labels = np.argmax(labels, axis=1)

            '''
            from torchvision.utils import save_image
            for img_index, iter_img in enumerate(image):
                save_img_path = Path("output") / (str(labels[img_index].numpy())) / (str(n_iter) + "_" + str(img_index) + ".jpg")
                os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
                save_image(iter_img, save_img_path)
                print(path[img_index], "save to", save_img_path)
            '''
            if args.gpu:
                image = image.cuda()
                labels = labels.cuda()

            output = net(image)
            #_, pred = output.topk(5 if args.classes > 5 else args.classes - 1, 1, largest=True, sorted=True)
            _, pred = output.topk(1, 1, largest=True, sorted=True)

            labels = labels.view(labels.size(0), -1).expand_as(pred)
            correct = pred.eq(labels).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

            prediction = torch.argmax(output, 1)
            res = prediction == labels
            for label_idx in range(len(labels)):
                label_single = labels[label_idx]
                correct_list[label_single] += res[label_idx].item()
                total_list[label_single] += 1
                t_or_f = res[label_idx].item()
                if not t_or_f:
                    print(path[label_idx], "label:" ,label_single.cpu().numpy(), "predict:", prediction[label_idx].cpu().numpy())
                    print(labels, prediction)

    print("Total correct:", sum(correct_list))
    print("Total:",sum(total_list))
    acc_str = 'Accuracy: %f\n'%(sum(correct_list)/sum(total_list))
    for acc_idx in range(len(train_class_correct)):
        try:
            acc = correct_list[acc_idx]/total_list[acc_idx] # 某个类的正确样本数/总样本数
        except:
            acc = 0
        finally:
            acc_str += '\tclassID:%d\tacc:%f\t\n'%(acc_idx, acc)

    print(acc_str)
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
