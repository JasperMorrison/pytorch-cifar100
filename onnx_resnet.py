"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import torchvision.models as models
import argparse

def get_pretrained_network(args):
    if args.net == 'resnet18':
        resnet18 = models.resnet18(pretrained=True)
        resnet18.fc = nn.Linear(512, args.classes)
        net = resnet18
    elif args.net == 'resnet50':
        resnet50 = models.resnet50(pretrained=True)
        resnet50.fc = nn.Linear(512, args.classes)
        net = resnet50
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net

def get_network(args):
    from models.resnet import resnet18
    return resnet18(args.classes, cut_layer=args.cut_layer) 

parser = argparse.ArgumentParser()
parser.add_argument('-pretrained', action='store_true', default=False, help='use pretrained at torchvision or not')
parser.add_argument('-classes', type=int, required=True, help='class number, for example 5')
parser.add_argument('-weights', type=str, required=True, help='the weights of model')
args = parser.parse_args()

args.net = 'resnet18'

if args.pretrained:
    model = get_pretrained_network(args)
else:
    model = get_network(args)

checkpoint = torch.load(args.weights, map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

example = torch.rand(1,3,64,64)
torch.onnx.export(model, example, "resnet.onnx", verbose=True)


