""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy,random
import numpy as np

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from PIL import Image
import torch.nn as nn
from torchvision import datasets

import pandas as pd
from randaugment import RandAugment

class ImageCut(object):
    def __init__(self, h_begin=0.0, h_scale=1.0, w_scale=1.0, p=0.5):
        self.h_scale = h_scale
        self.w_scale = w_scale
        self.h_begin = h_begin
        self.p = p
    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            w,h = img.size
            box = (0, int(h*self.h_begin), int(w*self.w_scale), int(h*self.h_scale) + int(h*self.h_begin))
            img = img.crop(box)
        return img 

class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p

    # transform 会调用该方法
    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        # 如果随机概率小于 seld.p，则执行 transform
        if random.uniform(0, 1) < self.p:
            # 把 image 转为 array
            img_ = np.array(img).copy()
            # 获得 shape
            h, w, c = img_.shape
            # 信噪比
            signal_pct = self.snr
            # 椒盐噪声的比例 = 1 -信噪比
            noise_pct = (1 - self.snr)
            # 选择的值为 (0, 1, 2)，每个取值的概率分别为 [signal_pct, noise_pct/2., noise_pct/2.]
            # 椒噪声和盐噪声分别占 noise_pct 的一半
            # 1 为盐噪声，2 为 椒噪声
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = np.random.randint(256)   # 盐噪声
            img_[mask == 2] = np.random.randint(50)   # 椒噪声
            # 再转换为 image
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        # 如果随机概率大于 seld.p，则直接返回原图
        else:
            return img

class MyDataset(Dataset):
  def __init__(self , csv, transform=None ):
    self.csv = csv
    self.df = pd.read_csv(self.csv)
    self.img_dir = self.csv.split('.')[:-1][0]
    self.transforms = transform

  def __getitem__(self,idx):
    d = self.df.iloc[idx]
    img_path = os.path.join(self.img_dir,d.image)
    image = Image.open(img_path).convert("RGB")
    label = torch.tensor(d[1:].tolist() , dtype=torch.float32)

    if self.transforms is not None:
      image = self.transforms(image)

    sample = {'image':image, 'label':label, 'img_path':img_path}
    return sample

  def __len__(self):
    return len(self.df)

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
 
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        sample = {'image':tuple_with_path[0], 'label':tuple_with_path[1], 'img_path':tuple_with_path[2]}

        '''
        from skimage import io
        import cv2
        img = io.imread(tuple_with_path[2])
        img = np.float32(cv2.resize(img, (128,128))) / 255
        image = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))  # channel first
        image = image[np.newaxis, ...]  # 增加batch维
        torch.tensor(image, requires_grad=True)
        sample = {'image':image, 'label':tuple_with_path[1], 'img_path':tuple_with_path[2]}
        '''
        
        return sample

def get_pretrained_network(args):
    #import torchvision.models as models
    import models.pretrained_resnet as models
    if args.net == 'resnet18':
        resnet18 = models.resnet18(pretrained=True)
        resnet18.fc = nn.Linear(512, args.classes)
        net = resnet18
    elif args.net == 'resnet50':
        resnet50 = models.resnet50(pretrained=True)
        resnet50.fc = nn.Linear(512*4, args.classes)
        net = resnet50
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(args.classes)
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(args.classes)
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56(args.classes)
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92(args.classes)
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net

# Only num_worker = 1
class SaveImage(object):
    def __init__(self, bs=32, path="./output"):
        self.path = path
        self.batchsize = bs
        self.num_batch = 0
        self.num = 0
        self.max_num = 10 # max number in one batch
        self.max_num = self.max_num if self.max_num < bs else bs 
        self.init()
        print("New SaveImage obj")
    
    def init(self):
        self.num = 0
        self.num_batch += 1

    def __call__(self, img):
        self.num += 1
        if self.num <= self.max_num:
            img_path = os.path.join(self.path, str(self.num_batch) + "_" + str(self.num) + ".jpg")
            img.save(img_path)
            print("save to", img_path)
        if self.num == self.batchsize:
            self.init()
        return img

class ToHSV(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = img.convert('HSV')
        return img

class MyCustomCrop(object):
    """自定义裁剪
    Args:
        size（float）: persent to crop, width min, height min
        p (float): 概率值
    """

    def __init__(self, snr, p=0.95):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.size = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = numpy.array(img).copy()  # 转imge到ndrrary形式
            img_bak = img_.copy()
            img_bak[:,:] = (0,0,0)
            h, w, c = img_.shape
            w_p = random.uniform(self.size[0], 1.0)
            h_p = random.uniform(self.size[1], 1.0)
            crop_w = int(w * w_p)
            crop_h = int(h * h_p)
            w_a = random.randint(0,w-crop_w)
            h_a = random.randint(0,int(min(h/4, h-crop_h)))
            img_bak[h_a:h_a+crop_h, w_a:w_a+crop_w] = img_[h_a:h_a+crop_h, w_a:w_a+crop_w]
            img = Image.fromarray(img_bak.astype('uint8')).convert('RGB')
            #img.save("./test.jpg")
        return img

def get_custom_training_dataloader(path, mean, std, batch_size=16, num_workers=2, img_size=64, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    # example
    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=25, translate=(0.2, 0.2), scale=(0.9, 1.1), shear=15),
        #transforms.Resize((img_size, img_size)),
        transforms.Resize((int(img_size*1.3), int(img_size*1.3))),
        transforms.RandomCrop(img_size, padding=4),
        AddPepperNoise(0.95,0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.01),
        #RandAugment(),
        #ToHSV(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.95, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=114/255.0, inplace=False),
    ])

    cifar100_training = MyDataset(path, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_custom_test_dataloader(path, mean, std, batch_size=16, num_workers=2, img_size=64, shuffle=True):
    transform_test = transforms.Compose([
        #transforms.RandomAffine(degrees=25, translate=(0.2, 0.2), scale=(0.9, 1.1), shear=15),
        transforms.Resize((img_size, img_size)),
        #AddPepperNoise(0.95,0.5),
        #RandAugment(),
        #ToHSV(),
        transforms.ToTensor(),
    ])
    cifar100_test = MyDataset(path, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def get_nornmal_training_dataloader(path, mean, std, batch_size=16, num_workers=2, img_size=64, shuffle=True, target_transform=None):
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        RandAugment(),
        #SaveImage(),
        transforms.ToTensor(),
    ])
    cifar100_training = ImageFolderWithPaths(root=path, transform=transform_train, target_transform=target_transform)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_normal_test_dataloader(path, mean, std, batch_size=16, num_workers=2, img_size=64, shuffle=True, target_transform=None):
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    cifar100_test = ImageFolderWithPaths(root=path, transform=transform_test, target_transform=target_transform)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]
