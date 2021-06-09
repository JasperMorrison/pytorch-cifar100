# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

from utils import *

from utils import get_custom_training_dataloader, get_custom_test_dataloader

#ASL
from loss.loss import *
#Focal loss
from loss.focal_loss import *

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, sample in enumerate(cifar100_training_loader):
        images = sample['image']
        labels = sample['label']

        if not args.asl:
            labels = np.argmax(labels, axis=1)

        if epoch <= args.warm:
            warmup_scheduler.step()

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = net(images)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        '''
        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        '''
        '''
        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))
        '''

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s LR: {:0.6f}'.format(epoch, finish - start, optimizer.param_groups[0]['lr']))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for sample in cifar100_test_loader:
        images = sample['image']
        labels = sample['label']

        if not args.asl:
            labels = np.argmax(labels, axis=1)

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)

        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        if args.asl:
            labels = torch.argmax(labels, axis=1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    '''
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    '''
    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, consumed:{:.2f}s'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fine', type=int, default=0, help='fine-ture free layers, e.g. 4')
    parser.add_argument('-classes', type=int, required=True, help='class number, for example 5')
    parser.add_argument('-size', type=int, default=64, help='image size, for example 64')
    parser.add_argument('-data', type=str, required=True, help='dataset path, csv file')
    parser.add_argument('-test', type=str, required=True, help='test dataset path')
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-asl', action='store_true', default=False, help='use AsymmetricLoss')
    parser.add_argument('-focal_loss', action='store_true', default=False, help='use Focal loss')
    parser.add_argument('-alpha', nargs='+', help='Set focal loss alpha')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=3, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-pretrained', action='store_true', default=False, help='using pytorch pretrained model')
    args = parser.parse_args()

    if args.pretrained:
        net = get_pretrained_network(args)
    else:
        net = get_network(args)
    if args.fine:
        layers = 0
        total_layers = 0
        for child in net.children():
            total_layers += 1
        lock_layers = total_layers - args.fine
        print('total_layers', total_layers, 'lock layers', lock_layers) 
        for child in net.children():
            if layers > lock_layers:
                break
            layers += 1
            for param in child.parameters():
                param.requires_grad = False

    #data preprocessing:
    cifar100_training_loader = get_custom_training_dataloader(
        args.data,
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=8,
        batch_size=args.b,
        img_size=args.size,
        shuffle=True
    )

    cifar100_test_loader = get_custom_test_dataloader(
        args.test,
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=8,
        batch_size=args.b,
        img_size=args.size,
        shuffle=True
    )

    def poly_lr_scheduler(epoch, num_epochs=settings.EPOCH, power=0.9):
        print("epoch & num_epochs", epoch, num_epochs)
        return (1 - epoch/num_epochs)**power


    if args.asl:
        loss_function = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.0, disable_torch_grad_focal_loss=True)
    elif args.focal_loss:
        alpha=[1] * args.classes # edit it
        if args.alpha:
            alpha = args.alpha
        loss_function = focal_loss(alpha=alpha, gamma=2, num_classes=args.classes)
    else:
        loss_function = nn.CrossEntropyLoss()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    #train_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr_scheduler)
    #train_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.00001, eps=1e-08)

    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    # lcr, set size, same as vgg16 network
    input_tensor = torch.Tensor(1, 3, 64, 64).cuda()
    #input_tensor = torch.Tensor(1, 3, 32, 32).cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[0] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
