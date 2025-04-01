from __future__ import print_function
import torch
import time
import os
import shutil
from torchvision import models, datasets, transforms
# from torch.utils.data import DataLoader
import torch.nn.functional as F
from functools import partial
import torch.nn as nn
from models import *
from net_util import *
from arg_parser import *


# Helper class for tracking metrics
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Accuracy computation function
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# Load dataset
def load_dataset(dataset_name, args, batch_size=128):

    if args.dataset=='cifar10':
        args.in_planes = 3
        args.input_size=32

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        args.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        print("| Preparing CIFAR-10 dataset...")
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        args.num_outputs = 10
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    elif (args.dataset == 'cifar100'):
        args.in_planes = 3
        args.input_size = 32

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])

        print("| Preparing CIFAR-100 dataset...")
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        args.num_outputs = 100
        
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    return test_loader


# Load model checkpoint
def load_model(net, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
   
    # model = getattr(models, architecture)(pretrained=False)
    net.load_state_dict(checkpoint['state_dict'], strict=False)  # Allow mismatches if minor
    net.eval()

    return net


# Evaluation function (from your `eval_net`)
def eval_net(net, data_loader, use_gpu=False):
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    net.eval()
    total_time = 0
    end_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            if isinstance(outputs, list):
                outputs = outputs[-1]

            loss = F.cross_entropy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1[0].item(), inputs.size(0))
            top5.update(prec5[0].item(), inputs.size(0))

            total_time += (time.time() - end_time)
            end_time = time.time()

    print(f"Loss: {losses.avg:.3f} | Top-1 Accuracy: {top1.avg:.3f}% | Top-5 Accuracy: {top5.avg:.3f}%")
    return losses.avg, top1.avg, top5.avg

# Main function
def main():
    parser = argparse.ArgumentParser()
    
    # args = parser.parse_args()
    args=parse_args()
    
    
    
    architecture = args.arch
    dataset = args.dataset
    
    if dataset=='cifar10':
        args.num_outputs = 10
    if dataset=='cifar100':
        args.num_outputs = 100
        
        
    if args.deconv:
        args.deconv = partial(FastDeconv,bias=args.bias, eps=args.eps, n_iter=args.deconv_iter,block=args.block,sampling_stride=args.stride)
    else:
        args.deconv=None

    if args.delinear:
        args.channel_deconv=None
        if args.block_fc > 0:
            args.delinear = partial(Delinear, block=args.block_fc, eps=args.eps,n_iter=args.deconv_iter)
        else:
            args.delinear = None
    else:
        args.delinear = None
        if args.block_fc > 0:
            args.channel_deconv = partial(ChannelDeconv, block=args.block_fc, eps=args.eps, n_iter=args.deconv_iter,sampling_stride=args.stride)
        else:
            args.channel_deconv = None        

    print("Loading model...")
    
    if args.deconv:
        args.batchnorm=False
        print('************ Batch norm disabled when deconv is used. ************')

    if (not args.deconv) and args.channel_deconv:
        print('************ Channel Deconv is used on the original model, this accelrates the training. If you want to turn it off set --num-groups-final 0 ************')


    if architecture == 'vgg16':
        net = VGG('VGG16',num_classes=args.num_outputs, deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if architecture == 'vgg16d':
        from models.vgg_imagenet import vgg16d
        net = vgg16d('VGG16d',num_classes=args.num_outputs, deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if architecture=='resnet':
        net = ResNet18(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if architecture=='resnet18d':
        from models.resnet_imagenet import resnet18d
        net = resnet18d(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    # if architecture == 'resnet34d':
    #     from models.resnet_imagenet import resnet34d
    #     model = resnet34d(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)
    # if architecture == 'resnet50d':
    #     from models.resnet_imagenet import resnet50d
    #     model = resnet50d(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if architecture=='resnet34':
        net = ResNet34(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if architecture=='resnet50':
        net = ResNet50(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if architecture=='preact':
        #newly added import line
        # from models.preact_resnet import *
        from models.preact_resnet import PreActResNet18
        # PreActResNet18 PreActResNet34
        net = PreActResNet18(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    # net = GoogLeNet()
    if architecture == 'densenet':
        net = densenet_cifar()

    if architecture == 'densenet121':
        net = DenseNet121(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if architecture == 'densenet121d':
        from models.densenet_imagenet import densenet121d
        net = densenet121d(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if architecture == 'efficient':
        # from models.efficientnet import *
        from models.efficientnet import EfficientNetB0
        net = EfficientNetB0(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if architecture == 'resnext':
        from models.resnext import ResNeXt29_32x4d
        net = ResNeXt29_32x4d(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if architecture == 'mobilev2':
        from models.mobilenetv2 import MobileNetV2
        net = MobileNetV2(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)
    # net = MobileNet()
    if architecture == 'dpn':
        from models.dpn import DPN92
        net = DPN92(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)
    # net = ShuffleNetG2()

    if architecture == 'senet':
        from models.senet import SENet18
        net = SENet18(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)

    if architecture == 'pnasnetA':
        from models.pnasnet import PNASNetA
        net = PNASNetA(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)
    if architecture == 'pnasnetB':
        net = PNASNetB(num_classes=args.num_outputs,deconv=args.deconv,delinear=args.delinear,channel_deconv=args.channel_deconv)


    if args.loss=='CE':
        args.criterion = nn.CrossEntropyLoss()
        if args.use_gpu:
            args.criterion = nn.CrossEntropyLoss().cuda()
            #args.criterion = torch.nn.DataParallel(args.criterion)
    elif args.loss=='L2':
        args.criterion = nn.MSELoss()
        if args.use_gpu:
            args.criterion = nn.MSELoss().cuda()    
    
    
    
    model = load_model(net, args.model_path)

    print("Loading dataset...")
    dataloader = load_dataset(args.dataset, args)

    print("Evaluating model...")
    eval_net(model, dataloader)

    
    
if __name__ == '__main__':
    main()


