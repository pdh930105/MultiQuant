# ref. https://github.com/pytorch/examples/blob/master/imagenet/main.py
import argparse
# from ast import arg
import os
import random
import shutil
import time
import warnings
import pdb
import logging
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from imagenet_model.quant_mv1_my import *
from imagenet_model.quant_conv import QConv, QConv_Tra_Mulit
# from imagenet_model.quant_mv1_my_234 import *
# from imagenet_model.quant_conv_234 import QConv, QConv_Tra_Mulit
from src.utils import save_checkpoint, accuracy, AverageMeter, ProgressMeter, Time2Str, setup_logging, CrossEntropyLossSoft
from option import args
import math

best_acc1 = 0
np.random.seed(0)


def main():
    if args.visible_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    args.log_dir += '/' + args.datasetsname  + '_' + args.arch  + '_' + ''.join(args.bit_list) + '/' + Time2Str()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        # cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def data_transforms(args):
    """get transform of dataset"""

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),  # scale=(crop_scale, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_transforms = val_transforms
    return train_transforms, val_transforms, test_transforms


def dataset(train_transforms, val_transforms, test_transforms, args):
    """get dataset for classification"""
    train_set = datasets.ImageFolder(
        os.path.join(args.data, 'train'),
        transform=train_transforms)
    val_set = datasets.ImageFolder(
        os.path.join(args.data, 'val'),
        transform=val_transforms)
    test_set = None
    return train_set, val_set, test_set


def data_loader(train_set, val_set, test_set, args):
    """get data loader"""

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        # shuffle=False,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=args.workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.eval_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers)
    return train_loader, val_loader, train_sampler


def init_quant_model(model, args, ngpus_per_node):
    for layers in model.modules():
        if hasattr(layers, 'init'):
            layers.init.data.fill_(1)

    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(args.batch_size / ngpus_per_node), shuffle=True,
        num_workers=int((args.workers + ngpus_per_node - 1) / ngpus_per_node), pin_memory=True)
    iterloader = iter(train_loader)

    model.to(args.gpu)
    model.train()
    # xxxx
    # model.eval()
    with torch.no_grad():
        for i in range(len(args.bit_list)):
            print('init for bit-width: ', args.bit_list[i])
            images, labels = next(iterloader)
            images = images.to(args.gpu)
            labels = labels.to(args.gpu)
            for name, layers in model.named_modules():
                if hasattr(layers, 'act_bit'):
                    setattr(layers, "act_bit", int(args.bit_list[i]))
                if hasattr(layers, 'weight_bit'):
                    setattr(layers, "weight_bit", int(args.bit_list[i]))
            model.forward(images)

    for layers in model.modules():
        if hasattr(layers, 'init'):
            layers.init.data.fill_(0)


def cal_params(model, args):
    trainable_params = list(model.parameters())
    model_params = []
    quant_params = []
    bn_params = []

    for name, layers in model.named_modules():
        # if isinstance(layers, QConv):
        if isinstance(layers, QConv) or isinstance(layers, QConv_Tra_Mulit):
            model_params.append(layers.weight)
            if layers.bias is not None:
                model_params.append(layers.bias)
            if layers.quan_weight:
                if isinstance(layers.lW, nn.ParameterList):
                    for x in layers.lW:
                        quant_params.append(x)
                    for x in layers.uW:
                        quant_params.append(x)
                else:
                    quant_params.append(layers.lW)
                    quant_params.append(layers.uW)
            if layers.quan_act:
                if isinstance(layers.lA, nn.ParameterList):
                    for x in layers.lA:
                        quant_params.append(x)
                    for x in layers.uA:
                        quant_params.append(x)
                else:
                    quant_params.append(layers.lA)
                    quant_params.append(layers.uA)
            if layers.quan_act or layers.quan_weight:
                if hasattr(layers, 'output_scale'):
                    if isinstance(layers.output_scale, nn.ParameterList):
                        for x in layers.output_scale:
                            quant_params.append(x)
                    else:
                        quant_params.append(layers.output_scale)

        elif isinstance(layers, nn.Conv2d):
            model_params.append(layers.weight)
            if layers.bias is not None:
                model_params.append(layers.bias)

        elif isinstance(layers, nn.Linear):
            model_params.append(layers.weight)
            if layers.bias is not None:
                model_params.append(layers.bias)

        elif isinstance(layers, nn.SyncBatchNorm) or isinstance(layers, nn.BatchNorm2d):
            if layers.bias is not None:
                bn_params.append(layers.weight)
                bn_params.append(layers.bias)

    log_string = "total params: {}, trainable model params: {}, trainable quantizer params: {}, trainable bn params: {}".format(
        sum(p.numel() for p in trainable_params), sum(p.numel() for p in model_params),
        sum(p.numel() for p in quant_params), sum(p.numel() for p in bn_params)
    )
    assert sum(p.numel() for p in trainable_params) == (sum(p.numel() for p in model_params)
                                                        + sum(p.numel() for p in quant_params)
                                                        + sum(p.numel() for p in bn_params))
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank  == 0):
        logging.info(log_string)

    return model_params, quant_params, bn_params


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu == 0:
        setup_logging(os.path.join(args.log_dir, "log.txt"))
        arg_dict = vars(args)
        log_string = 'configs\n'
        for k, v in arg_dict.items():
            log_string += "{}: {}\n".format(k, v)
        logging.info(log_string)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    # create model
    model_class = globals().get(args.arch)
    model = model_class(args, groups=args.groups)

    args.bit_list = [eval(x) for x in args.bit_list]
    if args.rank % ngpus_per_node == 0:
        # print(model)
        logging.info('args.bit_list: {}'.format(args.bit_list))


    ### initialze quantizer parameters
    if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):  # do only at rank=0 process
        if not args.evaluate:
            init_quant_model(model, args, ngpus_per_node)

    ###
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            print('have set gpu in args.gpu', args.gpu)
            # When using a single GPU per process and per DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.eval_size = args.batch_size
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  ########## SyncBatchnorm
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            args.eval_size = args.batch_size
            print('do not set gpu in args.gpu!')
            # DistributedDataParallel will divide and allocate batch_size to all available GPUs if device_ids are not set
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, find_unused_parameters=True)  ########## SyncBatchnorm
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        args.eval_size = args.batch_size
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        args.eval_size = args.batch_size * ngpus_per_node
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            print('torch.nn.DataParallel(model).cuda()!')
            model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    train_transforms, val_transforms, test_transforms = data_transforms(args)
    train_set, val_set, test_set = dataset(train_transforms, val_transforms, test_transforms, args)
    train_loader, val_loader, train_sampler = data_loader(train_set, val_set, test_set, args)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    model_params, quant_params, bn_params = cal_params(model, args)

    if args.optimizer_m == 'SGD':
        # all depthwise convolution (N, 1, x, x) has no weight decay
        model_params_tmp = []
        for params in model_params:
            ps = list(params.size())
            if len(ps) == 4 and ps[1] == 1:
                weight_decay = 0
                # weight_decay = args.weight_decay
            else:
                weight_decay = args.weight_decay
            item = {'params': params, 'weight_decay': weight_decay,
                    'lr': args.lr_m, 'momentum': 0.9,}
                    # 'nesterov': True}
            model_params_tmp.append(item)
        # all bn has no weight decay
        for params in bn_params:
            weight_decay = 0
            # weight_decay = args.weight_decay
            # momentum = 0
            item = {'params': params, 'weight_decay': weight_decay,
                    'lr': args.lr_m, 'momentum': 0.9,}
                    # 'nesterov': True}
            model_params_tmp.append(item)
        optimizer_m = torch.optim.SGD(model_params_tmp)

    if args.optimizer_q == 'SGD':
        optimizer_q = torch.optim.SGD(quant_params, lr=args.lr_q)
    elif args.optimizer_q == 'Adam':
        optimizer_q = torch.optim.Adam(quant_params, lr=args.lr_q)

    scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m, T_max=args.epochs, eta_min=args.lr_m_end)
    scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_q, T_max=args.epochs, eta_min=args.lr_q_end)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank  == 0):
                logging.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            # if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                # best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_m.load_state_dict(checkpoint['optimizer_m'])
            optimizer_q.load_state_dict(checkpoint['optimizer_q'])
            scheduler_m.load_state_dict(checkpoint['scheduler_m'])
            scheduler_q.load_state_dict(checkpoint['scheduler_q'])
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank  == 0):
                logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank  == 0):
                logging.info("=> no checkpoint found at '{}'".format(args.resume))

    start = time.time()
    epoch = 0
    if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

        acc1 = np.zeros(len(args.bit_list))
        for i in range(0, len(args.bit_list)):
            for name, layers in model.named_modules():
                if hasattr(layers, 'act_bit'):
                    setattr(layers, "act_bit", int(args.bit_list[i]))
                if hasattr(layers, 'weight_bit'):
                    setattr(layers, "weight_bit", int(args.bit_list[i]))
            acc1[i] = validate(val_loader, model, criterion, args, epoch, weight_bit=int(args.bit_list[i]),
                                act_bit=int(args.bit_list[i]))
            logging.info("act_bit: {}, weight_bit: {}, acc1: {}".format(int(args.bit_list[i]), int(args.bit_list[i]), acc1[i]))
            
        acc1_avg = np.mean(acc1)
        logging.info("Epoch: [{}]".format(epoch) + "[GPU{}]".format(args.gpu) + "current avg acc@1: {}".format(acc1_avg))
        is_best = acc1_avg > best_acc1
        best_acc1 = max(acc1_avg, best_acc1)

        log_string = "Epoch: [{}]".format(epoch) + "[GPU{}]".format(args.gpu) + \
                        ' 1 epoch spends {:2d}:{:.2f} mins\t remain {:2d}:{:2d} hours\n'. \
                            format(int((time.time() - start) // 60), (time.time() - start) % 60,
                                    int((args.epochs - epoch - 1) * (time.time() - start) // 3600),
                                    int((args.epochs - epoch - 1) * (time.time() - start) % 3600 / 60))
        for i in range(0, len(args.bit_list)):
                log_string += '[{}bit] {:.3f}\t'.format(args.bit_list[i], acc1[i])
        log_string += "current best acc@1: {}\n".format(best_acc1)
        logging.info(log_string)
        # wait for test in rank0 GPU
        # torch.distributed.barrier()



def validate(val_loader, model, criterion, args, epoch, weight_bit, act_bit):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        args=args,
        prefix="Epoch: [{}]".format(epoch) + '[Weight {}bit] Test: '.format(weight_bit)
               + '[Act {}bit] Test: '.format(act_bit))

    # switch to evaluate mode
    model.eval()

    # from pytorchcv.model_provider import get_model as ptcv_get_model
    # trained_model = ptcv_get_model('mobilenet_w1', pretrained=True)
    # trained_model.cuda()
    # trained_model.eval()
    #
    # import IPython
    # IPython.embed()

    with torch.no_grad():
        end = time.time()

        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                # print(i)
                progress.display(i)

        log_string = "Epoch: [{}]".format(epoch) + '[W{}bitA{}bit]'.format(weight_bit, act_bit) + "[GPU{}]".format(
            args.gpu) + ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank  == 0):
            logging.info(log_string)

    return top1.avg


if __name__ == '__main__':
    main()
