'''
code: https://github.com/iBelieveCJM/Tricks-of-Semi-supervisedDeepLeanring-Pytorch
Follow CUDA_VISIBLE_DEVICES=$1 python main.py --dataset=cifar10 --sup-batch-size=100 --usp-batch-size=100 --label-exclude=False --num-labels=4000 --arch=cnn13 --model=ipslab2013v2 --usp-weight=1.0 --optim=sgd --epochs=400 --lr=0.1 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=cos --min-lr=1e-4 --rampup-length=80 --rampdown-length=50 --data-idxs=False --save-freq=100 2>&1 | tee results/ipslab2013v2_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt
'''
#!coding:utf-8
from __future__ import print_function

import argparse
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
from progress.bar import Bar as Bar
from tensorboardX import SummaryWriter

from dataset.cifar100 import get_cifar100_dataloaders
from helper.util import AverageMeter, accuracy
from models import model_dict


parser = argparse.ArgumentParser(description='PyTorch pseudolabel Training')
# Optimization options
parser.add_argument('--ood', default='tin', type=str, choices=['tin', 'places'])
parser.add_argument('--arch', default='wrn_40_1', type=str, choices=['wrn_40_1', 'resnet8x4', 'ShuffleV1'],
                    help='dataset name')
parser.add_argument('--th', default=0, type=float, choices=[0, 0.8, 0.85, 0.9, 0.95])
parser.add_argument('--epochs', default=400, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=100, type=int, metavar='N', help='train batchsize')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
# Device options
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--train-iteration', type=int, default=1024, help='Number of iteration per epoch')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
args.out = './Results/PseudoLabel/' + str(args.arch) + '_ood_' + str(args.ood) + '_th_' + str(args.th)
os.makedirs(args.out, exist_ok=True)

best_acc = 0  # best test accuracy
n_cls = 100  # for cifar100


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0

    return warpper


def main():
    global best_acc
    rampup = exp_rampup(30)
    # Data
    labeled_trainloader, unlabeled_trainloader, test_loader, n_data = get_cifar100_dataloaders(
        batch_size=args.batch_size, num_workers=8, is_instance=True, is_sample=False, ood=args.ood)
    print(len(labeled_trainloader), len(unlabeled_trainloader))

    # Model
    print("==> creating %s" % args.arch)

    def create_model(ema=False):
        model = model_dict[args.arch](num_classes=n_cls)
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    cudnn.benchmark = True
    print(' Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    start_epoch = 0
    writer = SummaryWriter(args.out)
    test_accs = []
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_loss, train_loss_x, train_loss_u = train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
                                                       epoch, rampup, use_cuda)
        test_loss, test_acc = validate(test_loader, model, criterion, epoch, use_cuda, mode='Test Stats ')
        scheduler.step()
        step = args.train_iteration * (epoch + 1)
        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/test_loss', test_loss, step)
        writer.add_scalar('accuracy/test_acc', test_acc, step)

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        test_accs.append(test_acc)
        print('Best acc:')
        print(best_acc)
        print('Mean acc:')
        print(np.mean(test_accs[-20:]))
    writer.close()
    print('Best acc:')
    print(best_acc)
    print('Mean acc:')
    print(np.mean(test_accs[-20:]))

    with open(args.out + '/res_%s.txt' % time.ctime(), 'w') as f:
        f.write('%.4f' % best_acc)
        f.write('\n')
        f.write('%.4f' % np.mean(test_accs[-20:]))


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, epoch, rampup, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    iteration = max(len(labeled_trainloader), len(unlabeled_trainloader))
    iteration = min(iteration, 1024)
    for batch_idx in range(iteration):
        try:
            inputs_x, targets_x, index_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, index_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()

        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()

        outputs = model(inputs_x)
        Lx = F.cross_entropy(outputs, targets_x)
        unlab_outputs = model(inputs_u)
        with torch.no_grad():
            if args.th > 0:
                max_probs, targets_u = torch.max(unlab_outputs, dim=-1)
                mask = max_probs.ge(args.th).float()
            else:
                iter_unlab_pslab = unlab_outputs.max(1)[1]
                iter_unlab_pslab.detach_()

        if args.th > 0:
            Lu = (F.cross_entropy(unlab_outputs, targets_u, reduction='none') * mask).mean()
        else:
            Lu = F.cross_entropy(unlab_outputs, iter_unlab_pslab)

        u_w = rampup(epoch)
        loss = Lx + Lu * u_w

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) ETA:{eta:}|Loss:{loss:.2f}|Loss_x:{loss_x:.2f}|Loss_u:{loss_u:.2f}|w_u:{w_u:0.2f}'.format(
            batch=batch_idx + 1,
            size=iteration,
            eta=bar.eta_td,
            loss=losses.avg,
            loss_x=losses_x.avg,
            loss_u=losses_u.avg,
            w_u=u_w,
        )
        bar.next()
    bar.finish()
    return (losses.avg, losses_x.avg, losses_u.avg,)


def validate(valloader, model, criterion, epoch, use_cuda, mode):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size})|ETA: {eta:}|Loss:{loss:.2f}|top1:{top1:.2f}|top5:{top5:.2f}'.format(
                batch=batch_idx + 1,
                size=len(valloader),
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))







if __name__ == '__main__':
    main()
