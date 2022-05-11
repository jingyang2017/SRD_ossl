from __future__ import print_function
import argparse
import os
import shutil
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from progress.bar import Bar as Bar
from dataset.mtcr_data import get_cifar100
from models import model_dict
from helper.util import AverageMeter, accuracy
from tensorboardX import SummaryWriter
from skimage.filters import threshold_otsu
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='PyTorch MTCR Training')
# Optimization options
parser.add_argument('--ood', default='tin', type=str, choices=['tin', 'places'])
parser.add_argument('--arch', default='resnet8x4', type=str, choices=['wrn_40_1', 'resnet8x4', 'ShuffleV1'],
                    help='dataset name')
parser.add_argument('--num_classes', type=int, default=100, help='cifar100 classes')
parser.add_argument('--pre-epochs', default=100, type=int, metavar='N', help='number of warm-up epochs to run')
parser.add_argument('--epochs', default=1024, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')

# Method options
parser.add_argument('--val-iteration', type=int, default=1024,
                    help='number of iteration of each epoch')

# MixMatch options
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
args.out = './Results/MTCR/' + str(args.arch) + '_ood_' + str(args.ood)
os.makedirs(args.out, exist_ok=True)

use_cuda = torch.cuda.is_available()
print('use cuda', use_cuda)

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0  # best test accuracy


class new_model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        '''
        arg.arch -- model_name
        args.num_classes--class number
        '''
        self.model = model_dict[args.arch](num_classes=args.num_classes)
        self.head = torch.nn.Linear(self.model.feat_dim, 1)

    def forward(self, x, is_feat=False):
        feats, logits = self.model(x, True)
        if is_feat:
            feat = feats[-1].view(feats[-1].size(0), -1)
            return logits, self.head(feat)  # return logits, logits_open
        else:
            return logits


def main():
    global best_acc
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    train_labeled_set, train_unlabeled_set, domain_set, test_set = get_cifar100(transform_train=transform_train,
                                                                                transform_val=transform_val,
                                                                                out_dataset=args.ood)

    domain_trainloader = DataLoader(domain_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    labeled_trainloader = DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                     drop_last=True)
    unlabeled_trainloader = DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                       drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    print("==> creating model")

    def create_model(args=None, ema=False):
        model = new_model(args)
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model(args=args, ema=False)
    ema_model = create_model(args=args, ema=True)

    model = nn.DataParallel(model)
    ema_model = nn.DataParallel(ema_model)

    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)
    start_epoch = 0

    writer = SummaryWriter(args.out)
    test_accs = []
    if not args.resume:
        print('no pretrained model')
        for epoch in range(args.pre_epochs):
            print('\nDomain Epoch: [%d | %d] LR: %f' % (epoch + 1, args.pre_epochs, state['lr']))
            train_loss, train_loss_x, train_loss_u = domain_train(domain_trainloader, model, optimizer, use_cuda)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, False, filename='pretrain.pth.tar')
    else:
        checkpoint = torch.load(args.resume, 'cpu')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        train_loss, train_loss_x, train_loss_u, nsamples = train(domain_trainloader, labeled_trainloader,
                                                                 unlabeled_trainloader, model, optimizer, ema_optimizer,
                                                                 train_criterion, epoch, use_cuda)
        test_loss, val_acc = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats')
        writer.add_scalar('losses/train_loss', train_loss, epoch)
        writer.add_scalar('losses/test_loss', test_loss, epoch)
        writer.add_scalar('accuracy/test_acc', val_acc, epoch)
        writer.add_scalar('used_un', nsamples, epoch)

        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'acc': val_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        test_accs.append(val_acc)
        print('best acc:')
        print(best_acc)
        print('Mean acc:')
        print(np.mean(test_accs[-20:]))

    writer.close()
    print('best acc:')
    print(best_acc)
    print('Mean acc:')
    print(np.mean(test_accs[-20:]))

    with open(args.out + '/res_%s.txt' % time.ctime(), 'w') as f:
        f.write('%.4f' % best_acc)
        f.write('\n')
        f.write('%.4f' % np.mean(test_accs[-20:]))


def domain_train(domain_trainloader, model, optimizer, use_cuda):
    # ood binary classification for labeled and unlabeled data
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)

    results = np.zeros((len(domain_trainloader.dataset)), dtype=np.float32)

    model.train()
    for batch_idx, (inputs, domain_labels, indexs) in enumerate(domain_trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, domain_labels = inputs.cuda(), domain_labels.cuda(non_blocking=True)

        logits = model(inputs, is_feat=True)[1]
        probs = torch.sigmoid(logits).view(-1)
        print(logits.size())
        Ld = F.binary_cross_entropy_with_logits(logits, domain_labels.view(-1, 1))
        results[indexs.detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()
        loss = Ld
        # record loss
        losses.update(loss.item(), inputs.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}'.format(
            batch=batch_idx + 1,
            size=len(domain_trainloader),
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg)
        bar.next()
    bar.finish()
    domain_trainloader.dataset.label_update(results)
    return (losses.avg, losses_x.avg, losses_u.avg)


def train(domain_trainloader, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion,
          epoch, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    losses_d = AverageMeter()

    ws = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    train_iter = iter(domain_trainloader)

    # ood detection
    results = np.zeros((len(domain_trainloader.dataset)), dtype=np.float32)
    # Get OOD scores of unlabeled samples
    weights = domain_trainloader.dataset.soft_labels[len(labeled_trainloader.dataset):].copy()
    # Calculate threshold by otsu
    th = threshold_otsu(weights.reshape(-1, 1))

    # Select samples having small OOD scores as ID data
    '''
    Attention:
    Weights is the (1 - OOD score) in this implement, which is different from the paper.
    So a larger weight means the data is more likely to be ID data.
    '''
    subset_indexs = np.arange(len(unlabeled_trainloader.dataset))[weights >= th]
    if len(subset_indexs) < args.batch_size:
        nsamples = len(subset_indexs)
        use_ood = False # whether use ood data
    else:
        use_ood = True
        print(len(subset_indexs), len(unlabeled_trainloader.dataset))
        nsamples = len(subset_indexs)
        unlabeled_trainloader = DataLoader(Subset(unlabeled_trainloader.dataset, subset_indexs),
                                           batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        unlabeled_train_iter = iter(unlabeled_trainloader)
    print(nsamples, len(unlabeled_trainloader.dataset))
    model.train()
    for batch_idx in range(args.val_iteration):
        # all data
        try:
            inputs, domain_labels, indexs = train_iter.next()
        except:
            train_iter = iter(domain_trainloader)
            inputs, domain_labels, indexs = train_iter.next()

        # labeled
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        # sub unlabeled
        if use_ood:
            try:
                (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()

        # measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)
        # Transform label to one-hot
        if use_cuda:
            inputs = inputs.cuda()
            domain_labels = domain_labels.cuda()
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            if use_ood:
                inputs_u = inputs_u.cuda()
                inputs_u2 = inputs_u2.cuda()

        if use_ood:
            # if use ood we need to transform targets_x from label to one hot vector
            targets_x = (torch.zeros(batch_size, args.num_classes).to(inputs.device)).scatter_(1, targets_x.view(-1, 1),
                                                                                               1)
            model.apply(set_bn_eval)
            logits = model(inputs, is_feat=True)[1]
            model.apply(set_bn_train)
            probs = torch.sigmoid(logits).view(-1)
            # binary pseudo label for labeled+unlabeled data
            Ld = F.binary_cross_entropy_with_logits(logits, domain_labels.view(-1, 1))
            # save prediction
            results[indexs.detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()

            with torch.no_grad():
                # compute guessed labels of unlabel samples
                outputs_u = model(inputs_u)
                outputs_u2 = model(inputs_u2)
                p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
                pt = p ** (1 / args.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            # mixup
            all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

            l = np.random.beta(args.alpha, args.alpha)

            l = max(l, 1 - l)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = interleave(mixed_input, batch_size)

            logits = [model(mixed_input[0])]
            for input in mixed_input[1:]:
                logits.append(model(input))

            # put interleaved samples back
            logits = interleave(logits, batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:],
                                  epoch + batch_idx / args.val_iteration)

        else:
            logits = model(inputs_x)
            Lx = F.cross_entropy(logits, targets_x)
            Lu = torch.zeros(1).to(inputs_x.device).mean()
            Ld = torch.zeros(1).to(inputs_x.device).mean()
            w = 0

        loss = Ld + Lx + w * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        losses_d.update(Ld.item(), inputs.size(0))

        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size})| all:{loss:.4f}|x:{lossx:.4f}|u:{lossu:.4f}|d:{lossd:.4f}| W:{w:.4f} '.format(
            batch=batch_idx + 1,
            size=args.val_iteration,
            loss=losses.avg,
            lossx=losses_x.avg,
            lossu=losses_u.avg,
            lossd=losses_d.avg,
            w=ws.avg,
        )
        bar.next()
    bar.finish()
    domain_trainloader.dataset.label_update(results)
    ema_optimizer.step(bn=True)
    return (losses.avg, losses_x.avg, losses_u.avg, nsamples)


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
            bar.suffix = '({batch}/{size}) | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(valloader),
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


def linear_rampup(current, rampup_length=16):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean(torch.mean((probs_u - targets_u) ** 2, dim=1))

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.tmp_model = new_model(args=args)
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.copy_(param.data)

    def step(self, bn=False):
        if bn:
            # copy batchnorm stats to ema model
            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                tmp_param.data.copy_(ema_param.data.detach())

            self.ema_model.load_state_dict(self.model.state_dict())

            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                ema_param.data.copy_(tmp_param.data.detach())
        else:
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.alpha)
                ema_param.data.add_(param.data.detach() * one_minus_alpha)
                # customized weight decay
                param.data.mul_(1 - self.wd)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.track_running_stats = False


def set_bn_train(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.track_running_stats = True


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


if __name__ == '__main__':
    main()
