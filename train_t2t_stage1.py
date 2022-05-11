import argparse
import logging
import math
import os
import random
import shutil
import time
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from helper.util import AverageMeter, accuracy
from dataset.cifar100 import get_cifar100_dataloaders
from models import model_dict

logger = logging.getLogger(__name__)
best_acc = 0

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def main():
    parser = argparse.ArgumentParser(description='PyTorch T2T Stage1 Training')
    parser.add_argument('--arch', default='wrn_40_1', type=str,choices=['wrn_40_1','resnet8x4','ShuffleV1'], help='dataset name')
    parser.add_argument('--ood-dataset', type=str, default='tin',
                        choices=['tin','places'],
                        help='choose one dataset as ood data source')

    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar100', type=str,
                        choices=['cifar100'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=50000, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=500, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='./Results/T2T/Stage1/',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=5, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")


    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args)

    args.out = args.out+'/'+str(args.arch)+'_'+str(args.ood_dataset)
    os.makedirs(args.out, exist_ok=True)
    args.writer = SummaryWriter(args.out)

    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    if args.dataset == 'cifar100':
        args.num_classes = 100

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    args.epochs = math.ceil(args.total_steps / args.eval_step)

    def create_model(args):
        model = model_dict[args.arch](num_classes=args.num_classes)
        data = torch.randn(2, 3, 32, 32)
        model.eval()
        feats, _ = model(data, is_feat=True)
        logger.info("Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1e6))
        rotnet_head = torch.nn.Linear(feats[-1].size(1), 4)
        from utils.t2t import CrossModalMatchingHead
        cmm_head = CrossModalMatchingHead(args.num_classes, feats[-1].size(1))
        return model, rotnet_head, cmm_head

    train_loader, utrain_loader, test_loader, n_data = get_cifar100_dataloaders(batch_size=args.batch_size,
                                                                                num_workers=args.num_workers,
                                                                                is_instance=True,
                                                                                is_sample=False,
                                                                                ood = args.ood_dataset)

    # unlabeled_dataset contains all training data
    # however, it use randaugment as data transformation
    # so we copy it and use simple data transformation for rotnet dataloader

    udst_rotnet = deepcopy(utrain_loader.dataset)
    udst_rotnet.transform = utrain_loader.dataset.transform
    udst_rotnet_loader = DataLoader(
        udst_rotnet,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)

    labeled_trainloader = train_loader

    model, rotnet_head, cmm_head = create_model(args)

    model, rotnet_head, cmm_head = model.to(args.device), rotnet_head.to(args.device), cmm_head.to(args.device)

    train_stage1(args, labeled_trainloader, udst_rotnet_loader, test_loader,
                 model, rotnet_head, cmm_head)


def train_stage1(args, labeled_trainloader, unlabeled_trainloader, test_loader,
                 model, rotnet_head, cmm_head):
    """
    In this stage, we train the model with three losses:
    1. Lx:  cross-entropy loss for labeled data (ref to L_ce in origin paper)
    2. Lmx: cross-modal matching loss for labeled data (ref to L_cm^l in origin paper)
    3. Lr:  rotation recognition loss for all training data (ref to L_rot in origin paper)
    """

    global best_acc
    test_accs = []
    end = time.time()

    grouped_parameters = [
        {'params': model.parameters()},
        {'params': rotnet_head.parameters()},
        {'params': cmm_head.parameters()}
    ]

    optimizer = optim.SGD(grouped_parameters, lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80, 100], gamma=0.2)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_mx = AverageMeter()
        losses_r = AverageMeter()

        model.train()
        rotnet_head.train()
        cmm_head.train()

        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step))

        for batch_idx in range(args.eval_step):

            try:
                inputs_x, targets_x, index_x = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x, index_x = labeled_iter.next()

            try:
                (inputs_u_w, _), index_u = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, _), index_u = unlabeled_iter.next()

            # rotate unlabeled data with 0, 90, 180, 270 degrees
            inputs_r = torch.cat(
                [torch.rot90(inputs_u_w, i, [2, 3]) for i in range(4)], dim=0)
            targets_r = torch.cat(
                [torch.empty(inputs_u_w.size(0)).fill_(i).long() for i in range(4)], dim=0).to(args.device)

            data_time.update(time.time() - end)

            batch_size = inputs_x.shape[0]
            inputs_x, targets_x = inputs_x.to(args.device), targets_x.to(args.device)

            feats, logits_x = model(inputs_x, is_feat=True)
            feats_x = feats[-1].view(feats[-1].size(0),-1)

            # Cross Entropy Loss for Labeled Data
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            # Cross Modal Matching Training: 1 positve pair + 2 negative pair for each labeled data
            # [--pos--, --hard_neg--, --easy_neg--]
            matching_gt = torch.zeros(3 * batch_size).to(args.device)
            matching_gt[:batch_size] = 1
            y_onehot = torch.zeros((3 * batch_size, args.num_classes)).float().to(args.device)
            y = torch.zeros(3 * batch_size).long().to(args.device)
            y[:batch_size] = targets_x
            with torch.no_grad():
                prob_sorted_index = torch.argsort(logits_x, descending=True)
                for i in range(batch_size):
                    if prob_sorted_index[i, 0] == targets_x[i]:
                        y[1 * batch_size + i] = prob_sorted_index[i, 1]
                        y[2 * batch_size + i] = int(np.random.choice(prob_sorted_index[i, 2:].cpu(), 1))
                    else:
                        y[1 * batch_size + i] = prob_sorted_index[i, 0]
                        choice = int(np.random.choice(prob_sorted_index[i, 1:].cpu(), 1))
                        while choice == targets_x[i]:
                            choice = int(np.random.choice(prob_sorted_index[i, 1:].cpu(), 1))
                        y[2 * batch_size + i] = choice
            y_onehot.scatter_(1, y.view(-1, 1), 1)
            matching_score_x = cmm_head(feats_x.repeat(3, 1), y_onehot)
            Lmx = F.binary_cross_entropy_with_logits(matching_score_x.view(-1), matching_gt)

            # Cross Entropy Loss for Rotation Recognition
            inputs_r = inputs_r.to(args.device)
            feats, _ = model(inputs_r, is_feat=True)
            feats_r = feats[-1].view(feats[-1].size(0), -1)
            Lr = F.cross_entropy(rotnet_head(feats_r), targets_r, reduction='mean')

            loss = Lx + Lmx + Lr

            optimizer.zero_grad()
            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_mx.update(Lmx.item())
            losses_r.update(Lr.item())

            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. "
                                      "Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_mx: {loss_mx:.4f}. Loss_r: {loss_r:.4f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_mx=losses_mx.avg,
                    loss_r=losses_r.avg
                ))
                p_bar.update()

        scheduler.step(epoch)

        if not args.no_progress:
            p_bar.close()

        test_model = model

        test_loss, test_acc = test(args, test_loader, test_model, epoch)

        args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
        args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
        args.writer.add_scalar('train/3.train_loss_mx', losses_mx.avg, epoch)
        args.writer.add_scalar('train/4.train_loss_r', losses_r.avg, epoch)
        args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
        args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        model_to_save = model.module if hasattr(model, "module") else model
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model_to_save.state_dict(),
            'rotnet_state_dict': rotnet_head.state_dict(),
            'cmm_state_dict': cmm_head.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, args.out)

        test_accs.append(test_acc)
        logger.info('Best top-1 acc(test): {:.2f}'.format(best_acc))
        logger.info('Mean top-1 acc(test): {:.2f}'.format(np.mean(test_accs[-20:])))
        logger.info('curr top-1 acc(test): {:.2f}'.format(test_acc))


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
