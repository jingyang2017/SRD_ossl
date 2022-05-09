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
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage.filters import threshold_otsu
from helper.util import AverageMeter, accuracy
from dataset.cifar100 import get_cifar100_dataloaders
from models import model_dict
from torchvision import transforms
from utils.randaugment import RandAugmentMC

logger = logging.getLogger(__name__)
best_acc = 0

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,padding=int(32*0.125),padding_mode='reflect')])
        self.strong = transforms.Compose([transforms.Resize(32),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(size=32,padding=int(32*0.125),
                                          padding_mode='reflect'),
                                          RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


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
    parser = argparse.ArgumentParser(description='PyTorch T2T Stage2 Training')
    parser.add_argument('--gpu-id', default='0', type=int,help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--arch', default='wrn_40_1', type=str,choices=['wrn_40_1','resnet8x4','ShuffleV1'],help='dataset name')
    parser.add_argument('--mu', default=5, type=int, help='coefficient of unlabeled batch size')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--out', default='./Results/T2T/Stage2/', help='directory to output the result')
    parser.add_argument('--filter-every-epoch', type=int, default=20, help='every K epoch to filter in distribution unlabeled data')
    parser.add_argument('--lmx', default=1, type=float)
    parser.add_argument('--lrot', default=1, type=float)
    parser.add_argument('--lun', default=1, type=float)

    parser.add_argument('--total-steps', default=200000, type=int, help='number of total steps to run')
    parser.add_argument('--eval-step', default=1000, type=int, help='number of eval steps to run')
    parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
    parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar100'],help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000, help='number of labeled data')
    parser.add_argument('--num-val', type=int, default=5000, help='number of validation data')
    parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,help='use nesterov momentum')
    parser.add_argument('--lambda-u', default=1, type=float,help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,help='pseudo label threshold')
    parser.add_argument('--seed', default=5, type=int,help="random seed")
    parser.add_argument("--local_rank", type=int, default=-1,help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")
    parser.add_argument('--ood-dataset', type=str, default='tin',
                        choices=['tin','places'],
                        help='choose one dataset as ood data source')
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args)

    args.out = args.out+'/'+str(args.arch)+'_'+str(args.ood_dataset)+'_lmx_'+str(args.lmx)+'_lrot_'+str(args.lrot)+'_lun_'+str(args.lun)
    os.makedirs(args.out, exist_ok=True)
    args.writer = SummaryWriter(args.out)

    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

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
        logger.info("Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))
        rotnet_head = torch.nn.Linear(feats[-1].size(1), 4)
        from utils.t2t import CrossModalMatchingHead
        cmm_head = CrossModalMatchingHead(args.num_classes, feats[-1].size(1))
        model.train()
        return model, rotnet_head, cmm_head

    ## use labeled transformer
    train_loader, utrain_loader, test_loader, n_data = get_cifar100_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers, is_instance=True, ood = args.ood_dataset,is_sample=False, ratio=1)
    labeled_trainloader = train_loader

    ## fixmatch transformer
    unlabeled_dataset = deepcopy(utrain_loader.dataset)
    unlabeled_trainloader = DataLoader(unlabeled_dataset, batch_size=args.batch_size*args.mu, shuffle=True, num_workers=args.num_workers, drop_last=True)
    unlabeled_trainloader.dataset.transform = TransformFixMatch(mean=(0.5071, 0.4867, 0.4408),std=(0.2675, 0.2565, 0.2761))


    ## use labeled transformer
    udst_rotnet = deepcopy(utrain_loader.dataset)
    udst_rotnet.transform = labeled_trainloader.dataset.transform
    udst_rotnet_loader = DataLoader(udst_rotnet,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,drop_last=True)


    model, rotnet_head, cmm_head = create_model(args)
    model, rotnet_head, cmm_head = model.to(args.device), rotnet_head.to(args.device), cmm_head.to(args.device)

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(args.resume), "Error: no checkpoint directory found!"
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        rotnet_head.load_state_dict(checkpoint['rotnet_state_dict'])
        cmm_head.load_state_dict(checkpoint['cmm_state_dict'])

    test_loss, test_acc = test(args, test_loader, model, 0)
    print('before 2nd stage training', test_acc)

    udst_eval = deepcopy(utrain_loader.dataset)
    udst_eval.transform = test_loader.dataset.transform
    udst_eval_loader = DataLoader(udst_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    unlabeled_dataset = deepcopy(utrain_loader.dataset)
    train_stage2(args, labeled_trainloader, unlabeled_trainloader, test_loader,
                 udst_rotnet_loader, udst_eval_loader, unlabeled_dataset,
                 model, rotnet_head, cmm_head)


def train_stage2(args, labeled_trainloader, unlabeled_trainloader, test_loader,
                 udst_rotnet_loader, udst_eval_loader, unlabeled_dataset,
                 model, rotnet_head, cmm_head):
    """
    In this stage, we train the model with five losses:
    1. Lx:  cross-entropy loss for labeled data (ref to L_ce)
    2. Lmx: cross-modal matching loss for labeled data (ref to L_cm^l)
    3. Lr:  rotation recognition loss for all training data (ref to L_rot)
    4. Lmu: cross-modal matching loss for unlabeled data (ref to L_cm^u)
    5. Lu:  consistency constraint loss for filtered unlabeled data (ref to L_cc)
    """

    global best_acc
    test_accs = []
    end = time.time()

    grouped_parameters = [
        {'params': model.parameters()},
        {'params': rotnet_head.parameters()},
        {'params': cmm_head.parameters()}
    ]

    # args.epoch = 240
    # optimizer = optim.SGD(grouped_parameters, lr=0.03, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,180,210], gamma=0.2)
    # flag = True

    optimizer = optim.SGD(grouped_parameters, lr=0.03, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)
    flag = False

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    rotnet_iter = iter(udst_rotnet_loader)

    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        losses_mx = AverageMeter()
        losses_mu = AverageMeter()
        losses_r = AverageMeter()

        # clean unlabeled data periodically for consistency constraint loss
        if epoch % args.filter_every_epoch == 0:
            # from whole unlabeled dataset to select samples
            in_dist_idxs = filter_ood(args, udst_eval_loader, model, cmm_head)
            in_dist_unlabeled_dataset = Subset(unlabeled_dataset, in_dist_idxs)
            unlabeled_trainloader = DataLoader(in_dist_unlabeled_dataset, batch_size=args.batch_size * args.mu,
                                               shuffle=True, num_workers=args.num_workers, drop_last=True)
            logging.info('use %d/%d' % (len(in_dist_idxs), len(unlabeled_dataset)))

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
                (inputs_u_w, inputs_u_s), index_u = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), index_u = unlabeled_iter.next()

            try:
                inputs_r, index_u = rotnet_iter.next()
            except:
                rotnet_iter = iter(udst_rotnet_loader)
                inputs_r, index_u = rotnet_iter.next()

            # rotate unlabeled data with 0, 90, 180, 270 degrees
            inputs_r = torch.cat([torch.rot90(inputs_r, i, [2, 3]) for i in range(4)], dim=0)
            targets_r = torch.cat([torch.empty(index_u.size(0)).fill_(i).long() for i in range(4)], dim=0).to(
                args.device)

            data_time.update(time.time() - end)

            batch_size = inputs_x.shape[0]

            # TODO: check input
            inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).to(args.device)
            feats_all, logits = model(inputs, is_feat=True)
            feats = feats_all[-1].view(feats_all[-1].size(0), -1)

            logits_x = logits[:batch_size]  # labeled
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)  # unlabeled
            feats_x = feats[:batch_size]

            # del logits
            # Labeled Data: cross Entropy Loss for
            targets_x = targets_x.to(args.device)
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            # Unlabeled Data:Consistency Constraint Loss
            # hyper parameters for UDA
            T = 0.4
            p_cutoff = 0.8
            logits_tgt = logits_u_w / T
            probs_u_w = torch.softmax(logits_u_w, dim=1)
            loss_mask = probs_u_w.max(-1)[0].ge(p_cutoff)

            if loss_mask.sum() == 0:
                Lu = torch.zeros(1, dtype=torch.float).to(args.device)
            else:
                Lu = F.kl_div(torch.log_softmax(logits_u_s[loss_mask], -1),
                              torch.softmax(logits_tgt[loss_mask].detach().data, -1), reduction='batchmean')

            # Labeled: Cross Modal Matching Training:
            # 1 positve pair + 2 negative pairs for each labeled data
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

            # Unlabeled: Cross Entropy Loss for Rotation Recognition
            inputs_r = inputs_r.to(args.device)
            feats_all, logits_r = model(inputs_r, is_feat=True)
            feats_r = feats_all[-1].view(feats_all[-1].size(0), -1)
            Lr = F.cross_entropy(rotnet_head(feats_r), targets_r, reduction='mean')

            # unlabeled data: Cross Modal Matching Training:
            # Use Entropy Minimization Loss for all unlabeled data (including OOD data)
            # So we use data from RotNet Dataloder which has all training data
            batch_size = inputs_r.size(0) // 4
            y_onehot = torch.zeros((2 * batch_size, args.num_classes)).float().to(args.device)
            y = torch.zeros(2 * batch_size).long().to(args.device)
            # select the most confident class and randomly choose one from rest classes
            with torch.no_grad():
                prob_sorted_index = torch.argsort(logits_r[:batch_size], descending=True)
                y[:batch_size] = prob_sorted_index[:, 0]
                for i in range(batch_size):
                    y[batch_size + i] = int(np.random.choice(prob_sorted_index[i, 1:].cpu(), 1))
            y_onehot.scatter_(1, y.view(-1, 1), 1)
            matching_score_u = cmm_head(feats_r[:batch_size].repeat(2, 1), y_onehot)
            Lmu = F.binary_cross_entropy_with_logits(matching_score_u, torch.sigmoid(matching_score_u))

            # we use linear ramp up weighting here for stabilizing training process
            alpha = linear_rampup(epoch * args.eval_step + batch_idx, 40 * args.eval_step)

            # TODO：all losses
            loss = Lx + Lmx * args.lmx + Lr * args.lrot + alpha * (Lmu + Lu) * args.lun

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not flag:
                scheduler.step()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_mx.update(Lmx.item())
            losses_mu.update(Lmx.item())
            losses_r.update(Lr.item())
            losses_u.update(Lu.item())

            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. "
                    "Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_mx: {loss_mx:.4f}. Loss_r: {loss_r:.4f}. Loss_mu: {loss_mu:.4f}. "
                    "Loss_u: {loss_u:.4f}. alpha: {alpha:.4f}".format(
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
                        loss_r=losses_r.avg,
                        loss_mu=losses_mu.avg,
                        loss_u=losses_u.avg,
                        alpha=alpha
                    ))
                p_bar.update()

        if flag:
            scheduler.step()

        if not args.no_progress:
            p_bar.close()

        test_loss, test_acc = test(args, test_loader, model, epoch)

        args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
        args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
        args.writer.add_scalar('train/3.train_loss_mx', losses_mx.avg, epoch)
        args.writer.add_scalar('train/4.train_loss_r', losses_r.avg, epoch)
        args.writer.add_scalar('train/5.train_loss_mu', losses_mu.avg, epoch)
        args.writer.add_scalar('train/6.train_loss_u', losses_u.avg, epoch)
        args.writer.add_scalar('train/7.used_un', len(unlabeled_trainloader.dataset), epoch)
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

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
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
    model.train()

    return losses.avg, top1.avg


def filter_ood(args, loader, model, cmm_head):
    # switch to evaluate mode
    model.eval()
    cmm_head.eval()

    matching_scores = []
    idxs = []
    in_dist_idxs = []

    with torch.no_grad():
        for batch_idx, (input,indexs) in enumerate(loader):
            input = input.to(args.device)
            feats_all, logits = model(input, is_feat=True)
            feats = feats_all[-1].view(feats_all[-1].size(0),-1)

            y_onehot = torch.zeros((input.size(0), args.num_classes)).float().to(args.device)
            y_pred = torch.argmax(logits, dim=1, keepdim=True)
            y_onehot.scatter_(1, y_pred, 1)

            matching_score = torch.sigmoid(cmm_head(feats, y_onehot))

            for i in range(len(input)):
                matching_scores.append(matching_score[i].cpu().item())
                idxs.append(indexs[i].item())

    # use otsu threshold to adaptively compute threshold
    matching_scores = np.array(matching_scores)
    thresh = threshold_otsu(matching_scores)
    for i, s in enumerate(matching_scores):
        if s > thresh:
            in_dist_idxs.append(idxs[i])
    logger.info('OOD Filtering threshold: %.3f' % thresh)

    model.train()
    cmm_head.train()
    return in_dist_idxs
# def train_stage2_debug(args, labeled_trainloader, unlabeled_trainloader, test_loader,
#                  udst_rotnet_loader, udst_eval_loader, unlabeled_dataset,
#                  model, rotnet_head, cmm_head):
#     global best_acc
#     test_accs = []
#     end = time.time()
#
#     grouped_parameters = [
#         {'params': model.parameters()},
#         {'params': rotnet_head.parameters()},
#         {'params': cmm_head.parameters()}
#     ]
#
#     optimizer = optim.SGD(grouped_parameters, lr=0.03, momentum=0.9, weight_decay=5e-4, nesterov=True)
#     # scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)
#     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,180,210], gamma=0.2)
#
#     labeled_iter = iter(labeled_trainloader)
#     unlabeled_iter = iter(unlabeled_trainloader)
#     rotnet_iter = iter(udst_rotnet_loader)
#     args.eval_step = len(labeled_trainloader)
#
#     for epoch in range(args.start_epoch, args.epochs):
#         batch_time = AverageMeter()
#         data_time = AverageMeter()
#         losses = AverageMeter()
#
#         model.train()
#         rotnet_head.train()
#         cmm_head.train()
#
#         if not args.no_progress:
#             p_bar = tqdm(range(args.eval_step))
#
#         for batch_idx in range(args.eval_step):
#             try:
#                 inputs_x, targets_x, index_x = labeled_iter.next()
#             except:
#                 labeled_iter = iter(labeled_trainloader)
#                 inputs_x, targets_x, index_x = labeled_iter.next()
#
#             data_time.update(time.time() - end)
#             batch_size = inputs_x.shape[0]
#
#             inputs = inputs_x.to(args.device)
#             feats_all, logits = model(inputs, is_feat=True)
#             logits_x = logits[:batch_size]  # labeled
#             targets_x = targets_x.to(args.device)
#             Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
#             loss = Lx
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             losses.update(loss.item())
#             batch_time.update(time.time() - end)
#             end = time.time()
#             if not args.no_progress:
#                 p_bar.set_description(
#                     "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. "
#                     "Loss: {loss:.4f}".format(
#                         epoch=epoch + 1,
#                         epochs=args.epochs,
#                         batch=batch_idx + 1,
#                         iter=args.eval_step,
#                         lr=scheduler.get_last_lr()[0],
#                         data=data_time.avg,
#                         bt=batch_time.avg,
#                         loss=losses.avg,
#                     ))
#                 p_bar.update()
#
#         if not args.no_progress:
#             p_bar.close()
#
#         scheduler.step()
#         test_model = model
#         test_loss, test_acc = test(args, test_loader, test_model, epoch)
#         args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
#         args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
#
#         is_best = test_acc > best_acc
#         best_acc = max(test_acc, best_acc)
#
#         model_to_save = model.module if hasattr(model, "module") else model
#         save_checkpoint({
#             'epoch': epoch + 1,
#             'model_state_dict': model_to_save.state_dict(),
#             'rotnet_state_dict': rotnet_head.state_dict(),
#             'cmm_state_dict': cmm_head.state_dict(),
#             'acc': test_acc,
#             'best_acc': best_acc,
#             'optimizer': optimizer.state_dict(),
#             'scheduler': scheduler.state_dict(),
#         }, is_best, args.out)
#
#         test_accs.append(test_acc)
#         logger.info('Best top-1 acc(test): {:.2f}'.format(best_acc))
#         logger.info('Mean top-1 acc(test): {:.2f}'.format(np.mean(test_accs[-20:])))
#         logger.info('curr top-1 acc(test): {:.2f}'.format(test_acc))







if __name__ == '__main__':
    main()
