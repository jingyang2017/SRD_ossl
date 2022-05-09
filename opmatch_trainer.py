import logging
import time
import copy
import os
import numpy as np
import torch
import shutil
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils.opmatch_misc import AverageMeter, ova_loss, ova_ent, exclude_dataset, accuracy
from torchvision import transforms
from utils.randaugment import RandAugmentMC

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


class TransformFixMatch(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.Resize(size_image),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image * 0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.Resize(size_image),
            transforms.RandomHorizontalFlip(), ])

        self.strong = transforms.Compose([
            transforms.Resize(size_image),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(self.weak2(x))
        else:
            return weak, strong


class TransformOpenMatch(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.Resize(size_image),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image * 0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.Resize(size_image),
            transforms.RandomHorizontalFlip(), ])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.weak(x)

        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(self.weak2(x))
        else:
            return weak, strong


def train(args, labeled_dataset, unlabeled_dataset, test_loader, model, optimizer, ema_model, scheduler):
    print('**************** labeled data %d unlabeled data %d' % (len(labeled_dataset), len(unlabeled_dataset)))

    if args.amp:
        from apex import amp

    global best_acc

    test_accs = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_o = AverageMeter()
    losses_oem = AverageMeter()
    losses_socr = AverageMeter()
    losses_fix = AverageMeter()
    mask_probs = AverageMeter()
    end = time.time()

    default_out = "Epoch: {epoch}/{epochs:4}. " \
                  "LR: {lr:.6f}. " \
                  "Lab: {loss_x:.4f}. " \
                  "Open: {loss_o:.4f}"
    output_args = vars(args)
    default_out += " OEM  {loss_oem:.4f}"
    default_out += " SOCR  {loss_socr:.4f}"
    default_out += " Fix  {loss_fix:.4f}"

    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    # unlabeled data for ood detection
    unlabeled_dataset.transform = TransformFixMatch(mean=cifar100_mean, std=cifar100_std)
    unlabeled_trainloader = DataLoader(unlabeled_dataset, sampler=train_sampler(unlabeled_dataset),
                                       batch_size=args.batch_size * args.mu, num_workers=args.num_workers,
                                       drop_last=True)
    unlabeled_iter = iter(unlabeled_trainloader)

    # all unlabeled data for auxilary
    unlabeled_dataset_all = copy.deepcopy(unlabeled_dataset)
    unlabeled_dataset_all.transform = TransformOpenMatch(mean=cifar100_mean, std=cifar100_std)
    unlabeled_trainloader_all = DataLoader(unlabeled_dataset_all, sampler=train_sampler(unlabeled_dataset_all),
                                           batch_size=args.batch_size * args.mu, num_workers=args.num_workers,
                                           drop_last=True)
    unlabeled_all_iter = iter(unlabeled_trainloader_all)

    # labeled data
    labeled_dataset.transform = TransformOpenMatch(mean=cifar100_mean, std=cifar100_std)
    labeled_trainloader = DataLoader(labeled_dataset, sampler=train_sampler(labeled_dataset),
                                     batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    labeled_iter = iter(labeled_trainloader)

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0

    use_fix = True
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        output_args["epoch"] = epoch
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step))

        if epoch >= args.start_fix:
            # TODO check
            sel_inndex = exclude_dataset(args, unlabeled_dataset, ema_model.ema)
            if len(sel_inndex) <= args.batch_size * args.mu:
                # unlabeled_trainloader = DataLoader(unlabeled_dataset, sampler=train_sampler(unlabeled_dataset),
                #                                    batch_size=len(unlabeled_dataset), num_workers=args.num_workers,
                #                                    drop_last=True)
                use_fix = False
                # assert NotImplementedError
            else:
                unlabeled_dataset = Subset(unlabeled_dataset, sel_inndex)
                unlabeled_trainloader = DataLoader(unlabeled_dataset, sampler=train_sampler(unlabeled_dataset),
                                                   batch_size=args.batch_size * args.mu, num_workers=args.num_workers,
                                                   drop_last=True)
                unlabeled_iter = iter(unlabeled_trainloader)

        for batch_idx in range(args.eval_step):
            ## labeled data
            try:
                (_, inputs_x_s, inputs_x), targets_x, _ = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)

                labeled_iter = iter(labeled_trainloader)
                (_, inputs_x_s, inputs_x), targets_x, _ = labeled_iter.next()

            ## unlabeled data
            if use_fix:
                try:
                    (inputs_u_w, inputs_u_s, _), _ = unlabeled_iter.next()
                except:
                    if args.world_size > 1:
                        unlabeled_epoch += 1
                        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                    unlabeled_iter = iter(unlabeled_trainloader)
                    (inputs_u_w, inputs_u_s, _), _ = unlabeled_iter.next()

            ## all unlabeled data
            try:
                (inputs_all_w, inputs_all_s, _), _ = unlabeled_all_iter.next()
            except:
                unlabeled_all_iter = iter(unlabeled_trainloader_all)
                (inputs_all_w, inputs_all_s, _), _ = unlabeled_all_iter.next()

            data_time.update(time.time() - end)

            b_size = inputs_x.shape[0]

            # binary classification on all data
            inputs_all = torch.cat([inputs_all_w, inputs_all_s], 0)
            inputs = torch.cat([inputs_x, inputs_x_s, inputs_all], 0).to(args.device)
            targets_x = targets_x.to(args.device)

            ## Feed data
            logits, logits_open = model(inputs, is_feat=True)
            logits_open_u1, logits_open_u2 = logits_open[2 * b_size:].chunk(2)

            # for labeled data
            ## Loss for labeled samples
            Lx = F.cross_entropy(logits[:2 * b_size], targets_x.repeat(2), reduction='mean')
            # build the binary classification
            Lo = ova_loss(logits_open[:2 * b_size], targets_x.repeat(2))

            # for unlabeled data
            ## Open-set entropy minimization
            L_oem = ova_ent(logits_open_u1) / 2.
            L_oem += ova_ent(logits_open_u2) / 2.

            ## Soft consistenty regularization
            logits_open_u1 = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
            logits_open_u2 = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
            logits_open_u1 = F.softmax(logits_open_u1, 1)
            logits_open_u2 = F.softmax(logits_open_u2, 1)
            L_socr = torch.mean(torch.sum(torch.sum(torch.abs(logits_open_u1 - logits_open_u2) ** 2, 1), 1))

            if epoch >= args.start_fix and use_fix:
                ## pseduo label only on part data
                inputs_ws = torch.cat([inputs_u_w, inputs_u_s], 0).to(args.device)
                logits, _ = model(inputs_ws, is_feat=True)
                logits_u_w, logits_u_s = logits.chunk(2)
                ## weak guide strong
                pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()
                L_fix = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
                mask_probs.update(mask.mean().item())
            else:
                L_fix = torch.zeros(1).to(args.device).mean()

            loss = Lx + Lo + args.lambda_oem * L_oem + args.lambda_socr * L_socr + args.lambda_fix * L_fix

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_o.update(Lo.item())
            losses_oem.update(L_oem.item())
            losses_socr.update(L_socr.item())
            losses_fix.update(L_fix.item())

            output_args["batch"] = batch_idx
            output_args["loss_x"] = losses_x.avg
            output_args["loss_o"] = losses_o.avg
            output_args["loss_oem"] = losses_oem.avg
            output_args["loss_socr"] = losses_socr.avg
            output_args["loss_fix"] = losses_fix.avg
            output_args["used_un"] = len(unlabeled_dataset)
            output_args["lr"] = [group["lr"] for group in optimizer.param_groups][0]

            optimizer.zero_grad()
            if args.amp:
                # half precision
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            if args.opt != 'adam':
                scheduler.step()

            if args.use_ema:
                ema_model.update(model)

            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description(default_out.format(**output_args))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)
            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_o', losses_o.avg, epoch)
            args.writer.add_scalar('train/4.train_loss_oem', losses_oem.avg, epoch)
            args.writer.add_scalar('train/5.train_loss_socr', losses_socr.avg, epoch)
            args.writer.add_scalar('train/5.train_loss_fix', losses_fix.avg, epoch)
            args.writer.add_scalar('train/6.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('train/7.used_un', len(unlabeled_dataset), epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema

            save_checkpoint({
                'epoch': epoch + 1,

                'model_state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc, }, is_best, args.out)
            test_accs.append(test_acc)
            print('Best top-1 acc(test): {:.2f}'.format(best_acc))
            print('Mean top-1 acc(test): {:.2f}'.format(np.mean(test_accs[-20:])))
            print('curr top-1 acc(test): {:.2f}'.format(test_acc))

    if args.local_rank in [-1, 0]:
        args.writer.close()
    with open(args.out + '/res_%s.txt'%str(time.ctime()), 'w') as f:
        f.write('%.4f' % best_acc)
        f.write('\n')
        f.write('%.4f' % np.mean(test_accs[-20:]))




def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    model.eval()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

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
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
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
    model.train()
    return losses.avg, top1.avg
