import logging
import os
import torch
import time
import random
from torch.utils.tensorboard import SummaryWriter
from opmatch_trainer import train
from train_t2t_stage2 import set_seed, get_cosine_schedule_with_warmup
logger = logging.getLogger(__name__)
from dataset.cifar100 import get_cifar100_dataloaders
from models import model_dict
import argparse
import torch.optim as optim
__all__ = ['set_parser']
from copy import deepcopy

class ModelEMA(object):
    def __init__(self, args, model, decay):
        self.ema = deepcopy(model)
        self.ema.to(args.device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])

class new_model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        '''
        arg.arch -- model_name
        args.num_classes--class number
        '''
        self.model = model_dict[args.arch](num_classes=args.num_classes)
        head = torch.nn.Linear(self.model.feat_dim, 2 * args.num_classes, bias=False)
        # initialization is from OP_match github
        w = head.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        head.weight.data = w.div(norm.expand_as(w))
        self.head = head

    def forward(self, x, is_feat=False):
        feats, logits = self.model(x, True)

        if is_feat:
            feat = feats[-1].view(feats[-1].size(0), -1)
            return logits, self.head(feat)      # return logits, logits_open
        else:
            return logits


def set_parser():
    parser = argparse.ArgumentParser(description='PyTorch OpenMatch Training')
    parser.add_argument('--arch', default='wrn_40_1', type=str,choices=['wrn_40_1','resnet8x4','ShuffleV1'],help='dataset name')
    parser.add_argument('--ood', type=str, default='tin',choices=['tin','places','None'])
    parser.add_argument('--num_classes', type=int, default=100,help='for cifar100')
    parser.add_argument('--dataset', default='cifar100', help='dataset name')
    parser.add_argument('--lambda_oem', default=0.1, type=float, help='coefficient of OEM loss')
    parser.add_argument('--lambda_socr', default=1, type=float, help='coefficient of SOCR loss, 0.5 for CIFAR10, ImageNet, ''1.0 for CIFAR100')
    parser.add_argument('--lambda_fix', default=1, type=float, help='coefficient for cross entropy loss at higest response')
    parser.add_argument('--mu', default=2, type=int, help='coefficient of unlabeled batch size')
    parser.add_argument('--opt', default='sgd', type=str,choices=['sgd', 'adam'],help='optimize name')

    #train
    parser.add_argument('--gpu-id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    # parser.set_defaults(amp=True)
    parser.add_argument("--opt_level", type=str, default="O2",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")
    parser.add_argument('--eval_only', type=int, default=0, help='1 if evaluation mode ')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    ## Hyper-parameters

    ## HP unique to OpenMatch (Some are changed from FixMatch)
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--start_fix', default=10, type=int,help='epoch to start fixmatch training')
    parser.add_argument('--total-steps', default=2 ** 19, type=int,help='number of total steps to run')
    parser.add_argument('--epochs', default=512, type=int,help='number of epochs to run')
    parser.add_argument('--threshold', default=0.0, type=float,help='pseudo label threshold')
    parser.add_argument('--eval-step', default=1024, type=int,help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True, help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay rate')
    parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
    args = parser.parse_args()
    return args


def main():
    args = set_parser()
    global best_acc

    if args.local_rank == -1:
        # simple
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        # distributed local_rank is 0
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    if args.seed is not None:
        set_seed(args)

    train_loader, utrain_loader, test_loader, n_data = get_cifar100_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers, is_instance=True, is_sample=False,ood=args.ood)

    args.out = './Results/OP_MATCH/'+str(args.arch)+'_ood_'+str(args.ood)+'_amp_'+str(args.amp)

    os.makedirs(args.out, exist_ok=True)
    print(args.out)
    print(args.local_rank)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.writer = SummaryWriter(args.out)

    model = new_model(args)
    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    if args.opt == 'sgd':
        optimizer = optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9, nesterov=args.nesterov)
    elif args.opt == 'adam':
        optimizer = optim.Adam(grouped_parameters, lr=2e-3)
    else:
        raise NotImplementedError

    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)
    logger.info("Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    if args.use_ema:
        ema_model = ModelEMA(args, model, args.ema_decay)
    else:
        ema_model = None

    args.start_epoch = 0

    if args.amp:
        # half precision
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        #distributed
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")
    labeled_dataset = deepcopy(train_loader.dataset)
    unlabeled_dataset = deepcopy(utrain_loader.dataset)

    train(args, labeled_dataset, unlabeled_dataset, test_loader, model, optimizer, ema_model, scheduler)



if __name__ == '__main__':
    main()
