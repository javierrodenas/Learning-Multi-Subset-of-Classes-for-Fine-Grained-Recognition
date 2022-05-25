# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------
from torchvision import transforms
import argparse
import math
import time
import yaml
import kornia as K
import os
import logging
import timm
from torch.cuda.amp import GradScaler
from collections import OrderedDict, defaultdict
from contextlib import suppress
from datetime import datetime
import wandb
from tqdm import tqdm
from timm.models.helpers import load_state_dict
import pandas as pd
import models
import numpy as np

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torchvision.transforms.functional as fn
from timm.data import ImageDataset, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import load_checkpoint, create_model, resume_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
from checkpoint_saver import CheckpointSaver
from labeled_memcached_dataset import McDataset
from dataset.datasets import Food1k, FoodX251Dataset, Food101
from dataset.datasets import get_next_batch
from utils.extra_utils import create_loader
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import os

# from timm.data import create_loader


torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

import warnings

warnings.filterwarnings('ignore')

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='CSWin Training and Evaluating')

# Dataset / Model parameters
parser.add_argument('--data', default='/mnt/blob/testset/ImageNet', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset_type', default=['iccv', "logmeal_drinks", "logmeal_types", "logmeal_ingredients",
                                               "logmeal_dishes", "logmeal_hydra", "iccv_hydra"], type=str,
                    help='dataset type')
parser.add_argument('--data_reduction_factor', default=1, type=int)
parser.add_argument("--batch_multiply", default=8, type=int)
parser.add_argument('--experiment_name', default=None, type=str,
                    help='Experiment name to be used for wandb logging')
parser.add_argument("--wandb_resume_id", default=None, type=str)
parser.add_argument('--model', default='CSWin_64_12211_tiny_224', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--eval_checkpoint', default='', type=str, metavar='PATH',
                    help='path to eval checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=251, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--gradient_accumulation', default=1, type=int)
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=224, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.05,
                    help='weight decay (default: 0.005 for adamw)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=3e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.5, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=["augmix", "augmix-m2-w4-d2", 'rand-m9-mstd0.5-inc1'][0], metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug_splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.0)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=True,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.99992,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=True,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels_last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='/media/HDD_4TB/models_checkpoints/CSWIN_ICCV', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument('--TTA_enabled', action='store_true', default=False,
                    help='Apply TTA')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')

parser.add_argument('--use-chk', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')

parser.add_argument('--cluster_file', default='251_clusters_CSWin.csv', type=str,
                    help='path to clusters csv')
parser.add_argument('--bs_reg', action='store_true', default=False,
                    help='Use different batch size for heads')


# MultiSubset
parser.add_argument('--step', default=0, type=int)

has_apex = False
has_native_amp = True
hydra_enabled = True


def make_weights_for_balanced_classes(dataset, nclasses=251, num_cluster=0):

    print(dataset)

    if num_cluster:
        images = [mapped_targets[int(label)][num_cluster - 1] for label in dataset.labels]
        nclasses = np.count_nonzero(mapped_targets[:, num_cluster - 1]) + 1
    else:
        images = [int(label) for label in dataset.labels]

    count = [0] * nclasses
    for item in images:
        count[item] += 1
    weight_per_class = [0.] * nclasses

    for i in range(nclasses):
        weight_per_class[i] = 1. / float(count[i])

    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val]

    median = sorted(count)[nclasses // 2]  # Uses +1 if even

    return weight, median


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def processTTA(img):
    img = transforms.ToPILImage()(img.squeeze_(0))

    width, height = img.size

    start_x = int(width/8)
    start_y = int(height/8)

    crop1 = img.crop([0, 0, width - start_x, height-start_y])
    crop2 = img.crop([start_x, 0, width, height-start_y])
    crop3 = img.crop([start_x, start_y, width, height])
    crop4 = img.crop([0, start_y, width-start_x, height])

    crop1 = fn.resize(crop1, size=[width])
    crop2 = fn.resize(crop2, size=[width])
    crop3 = fn.resize(crop3, size=[width])
    crop4 = fn.resize(crop4, size=[width])

    img = transforms.ToTensor()(img).unsqueeze_(0)
    crop1 = transforms.ToTensor()(crop1).unsqueeze_(0)
    crop2 = transforms.ToTensor()(crop2).unsqueeze_(0)
    crop3 = transforms.ToTensor()(crop3).unsqueeze_(0)
    crop4 = transforms.ToTensor()(crop4).unsqueeze_(0)


    return [img, crop1, crop2, crop3, crop4]

def main():
    global lookup_table_df
    global mapped_targets
    global real_targets
    global num_heads
    args, args_text = _parse_args()

    cluster_df = pd.read_csv(args.cluster_file, dtype='Int64')
    num_heads = len(cluster_df.columns)
    real_targets = cluster_df.fillna(0).values.astype(int)  # ints [class,head] real index
    real_targets = np.vstack((np.zeros(real_targets.shape[1]), real_targets)).astype(int)
    lookup_table_df = pd.DataFrame(0, index=range(0, args.num_classes), columns=range(0, num_heads))
    for head in range(num_heads):
        # if not head in lookup_table_dict:
        labels_cluster = cluster_df.iloc[:, head].dropna().tolist()
        for idx, cls in enumerate(labels_cluster, start=1):
            lookup_table_df.loc[cls, head] = idx

    mapped_targets = lookup_table_df.values  # ints [class,head] mapped_index


    setup_default_logging()
    args, args_text = _parse_args()


    if args.experiment_name is not None:
        wandb.init(project="Learning_MultiSubset", name=args.experiment_name,
                   resume=True if args.wandb_resume_id is not None else None,
                   id=args.wandb_resume_id)
        wandb.config.update(args, allow_val_change=True)


    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.num_gpu > 1:
            _logger.warning(
                'Using more than one GPU per process in distributed mode is not allowed.Setting num_gpu to 1.')
            args.num_gpu = 1

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    assert args.rank >= 0

    if args.distributed:
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on %d GPUs.' % args.num_gpu)

    torch.manual_seed(args.seed + args.rank)

    if args.dataset_type == "foodx251":

        import models.cswin251
        datasets_train = [FoodX251Dataset(args.data, "train", dataset_reduction_factor=args.data_reduction_factor)]
        dataset_eval = FoodX251Dataset(args.data, "val", dataset_reduction_factor=args.data_reduction_factor)

        cluster_df = pd.read_csv(args.cluster_file,
                                      dtype='Int64')

        if args.step == 1:
            for index, iccv_hydra_dataset in enumerate(cluster_df[1:]):
                datasets_train.append(FoodX251Dataset(args.data, "train", dataset_reduction_factor=args.data_reduction_factor))

    elif args.dataset_type == "food101":
        import models.cswin101
        datasets_train = [Food101(args.data, "train", dataset_reduction_factor=args.data)]
        dataset_eval = Food101(args.data, "val", dataset_reduction_factor=args.data)

        cluster_df = pd.read_csv(args.cluster_file,
                                      dtype='Int64')
        if args.step == 1:
            for index, iccv_hydra_dataset in enumerate(cluster_df[1:]):
                datasets_train.append(Food101(args.data, "train", dataset_reduction_factor=args.data_reduction_factor))

    elif args.dataset_type == "food1k":

        import models.cswin1k
        datasets_train = [Food1k(args.data, None, "train", dataset_reduction_factor=args.data_reduction_factor)]
        dataset_eval = Food1k(args.data, None, "val", dataset_reduction_factor=args.data_reduction_factor)

        cluster_df = pd.read_csv(args.cluster_file,
                                      dtype='Int64')
        if args.step == 1:
            for index, dataset in enumerate(cluster_df[1:]):
                datasets_train.append(
                    Food1k(args.data, None, "train", dataset_reduction_factor=args.data_reduction_factor))


        print("DATASETS", datasets_train)

    print("=========== DATA LOADED ===========")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        checkpoint_path=None,
        img_size=args.img_size,
        use_chk=args.use_chk)


    if args.step == 1:
        for param in model.hydra_heads.parameters():
            param.requires_grad = True

        for param in model.hydra_heads[0].parameters():
            param.requires_grad = False

    elif args.step == 2:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc_out.parameters():
            param.requires_grad = True



    #model.fc_out.weight[:1000, 1000].mul_(torch.eye(1000))

    #model.fc_out.weight[:1000, 1000].required_grad = False
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    summary(model.to(device), (3, 224, 224))
    print('')

    if args.initial_checkpoint != "":
        load_checkpoint(model, args.initial_checkpoint, True, strict=False)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Num:', count_parameters(model) / 1e6)

    if args.local_rank == 0:
        _logger.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters()])))

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    args.native_amp = True
    use_amp = 'native'

    if args.num_gpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
        assert not args.channels_last, "Channels last not supported with DP, use DDP."
    else:
        model.cuda()
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)
            print("Using channels last")

    optimizer = create_optimizer(args, model)

    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')



    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:


        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    model_ema = None
    args.model_ema = False
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.resume)

    if args.distributed:
        if args.sync_bn:
            assert not args.split_bn
            try:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if args.local_rank == 0:
                    _logger.info(
                        'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                        'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
            except Exception as e:
                _logger.error('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
        if args.local_rank == 0:
            _logger.info("Using native Torch DistributedDataParallel.")
        model = NativeDDP(model, device_ids=[args.local_rank],
                          find_unused_parameters=True)  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    collate_fn = None
    mixup_fns = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    mixup_active = False
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            if args.dataset_type == "logmeal_hydra":
                mixup_fns = [Mixup(**mixup_args, num_classes=num_head_class) for num_head_class in
                             model.hydra_heads_class_counts]

            elif args.dataset_type == "iccv_hydra":
                mixup_fns = [Mixup(**mixup_args, num_classes=num_head_class) for num_head_class in
                             model.hydra_heads_class_counts]
            else:
                mixup_fns = [Mixup(**mixup_args, num_classes=args.num_classes)]

    if num_aug_splits > 1:
        datasets_train = AugMixDataset(datasets_train, num_splits=num_aug_splits)

    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    if args.aa == 'None':
        args.aa = None

    # batch_sizes = [4, 4, 32, 8, 4]
    # batch_sizes = len(datasets_train) * [4]

    samples_weights_list = []

    for dataset_idx, dataset_train in enumerate(datasets_train):
        if dataset_idx:
            samples_weights, _ = make_weights_for_balanced_classes(dataset_train,
                                                                   num_cluster=dataset_idx)
        else:
            samples_weights, samples_median = make_weights_for_balanced_classes(dataset_train,
                                                                                args.num_classes)
        samples_weights = torch.tensor(samples_weights, dtype=torch.float)
        samples_weights_list += torch.unsqueeze(samples_weights, 0)


    loaders_train = [create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size // 8 if (args.bs_reg and dataset_idx) else args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        weights=None if (args.step==0 or args.step==2) else samples_weights_list[dataset_idx],
        epoch_length=samples_median * args.num_classes // 8 if (args.bs_reg and dataset_idx) else samples_median * args.num_classes,

    ) for dataset_idx, dataset_train in enumerate(datasets_train)]

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'] if not args.TTA_enabled else 1,
        pin_memory=args.pin_mem,
    )

    if args.gradient_accumulation > 1:
        grad_scaler = GradScaler()
        print(" =============== GRADIENT ACCUMALATION ===============")
    else:
        grad_scaler = None

    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
    elif mixup_active:
        # smoothing is handled with mixup target transform
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        # train_loss_fn = {0: nn.CrossEntropyLoss(weight=
        #                                         torch.from_numpy(
        #                                             np.repeat(1 / num_classes, num_classes)).float()).cuda()}
        #
        # for head_name, head in iccv_cluster_df.iteritems():
        #     num_classes_cluster = len(head.dropna().values)
        #     weight = np.repeat(1 / (num_classes_cluster + 1), num_classes_cluster + 1)
        #     weight[0] /= (num_classes - num_classes_cluster)
        #     train_loss_fn[int(head_name)] = nn.CrossEntropyLoss(weight=
        #                                                         torch.from_numpy(weight).float()
        #                                                         ).cuda()
        #     break  # for head = 1

        train_loss_fn = nn.CrossEntropyLoss().cuda()

    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None


    if args.eval_checkpoint:  # evaluate the model
        if args.step == 2:
            load_checkpoint(model, args.eval_checkpoint, True, strict=False)
            model.fc_out.weight.data[:, :251] = torch.eye(251)

        val_metrics = validate(model, loader_eval, validate_loss_fn, args)
        print(f"Top-1 accuracy of the model is: {val_metrics['top1']:.1f}%")
        # return
    saver = None
    output_dir = ''
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.model,
            str(data_config['input_size'][-1])
        ])
        output_dir = get_outdir(output_base, 'train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:  # train the model
        for epoch in range(start_epoch, num_epochs):
            if args.distributed:
                for dataset_train in datasets_train:
                    loaders_train.sampler.set_epoch(epoch)

            train_metrics = train_epoch(
                epoch, model, loaders_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                grad_scaler=grad_scaler, amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fns)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.ema, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                print(eval_metrics.keys())
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            print(train_metrics)
            print(eval_metrics)
            print()

            if args.experiment_name is not None:
                log_dict = {
                    "train_loss": train_metrics["train_loss"],
                    "lr": train_metrics["lr"],
                    "epoch": epoch,
                    "batch": train_metrics["batch"]
                }

                for key, value in eval_metrics.items():
                    log_dict[key] = value

                print(log_dict)
                wandb.log(log_dict)

            update_summary(
                epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None, log_wandb=False)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))
        
def getFraLoss(weights):


    ii_values = []
    ij_values = []
    for id_mc_cluster, mc_cluster in neurons_distribution.items():
        labels_in_cluster = list(mc_cluster.keys())
        for neuron, id_class in mc_cluster.items():
            values = weights[id_class, :]
            ii_values.append(values[id_class].item()) #main head value
            ii_values.append(values[neuron].item()) #mc value

            for label in labels_in_cluster:
                if label != neuron:
                    ij_values.append(values[label].item())


    

    ii_values = torch.FloatTensor(ii_values)
    ij_values = torch.FloatTensor(ij_values)
    len_ij = len(ij_values)
    loss_ii = torch.mean((torch.exp(-ii_values)))
    sign_ij = np.sign(ij_values)
    loss_ij = torch.sum(sign_ij*(ij_values**2)) / (len_ij)# * (len_ij - 1))
 
    return loss_ii, loss_ij

def train_epoch(
        epoch, model, loaders, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir='', grad_scaler=None,
        amp_autocast=suppress, loss_scaler=None, model_ema=None,
        mixup_fn=None):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            for loader in loaders:
                loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()
    end = time.time()

    # last_idx = len(loader) - 1
    # num_updates = epoch * len(loader)

    loader_idx = 0
    print("LOADERS", loaders)
    last_idx = len(loaders[loader_idx]) - 1
    num_updates = epoch * len(loaders[loader_idx])
    num_iters_per_epoch = len(loaders[loader_idx])
    num_batch = num_updates
    iterators = {}

    # freq_labels = {0: np.zeros(1000),
    #                1: np.zeros(7)}
    model.to('cuda')
    for batch_idx in range(num_iters_per_epoch):
        last_batch = batch_idx == last_idx
        total_loss = torch.empty(1, device="cuda")
        train_metrics = OrderedDict()
        for loader_idx, loader in enumerate(loaders):
            for gradient_accumulation in range(args.gradient_accumulation):
                loss = None
                (input, target), iterator = get_next_batch(loader,
                                                           iterators[loader_idx]
                                                           if loader_idx in iterators
                                                           else None)
                iterators[loader_idx] = iterator

                last_batch = batch_idx == last_idx
                data_time_m.update(time.time() - end)

                if not args.prefetcher:
                    input, target = input.cuda(), target.cuda()
                    if mixup_fn is not None:
                        input, target = mixup_fn(input, target)
                if args.channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)

                with amp_autocast():

                    output = model(input)[loader_idx]
                    # print(outputs)
                    target_np = target.cpu().detach().numpy()

                    # print(f"==============={idx_head}=================")
                    if loader_idx:
                        num_cluster = loader_idx - 1
                        mp_target = torch.from_numpy(mapped_targets[target_np][:, num_cluster]).cuda()
                        loss = loss_fn(output, mp_target)
                        # unique, counts = np.unique(mp_target.cpu().detach().numpy(), return_counts=True)
                        # freq_labels[loader_idx][unique] += counts
                    else:
                        loss = loss_fn(output, target)
                        # unique, counts = np.unique(target_np, return_counts=True)
                        # freq_labels[loader_idx][unique] += counts

                    loss = loss / (args.gradient_accumulation)

                total_loss += loss

        loss = total_loss / (loader_idx + 1)  # check

        if args.gradient_accumulation > 1 and grad_scaler is not None:
            grad_scaler.scale(loss).backward()

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        if grad_scaler is not None:
            if batch_idx % args.gradient_accumulation == 0 and batch_idx != 0:
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad()
            if loss_scaler is not None:
                loss_scaler(
                    loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(), create_graph=second_order)
            else:
                loss.backward(create_graph=second_order)
                if args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)] '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f}) '
                    'Time: {rate:>4.0f}/s ({rate_avg:>4.0f}/s) '
                    'LR: {lr:.3e} '
                    'Data: {data_time.sum:.3f}'.format(
                        epoch,
                        batch_idx, len(loaders[0]),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)
        train_metrics.update([("train_loss", loss.item()),
                              ("lr", lr),
                              ("epoch", epoch),
                              ("batch", num_batch)
                              ])
        num_batch += 1
        if args.experiment_name is not None:
            wandb.log(train_metrics)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for
    # print(freq_labels)

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return train_metrics

def processTTAOutput(output):

    sum_output = None

    for pred in output:
        if sum_output is None:
            sum_output = pred
        else:
            sum_output += pred

    sum_output /= len(output)

    return sum_output


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    metrics = OrderedDict()
    head_names = ['all'] + [str(index) for index in range(0, num_heads)]
    heur_acc = AverageMeter()
    losses_m = defaultdict(AverageMeter)
    top1_m = defaultdict(AverageMeter)
    balanced_top1_m = defaultdict(AverageMeter)
    total_outputs = defaultdict(list)
    total_outputs_batch = defaultdict(list)
    total_targets_batch = defaultdict(list)
    total_targets = defaultdict(list)

    softmax = torch.nn.Softmax(dim=1)
    # top5_m = defaultdict(AverageMeter)
    nb_classes = args.num_classes
    confusion_matrix_all = torch.zeros(nb_classes, nb_classes)
    confusion_matrixs = defaultdict()
    accuracy_per_class = defaultdict()

    model.eval()
    data_transforms = torch.nn.Sequential(
        K.augmentation.RandomRotation(15),
        K.augmentation.RandomHorizontalFlip(),
        K.augmentation.RandomResizedCrop((224, 224), (0.8, 0.8)),
    ).cuda()
    k = 4
    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():

                if args.TTA_enabled:
                    TTA_COUNT = 5
                    img = input.repeat(TTA_COUNT, 1, 1, 1)
                    img[1:] = data_transforms(img[1:])
                    outputs = model(img)
                    ouputs_head = []
                    for output in outputs:
                        output, idx = output.topk(k, -2, True, True)
                        output = torch.einsum("bc->c", [output]).unsqueeze(0) / k
                        ouputs_head.append(output)
                        outputs = [output]
                else:
                    outputs = model(input)

                h0_num_classes = outputs[0].shape[1]
                target_np = target.cpu().detach().numpy()

                for idx_head, output in enumerate(outputs):

                    if idx_head not in confusion_matrixs:
                        confusion_matrixs[idx_head] = torch.zeros(nb_classes, nb_classes)

                    loss = None
                    # print(f":=============={idx_head}==============")
                    if idx_head == 0:
                        # print(target)
                        # print(torch.argmax(output, dim=1))
                        loss = loss_fn(output, target)
                        acc1, acc5 = accuracy(output, target, topk=(1, min(5, output.shape[1])))
                        top1_m[idx_head].update(acc1.item(), output.size(0))
                        total_outputs[idx_head] += torch.argmax(output.detach(), dim=1).cpu().tolist()
                        total_outputs_batch[idx_head] = torch.argmax(output.detach(), dim=1)
                        total_targets_batch[idx_head] = target
                        total_targets[idx_head] += target.cpu().tolist()
                        accum_ensemble = softmax(output)

                    else:
                        num_cluster = idx_head - 1
                        mp_target = torch.from_numpy(mapped_targets[target_np][:, num_cluster]).cuda()
                        out_ph = output.clone().cuda()  # output place_holder with the probabilities
                        out_ph = softmax(out_ph)
                        expanded_tesor = out_ph[:, 0].unsqueeze(1).repeat(1, h0_num_classes).cuda()
                        h0_vector_targets = np.arange(1, out_ph.shape[1])
                        expanded_tesor[:, real_targets[h0_vector_targets, num_cluster]] = \
                            out_ph[:, h0_vector_targets]

                        loss = loss_fn(output, mp_target)
                        acc1, _ = accuracy(output, mp_target, topk=(1, min(5, output.shape[1])))
                        top1_m[idx_head].update(acc1.item(), output.size(0))
                        total_outputs[idx_head] += torch.argmax(output.detach(), dim=1).cpu().tolist()
                        total_outputs_batch[idx_head] = torch.argmax(output.detach(), dim=1)
                        total_targets[idx_head] += mp_target.cpu().tolist()
                        total_targets_batch[idx_head] = mp_target
                        accum_ensemble = torch.add(accum_ensemble, expanded_tesor)

                    losses_m[idx_head].update(loss.data.item(), input.size(0))


                total_outputs["all"] += torch.argmax(accum_ensemble, dim=1).cpu().tolist()

                total_outputs_batch_all = torch.argmax(accum_ensemble, dim=1)

                for t, p in zip(total_targets_batch[0].view(-1), total_outputs_batch_all.view(-1)):
                    confusion_matrix_all[t.long(), p.long()] += 1

                y_max_scores, y_max_idx = accum_ensemble.max(dim=1)
                y_pred = y_max_idx  # predictions are really the inx \in [n_classes] with the highest scores

                for idx_head, confusion_matrix in confusion_matrixs.items():
                    for t, p in zip(total_targets_batch[idx_head].view(-1), total_outputs_batch[idx_head].view(-1)):
                        confusion_matrixs[idx_head][t.long(), p.long()] += 1

            print("BATCH", batch_idx)


    accuracy_per_class_heuristic = (confusion_matrix_all.diag() / confusion_matrix_all.sum(1)).tolist()

    for idx_head, confusion_matrix in confusion_matrixs.items():
        accuracy_per_class[idx_head] = (confusion_matrix.diag() / confusion_matrix.sum(1)).tolist()
        px = pd.DataFrame(confusion_matrix.numpy())
        px.to_csv('./confusion_matrix/confusionmatrix_{}_{}.csv'.format(idx_head, args.dataset_type))

    for head_idx, head_name in enumerate(head_names):
        metrics[f"val_loss_head_{head_idx}"] = losses_m[head_idx].avg
        metrics[f"val_balanced_top1_head_{head_idx}"] = 100 * balanced_accuracy_score(total_targets[head_idx],
                                                                                      total_outputs[head_idx])
        metrics[f"val_top1_head_{head_idx}"] = top1_m[head_idx].avg
        losses_m["all"].update(losses_m[head_idx].avg)
        top1_m["all"].update(top1_m[head_idx].avg)
        balanced_top1_m["all"].update(metrics[f"val_balanced_top1_head_{head_idx}"].item())

    list_acc_per_class = []
    for cls in range(args.num_classes):
        accuracies_per_class = []
        accuracies_per_class.append(accuracy_per_class_heuristic[cls] * 100)
        for idx_head, confusion_matrix in confusion_matrixs.items():
            accuracies_per_class.append(accuracy_per_class[idx_head][cls] * 100)
            metrics[f"val_class_{idx_head}_{cls}"] = accuracy_per_class[idx_head][cls] * 100

        list_acc_per_class.append(accuracies_per_class)

    dataframe_acc_per_class = pd.DataFrame(list_acc_per_class,
                                           columns=[['heuristic'] + list(confusion_matrixs.keys())])
    dataframe_acc_per_class.to_csv(f'accuracy_per_class_{args.dataset_type}.csv')



    metrics.update({'val_loss': losses_m["all"].avg,
                    'top1': top1_m["all"].avg,
                    'balanced_top1': balanced_top1_m["all"].avg,
                    'balanced_acc_ensemble_heuristic': 100 * balanced_accuracy_score(total_targets[0],
                                                                                     total_outputs["all"]),
                    'top1_ensemble_heuristic': 100 * accuracy_score(total_targets[0], total_outputs["all"])
                    })

    if args.experiment_name is not None:
        wandb.log(metrics)


    return metrics


if __name__ == '__main__':
    main()

