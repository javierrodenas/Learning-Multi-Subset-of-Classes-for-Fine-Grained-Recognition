# Copyright 2021 Sea Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted for VOLO
'''
- load_pretrained_weights: load pretrained paramters to model in transfer learning
- resize_pos_embed: resize position embedding
- get_mean_and_std: calculate the mean and std value of dataset.
'''
import torch
import math

import logging
import os
from collections import OrderedDict
import torch.nn.functional as F
from timm.data.loader import MultiEpochsDataLoader, fast_collate, PrefetchLoader
_logger = logging.getLogger(__name__)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_transform
import torch
import torch.utils.data
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_transform
from timm.data.distributed_sampler import OrderedDistributedSampler
from timm.data.loader import MultiEpochsDataLoader, fast_collate, PrefetchLoader
from torch.utils.data import Sampler
from typing import List, Iterator

def resize_pos_embed(posemb, posemb_new):
    '''
    resize position embedding with class token
    example: 224:(14x14+1)-> 384: (24x24+1)
    return: new position embedding
    '''
    ntok_new = posemb_new.shape[1]

    posemb_tok, posemb_grid = posemb[:, :1], posemb[0,1:]  # posemb_tok is for cls token, posemb_grid for the following tokens
    ntok_new -= 1
    gs_old = int(math.sqrt(len(posemb_grid)))  # 14
    gs_new = int(math.sqrt(ntok_new))  # 24
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(
        0, 3, 1, 2)  # [1, 196, dim]->[1, 14, 14, dim]->[1, dim, 14, 14]
    posemb_grid = F.interpolate(
        posemb_grid, size=(gs_new, gs_new),
        mode='bicubic')  # [1, dim, 14, 14] -> [1, dim, 24, 24]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(
        1, gs_new * gs_new, -1)  # [1, dim, 24, 24] -> [1, 24*24, dim]
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)  # [1, 24*24+1, dim]
    return posemb


def resize_pos_embed_without_cls(posemb, posemb_new):
    '''
    resize position embedding without class token
    example: 224:(14x14)-> 384: (24x24)
    return new position embedding
    '''
    ntok_new = posemb_new.shape[1]
    posemb_grid = posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))  # 14
    gs_new = int(math.sqrt(ntok_new))  # 24
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(
        0, 3, 1, 2)  # [1, 196, dim]->[1, 14, 14, dim]->[1, dim, 14, 14]
    posemb_grid = F.interpolate(
        posemb_grid, size=(gs_new, gs_new),
        mode='bicubic')  # [1, dim, 14, 14] -> [1, dim, 24, 24]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(
        1, gs_new * gs_new, -1)  # [1, dim, 24, 24] -> [1, 24*24, dim]
    return posemb_grid


def resize_pos_embed_4d(posemb, posemb_new):
    '''return new position embedding'''
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    gs_old = posemb.shape[1]  # 14
    gs_new = posemb_new.shape[1]  # 24
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb
    posemb_grid = posemb_grid.permute(0, 3, 1,
                                      2)  # [1, 14, 14, dim]->[1, dim, 14, 14]
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bicubic')  # [1, dim, 14, 14] -> [1, dim, 24, 24]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1)  # [1, dim, 24, 24]->[1, 24, 24, dim]
    return posemb_grid

def load_state_dict(checkpoint_path, model, use_ema=False, num_classes=1000):
    # load state_dict
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(
            state_dict_key, checkpoint_path))
        if num_classes != 1000:
            # completely discard fully connected for all other differences between pretrained and created model
            del state_dict['head' + '.weight']
            del state_dict['head' + '.bias']
            old_aux_head_weight = state_dict.pop('aux_head.weight', None)
            old_aux_head_bias = state_dict.pop('aux_head.bias', None)

        old_posemb = state_dict['pos_embed']
        if model.pos_embed.shape != old_posemb.shape:  # need resize the position embedding by interpolate
            if len(old_posemb.shape) == 3:
                if int(math.sqrt(
                        old_posemb.shape[1]))**2 == old_posemb.shape[1]:
                    new_posemb = resize_pos_embed_without_cls(
                        old_posemb, model.pos_embed)
                else:
                    new_posemb = resize_pos_embed(old_posemb, model.pos_embed)
            elif len(old_posemb.shape) == 4:
                new_posemb = resize_pos_embed_4d(old_posemb, model.pos_embed)
            state_dict['pos_embed'] = new_posemb

        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_pretrained_weights(model,
                            checkpoint_path,
                            use_ema=False,
                            strict=True,
                            num_classes=1000):
    '''load pretrained weight for VOLO models'''
    state_dict = load_state_dict(checkpoint_path, model, use_ema, num_classes)
    model.load_state_dict(state_dict, strict=strict)


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

class AugMultBatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        > list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        > list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool, augmult: int = 8) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        self.augmult = augmult

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        assert batch_size % self.augmult == 0, "Batch size must be a multiple of augmult"
        self.batch_size = int(batch_size)
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            for augmult_idx in range(self.augmult):
                batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]

def create_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        no_aug=False,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_split=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        num_aug_splits=0,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        crop_pct=None,
        collate_fn=None,
        pin_memory=False,
        fp16=False,
        tf_preprocessing=False,
        use_multi_epochs_loader=False,
        persistent_workers=True,
        shuffle=True,
        weights=None,
        epoch_length=0
):
    re_num_splits = 0
    if re_split:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2
    dataset.transform = create_transform(
        input_size,
        is_training=is_training,
        use_prefetcher=use_prefetcher,
        no_aug=no_aug,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        tf_preprocessing=tf_preprocessing,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0,
    )
    if epoch_length == 0:
        epoch_length = len(dataset)
    print(f"Dataset_size: {epoch_length}")
    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    if (weights is not None) and is_training:
        sampler = torch.utils.data.WeightedRandomSampler(weights=weights,
                                                         num_samples=epoch_length,
                                                         replacement=True)
    elif shuffle and is_training:
        # Cannot statically verify that dataset is Sized
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        sampler = torch.utils.data.RandomSampler(dataset)  # type: ignore[arg-type]
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)  # type: ignore[arg-type]

    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate

    loader_class = torch.utils.data.DataLoader

    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader

    # if (weights is not None) and is_training:
    #     loader_args = dict(
    #         batch_size=batch_size,
    #         num_workers=num_workers,
    #         sampler=sampler,
    #         collate_fn=collate_fn,
    #         pin_memory=pin_memory,
    #         drop_last=False if is_training else True,
    #         persistent_workers=persistent_workers)
    # else:
    #     loader_args = dict(
    #         batch_size=1 if is_training else batch_size,
    #         shuffle=False,
    #         num_workers=num_workers,
    #         sampler=None if is_training else sampler,
    #         batch_sampler=AugMultBatchSampler(sampler, batch_size, True, 1) if is_training else None,
    #         collate_fn=collate_fn,
    #         pin_memory=pin_memory,
    #         drop_last=False if is_training else True,
    #         persistent_workers=persistent_workers)
    loader_args = dict(
        batch_size=1 if is_training else batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=None if is_training else sampler,
        batch_sampler=AugMultBatchSampler(sampler, batch_size, True, 1) if is_training else None,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False if is_training else True,
        persistent_workers=persistent_workers)

    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)
    if use_prefetcher:
        prefetch_re_prob = re_prob if is_training and not no_aug else 0.
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            fp16=fp16,
            re_prob=prefetch_re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits
        )
    return loader