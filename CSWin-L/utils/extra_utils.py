import time

import torch
import torch.utils.data
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_transform
from timm.data.distributed_sampler import OrderedDistributedSampler
from timm.data.loader import MultiEpochsDataLoader, fast_collate, PrefetchLoader
from torch.utils.data import Sampler
from typing import List, Iterator


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
