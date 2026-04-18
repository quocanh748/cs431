# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Tianxiao Zhang
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

from .cached_image_folder import CachedImageFolder
from .imagenet22k_dataset import IN22KDATASET
from .samplers import SubsetRandomSampler

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR

    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp

CIFAR10_DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_DEFAULT_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_DEFAULT_MEAN = (0.5070, 0.4865, 0.4409)
CIFAR100_DEFAULT_STD = (0.2673, 0.2564, 0.2762)
STL10_DEFAULT_MEAN = (0.4467, 0.4398, 0.4066)
STL10_DEFAULT_STD = (0.2603, 0.2566, 0.2713)
FLOWERS102_DEFAULT_MEAN = (0.485, 0.456, 0.406)
FLOWERS102_DEFAULT_STD = (0.229, 0.224, 0.225)
CALTECH101_DEFAULT_MEAN = (0.485, 0.456, 0.406)
CALTECH101_DEFAULT_STD = (0.229, 0.224, 0.225)


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.TEST.SHUFFLE
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0:
        os.makedirs(config.DATA.DATA_PATH, exist_ok=True)

    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'tiny-imagenet':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        # Check if the formatted validation exists, if not use the default root
        # This keeps compatibility with the Kaggle setup while allowing local use
        kag_val = '/kaggle/working/val_formatted'
        if not is_train and os.path.exists(kag_val):
            root = kag_val
        
        dataset = datasets.ImageFolder(root, transform=transforms.Compose([transforms.Resize(config.DATA.IMG_SIZE), transform]))
        nb_classes = 200
    elif config.DATA.DATASET == 'cifar10':
        dataset = datasets.CIFAR10(root=config.DATA.DATA_PATH,
                                   train=is_train,
                                   transform=transforms.Compose([transforms.Resize(config.DATA.IMG_SIZE), transform]),
                                   download=True)
        nb_classes = 10
    elif config.DATA.DATASET == 'cifar100':
        dataset = datasets.CIFAR100(root=config.DATA.DATA_PATH,
                                    train=is_train,
                                    transform=transforms.Compose([transforms.Resize(config.DATA.IMG_SIZE), transform]),
                                    download=True)
        nb_classes = 100
    elif config.DATA.DATASET == 'stl10':
        # Trên Kaggle, nếu dùng /kaggle/input thì không được download.
        # Nhưng nếu dùng /kaggle/working thì có thể download thoải mái.
        download = True
        if 'kaggle/input' in config.DATA.DATA_PATH:
            download = False
            
        # Tìm kiếm file nhị phân gốc
        stl10_root = config.DATA.DATA_PATH
        found_bin = False
        for root_dir, dirs, files in os.walk(config.DATA.DATA_PATH):
            if 'train_X.bin' in files:
                if 'stl10_binary' in root_dir:
                    stl10_root = os.path.dirname(root_dir)
                else:
                    stl10_root = root_dir
                found_bin = True
                break
        
        if found_bin or download:
            import torchvision.datasets as datasets_torch
            original_base_folder = datasets_torch.STL10.base_folder
            if found_bin and not os.path.exists(os.path.join(stl10_root, original_base_folder)):
                datasets_torch.STL10.base_folder = ""
            
            try:
                # Nếu ở /kaggle/input mà không thấy bin (found_bin=False) và download=False, 
                # thì lệnh này sẽ tự văng lỗi của torchvision rất rõ ràng.
                dataset = datasets_torch.STL10(root=stl10_root,
                                             split='train' if is_train else 'test',
                                             transform=transforms.Compose([transforms.Resize(config.DATA.IMG_SIZE), transform]),
                                             download=download)
            finally:
                datasets_torch.STL10.base_folder = original_base_folder
        else:
            # Phương án dự phòng ImageFolder (chỉ dùng nếu thư mục chia theo class)
            train_path = os.path.join(config.DATA.DATA_PATH, 'train_images')
            test_path = os.path.join(config.DATA.DATA_PATH, 'test_images')
            if os.path.exists(train_path):
                root = train_path if is_train else test_path
                dataset = datasets.ImageFolder(root, transform=transforms.Compose([transforms.Resize(config.DATA.IMG_SIZE), transform]))
            else:
                raise RuntimeError(f"Không tìm thấy file .bin hoặc thư mục ảnh hợp lệ tại {config.DATA.DATA_PATH}. "
                                 f"Gợi ý: Hãy đổi --data-path thành /kaggle/working để tự động tải về bộ dữ liệu chuẩn.")
                
        nb_classes = 10
    elif config.DATA.DATASET == 'imagenet22K':
        prefix = 'ILSVRC2011fall_whole'
        if is_train:
            ann_file = prefix + "_map_train.txt"
        else:
            ann_file = prefix + "_map_val.txt"
        dataset = IN22KDATASET(config.DATA.DATA_PATH, ann_file, transform)
        nb_classes = 21841
    elif config.DATA.DATASET == 'flowers102':
        dataset = datasets.Flowers102(root=config.DATA.DATA_PATH,
                                     split='train' if is_train else 'test',
                                     transform=transforms.Compose([transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)), transform]),
                                     download=True)
        nb_classes = 102
    elif config.DATA.DATASET == 'caltech101':
        dataset = datasets.Caltech101(root=config.DATA.DATA_PATH,
                                      transform=transforms.Compose([
                                          transforms.Lambda(lambda x: x.convert('RGB')),
                                          transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
                                          transform
                                      ]),
                                      download=True)
        n_total = len(dataset)
        n_train = int(0.8 * n_total)
        n_val = n_total - n_train
        train_idx, val_idx = torch.utils.data.random_split(
            range(n_total), [n_train, n_val], generator=torch.Generator().manual_seed(config.SEED))
        if is_train:
            dataset = torch.utils.data.Subset(dataset, train_idx)
        else:
            dataset = torch.utils.data.Subset(dataset, val_idx)
        nb_classes = 101
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    # Barrier to ensure all processes wait for rank 0 to finish downloading/preparing
    if dist.is_initialized():
        dist.barrier()

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        if config.DATA.DATASET == 'cifar10':
            transform = create_transform(
                input_size=config.DATA.IMG_SIZE,
                is_training=True,
                color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                re_prob=config.AUG.REPROB,
                re_mode=config.AUG.REMODE,
                re_count=config.AUG.RECOUNT,
                interpolation=config.DATA.INTERPOLATION,
                mean=CIFAR10_DEFAULT_MEAN,
                std=CIFAR10_DEFAULT_STD,
            )
        elif config.DATA.DATASET == 'cifar100':
            transform = create_transform(
                input_size=config.DATA.IMG_SIZE,
                is_training=True,
                color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                re_prob=config.AUG.REPROB,
                re_mode=config.AUG.REMODE,
                re_count=config.AUG.RECOUNT,
                interpolation=config.DATA.INTERPOLATION,
                mean=CIFAR100_DEFAULT_MEAN,
                std=CIFAR100_DEFAULT_STD,
            )
        elif config.DATA.DATASET == 'stl10':
            transform = create_transform(
                input_size=config.DATA.IMG_SIZE,
                is_training=True,
                color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                re_prob=config.AUG.REPROB,
                re_mode=config.AUG.REMODE,
                re_count=config.AUG.RECOUNT,
                interpolation=config.DATA.INTERPOLATION,
                mean=STL10_DEFAULT_MEAN,
                std=STL10_DEFAULT_STD,
            )
        elif config.DATA.DATASET == 'flowers102':
            transform = create_transform(
                input_size=config.DATA.IMG_SIZE,
                is_training=True,
                color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                re_prob=config.AUG.REPROB,
                re_mode=config.AUG.REMODE,
                re_count=config.AUG.RECOUNT,
                interpolation=config.DATA.INTERPOLATION,
                mean=FLOWERS102_DEFAULT_MEAN,
                std=FLOWERS102_DEFAULT_STD,
            )
        elif config.DATA.DATASET == 'caltech101':
            transform = create_transform(
                input_size=config.DATA.IMG_SIZE,
                is_training=True,
                color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                re_prob=config.AUG.REPROB,
                re_mode=config.AUG.REMODE,
                re_count=config.AUG.RECOUNT,
                interpolation=config.DATA.INTERPOLATION,
                mean=CALTECH101_DEFAULT_MEAN,
                std=CALTECH101_DEFAULT_STD,
            )
        elif config.DATA.DATASET == 'tiny-imagenet':
            transform = create_transform(
                input_size=config.DATA.IMG_SIZE,
                is_training=True,
                color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                re_prob=config.AUG.REPROB,
                re_mode=config.AUG.REMODE,
                re_count=config.AUG.RECOUNT,
                interpolation=config.DATA.INTERPOLATION,
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            )
        else:
            transform = create_transform(
                input_size=config.DATA.IMG_SIZE,
                is_training=True,
                color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                re_prob=config.AUG.REPROB,
                re_mode=config.AUG.REMODE,
                re_count=config.AUG.RECOUNT,
                interpolation=config.DATA.INTERPOLATION,
            )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    if config.DATA.DATASET == 'cifar10':
        t.append(transforms.Normalize(CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD))
    elif config.DATA.DATASET == 'cifar100':
        t.append(transforms.Normalize(CIFAR100_DEFAULT_MEAN, CIFAR100_DEFAULT_STD))
    elif config.DATA.DATASET == 'stl10':
        t.append(transforms.Normalize(STL10_DEFAULT_MEAN, STL10_DEFAULT_STD))
    elif config.DATA.DATASET == 'tiny-imagenet':
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    elif config.DATA.DATASET == 'flowers102':
        t.append(transforms.Normalize(FLOWERS102_DEFAULT_MEAN, FLOWERS102_DEFAULT_STD))
    elif config.DATA.DATASET == 'caltech101':
        t.append(transforms.Normalize(CALTECH101_DEFAULT_MEAN, CALTECH101_DEFAULT_STD))
    else:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
