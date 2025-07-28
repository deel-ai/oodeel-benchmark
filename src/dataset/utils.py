import os

import torch
from torch.utils.data import DataLoader, Subset, RandomSampler
from collections import defaultdict
import numpy as np
import random

from .img_list_dataset import ImglistDataset
from .preprocess import TestPreprocessor


DATASETS_INFO = {
    # === ID DATASETS ===
    "imagenet": {
        "num_classes": 1000,
        "data_dir": "./images_largescale/",
        "imglist_pth": {
            "train": "./benchmark_imglist/imagenet/train_imagenet.txt",
            "val": "./benchmark_imglist/imagenet/val_imagenet.txt",
            "test": "./benchmark_imglist/imagenet/test_imagenet.txt",
        },
    },
    "imagenet200": {
        "num_classes": 200,
        "data_dir": "./images_largescale/",
        "imglist_pth": {
            "train": "./benchmark_imglist/imagenet200/train_imagenet200.txt",
            "val": "./benchmark_imglist/imagenet200/val_imagenet200.txt",
            "test": "./benchmark_imglist/imagenet200/test_imagenet200.txt",
        },
    },
    "cifar10": {
        "num_classes": 10,
        "data_dir": "./images_classic/",
        "imglist_pth": {
            "train": "./benchmark_imglist/cifar10/train_cifar10.txt",
            "val": "./benchmark_imglist/cifar10/val_cifar10.txt",
            "test": "./benchmark_imglist/cifar10/test_cifar10.txt",
        },
    },
    "cifar100": {
        "num_classes": 100,
        "data_dir": "./images_classic/",
        "imglist_pth": {
            "train": "./benchmark_imglist/cifar100/train_cifar100.txt",
            "val": "./benchmark_imglist/cifar100/val_cifar100.txt",
            "test": "./benchmark_imglist/cifar100/test_cifar100.txt",
        },
    },
    # === CIFAR 10 / 100 OOD DATASETS ===
    "tin": {
        "num_classes": None,
        "data_dir": "./images_classic/",
        "imglist_pth": {
            "test": "./benchmark_imglist/cifar10/test_tin.txt",
        },
    },
    "mnist": {
        "num_classes": None,
        "data_dir": "./images_classic/",
        "imglist_pth": {
            "test": "./benchmark_imglist/cifar10/test_mnist.txt",
        },
    },
    "svhn": {
        "num_classes": None,
        "data_dir": "./images_classic/",
        "imglist_pth": {
            "test": "./benchmark_imglist/cifar10/test_svhn.txt",
        },
    },
    "texture": {
        "num_classes": None,
        "data_dir": "./images_classic/",
        "imglist_pth": {
            "test": "./benchmark_imglist/cifar10/test_texture.txt",
        },
    },
    "places365": {
        "num_classes": None,
        "data_dir": "./images_classic/",
        "imglist_pth": {
            "test": "./benchmark_imglist/cifar10/test_places365.txt",
        },
    },
    # === IMAGENET OOD DATASETS ===
    "ssb_hard": {
        "num_classes": None,
        "data_dir": "./images_largescale/",
        "imglist_pth": {
            "test": "./benchmark_imglist/imagenet/test_ssb_hard.txt",
        },
    },
    "ninco": {
        "num_classes": None,
        "data_dir": "./images_largescale/",
        "imglist_pth": {
            "test": "./benchmark_imglist/imagenet/test_ninco.txt",
        },
    },
    "inaturalist": {
        "num_classes": None,
        "data_dir": "./images_largescale/",
        "imglist_pth": {
            "test": "./benchmark_imglist/imagenet/test_inaturalist.txt",
        },
    },
    "openimageo": {
        "num_classes": None,
        "data_dir": "./images_largescale/",
        "imglist_pth": {
            "test": "./benchmark_imglist/imagenet/test_openimage_o.txt",
        },
    },
    "cifar10_oe": {
        "num_classes": None,
        "data_dir": "./images_classic/",
        "imglist_pth": {
            "test": "./benchmark_imglist/cifar10/train_tin597.txt",
        },
    },
    "cifar100_oe": {
        "num_classes": None,
        "data_dir": "./images_classic/",
        "imglist_pth": {
            "test": "./benchmark_imglist/cifar100/train_tin597.txt",
        },
    },
    "imagenet200_oe": {
        "num_classes": None,
        "data_dir": "./images_largescale/",
        "imglist_pth": {
            "test": "./benchmark_imglist/imagenet200/train_imagenet800.txt",
        },
    },
    # === IMAGENET FULL-SPECTRUM (with covariate shift) ===
    "imagenet_fs": {
        "num_classes": 1000,
        "data_dir": "./images_largescale/",
        "imglist_pth": {
            "train": "./benchmark_imglist/imagenet/train_imagenet.txt",
            "val": "./benchmark_imglist/imagenet/val_imagenet.txt",
            "test": [
                "./benchmark_imglist/imagenet/test_imagenet.txt",  # in-distribution
                "./benchmark_imglist/imagenet/test_imagenet_r.txt",  # style changes
                "./benchmark_imglist/imagenet/test_imagenet_c.txt",  # corruptions
                "./benchmark_imglist/imagenet/test_imagenet_v2.txt",  # ImageNetV2
            ],
        },
    },
}


def get_dataset(
    dataset_name: str,
    split: str,
    preprocessor_dataset_name: str,
    root_dir: str = "/datasets/openood",
    **kwargs,
):
    preprocessor = TestPreprocessor(preprocessor_dataset_name)

    num_classes = DATASETS_INFO[dataset_name]["num_classes"]
    if num_classes is None:
        num_classes = DATASETS_INFO[preprocessor_dataset_name]["num_classes"]

    # create dataset
    if isinstance(DATASETS_INFO[dataset_name]["imglist_pth"][split], list):
        imglist_pths = DATASETS_INFO[dataset_name]["imglist_pth"][split]
        # one dataset per imglist_pth, then concatenate them
        datasets = []
        for imglist_pth in imglist_pths:
            dataset = ImglistDataset(
                name=dataset_name + "_" + split,
                imglist_pth=os.path.join(root_dir, imglist_pth),
                data_dir=os.path.join(
                    root_dir, DATASETS_INFO[dataset_name]["data_dir"]
                ),
                num_classes=num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=preprocessor,
            )
            datasets.append(dataset)
        dataset = torch.utils.data.ConcatDataset(datasets)
    else:
        # single dataset
        dataset = ImglistDataset(
            name=dataset_name + "_" + split,
            imglist_pth=os.path.join(
                root_dir, DATASETS_INFO[dataset_name]["imglist_pth"][split]
            ),
            data_dir=os.path.join(root_dir, DATASETS_INFO[dataset_name]["data_dir"]),
            num_classes=num_classes,
            preprocessor=preprocessor,
            data_aux_preprocessor=preprocessor,
        )
    return dataset


def _balanced_subset(ds, per_class, max_samples, seed):
    rnd = random.Random(seed)
    buckets = defaultdict(list)
    labels = ds.get_labels()
    for idx, label in enumerate(labels):
        buckets[label].append(idx)
    selected = []
    for idxs in buckets.values():
        rnd.shuffle(idxs)
        selected.extend(idxs[:per_class])
    rnd.shuffle(selected)
    if max_samples:
        selected = selected[:max_samples]
    return Subset(ds, selected)


def _collate_fn(batch):
    data = torch.stack([item["data"] for item in batch])
    label = torch.LongTensor([item["label"] for item in batch])
    return tuple([data, label])


def get_dataloader(
    dataset_name: str,
    split: str,
    preprocessor_dataset_name: str,
    batch_size=128,
    num_workers=8,
    root_dir="/datasets/openood",
    fit_subset_cfg=None,
    **kwargs,
):
    ds = get_dataset(
        dataset_name, split, preprocessor_dataset_name, root_dir=root_dir, **kwargs
    )

    # ---------- apply subset only for training split ----------
    if split == "train" and fit_subset_cfg:
        ds = _balanced_subset(
            ds,
            per_class=fit_subset_cfg.get("per_class", 0) or len(ds),
            max_samples=fit_subset_cfg.get("max_samples"),
            seed=fit_subset_cfg.get("seed", 0),
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
    )
