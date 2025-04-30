import os

import torch
from torch.utils.data import DataLoader

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
}


def get_dataset(
    dataset_name: str,
    split: str,
    preprocessor_dataset_name: str,
    root_dir: str = "/datasets/openood",
):
    preprocessor = TestPreprocessor(preprocessor_dataset_name)

    num_classes = DATASETS_INFO[dataset_name]["num_classes"]
    if num_classes is None:
        num_classes = DATASETS_INFO[preprocessor_dataset_name]["num_classes"]

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


def _collate_fn(batch):
    data = torch.stack([item["data"] for item in batch])
    label = torch.LongTensor([item["label"] for item in batch])
    return tuple([data, label])


def get_dataloader(
    dataset_name,
    split,
    preprocessor_dataset_name,
    batch_size,
    num_workers=8,
    root_dir="/datasets/openood",
):
    dataset = get_dataset(
        dataset_name, split, preprocessor_dataset_name, root_dir=root_dir
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )
    return dataloader
