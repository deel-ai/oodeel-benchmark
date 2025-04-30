import os
from pathlib import Path

import torch

from .resnet18_32x32 import ResNet18_32x32
from .resnet18_224x224 import ResNet18_224x224
from .resnet50 import ResNet50

REPO_PATH = Path(__file__).parent.parent.parent

CHECKPOINTS_DICT = {
    "cifar10": {
        "resnet18": {
            "class": ResNet18_32x32,
            "kwargs": {"num_classes": 10},
            "ckpt_path": "models/cifar10_resnet18_32x32_base_e100_lr0.1_default/"
            + "s2/best.ckpt",
        }
    },
    "cifar100": {
        "resnet18": {
            "class": ResNet18_32x32,
            "kwargs": {"num_classes": 100},
            "ckpt_path": "models/cifar100_resnet18_32x32_base_e100_lr0.1_default/"
            + "s2/best.ckpt",
        }
    },
    "imagenet200": {
        "resnet18": {
            "class": ResNet18_224x224,
            "kwargs": {"num_classes": 200},
            "ckpt_path": "models/imagenet200_resnet18_224x224_base_e90_lr0.1_default/"
            + "s2/best.ckpt",
        }
    },
    "imagenet": {
        "resnet50": {
            "class": ResNet50,
            "kwargs": {"num_classes": 1000},
            "ckpt_path": "models/pretrained_weights/resnet50_imagenet1k_v1.pth",
        }
    },
}


def get_network(dataset_name, model_name):
    model_dict = CHECKPOINTS_DICT[dataset_name][model_name]
    kwargs = model_dict["kwargs"]
    model = model_dict["class"](**kwargs)
    model.load_state_dict(torch.load(REPO_PATH / model_dict["ckpt_path"]))
    return model
