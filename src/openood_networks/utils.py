from pathlib import Path

import torch
import yaml

from .resnet18_32x32 import ResNet18_32x32
from .resnet18_224x224 import ResNet18_224x224
from .resnet50 import ResNet50

REPO_PATH = Path(__file__).parent.parent.parent
CFG_ROOT = REPO_PATH / "configs"

_CLASS_MAP = {
    "ResNet18_32x32": ResNet18_32x32,
    "ResNet18_224x224": ResNet18_224x224,
    "ResNet50": ResNet50,
}

_NETWORK_CFG = yaml.safe_load(open(CFG_ROOT / "openood_networks.yaml"))


def get_network(dataset_name, model_name, device=None):
    model_dict = _NETWORK_CFG[dataset_name][model_name]
    cls = _CLASS_MAP[model_dict["class"]]
    kwargs = model_dict.get("kwargs", {})
    model = cls(**kwargs)
    ckpt = REPO_PATH / model_dict["ckpt_path"]
    model.load_state_dict(torch.load(ckpt))
    if device is not None:
        model.to(device)
    return model
