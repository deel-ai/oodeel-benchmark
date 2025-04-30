import torchvision.transforms as tvs_trans


normalization_dict = {
    "cifar10": [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    "cifar100": [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    "imagenet": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    "imagenet200": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
}

image_sizes = {
    "cifar10": 32,
    "cifar100": 32,
    "imagenet": 224,
    "imagenet200": 224,
}

images_pre_sizes = {
    "cifar10": 32,
    "cifar100": 32,
    "imagenet": 256,
    "imagenet200": 256,
}


class Convert:
    def __init__(self, mode="RGB"):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class TestPreprocessor:

    def __init__(self, dataset_name: str):
        self.pre_size = images_pre_sizes[dataset_name]
        self.image_size = image_sizes[dataset_name]
        self.mean = normalization_dict[dataset_name][0]
        self.std = normalization_dict[dataset_name][1]

        self.transform = tvs_trans.Compose(
            [
                Convert("RGB"),
                tvs_trans.Resize(
                    self.pre_size, interpolation=tvs_trans.InterpolationMode.BILINEAR
                ),
                tvs_trans.CenterCrop(self.image_size),
                tvs_trans.ToTensor(),
                tvs_trans.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def __call__(self, image):
        return self.transform(image)
