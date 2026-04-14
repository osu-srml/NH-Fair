from torchvision import transforms

from .randaugment import RandAugmentMC


def getTransforms(mean=None, std=None, size=224):

    if std is None:
        std = [0.229, 0.224, 0.225]
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    transform_train_weak_aug = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    transform_train_strong_aug = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.Resize((size, size)),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return transform_train_weak_aug, transform_test, transform_train_strong_aug
