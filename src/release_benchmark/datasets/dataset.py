import os
import time

from torchvision import datasets

from .celeba_dataset import CelebaDataset
from .facet_dataset import FacetDataset
from .fairface_dataset import FairFaceDataset
from .fitz17k_dataset import FitzDataset
from .ham10000_dataset import HAM10000Dataset
from .transforms import getTransforms
from .utk_dataset import UTKDataset
from .waterbirds_dataset import WaterbirdsDataset

clip_mean = [0.481, 0.457, 0.408]
clip_std = [0.268, 0.261, 0.275]


class TransformTwo:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        view1 = self.transform(x)
        self.transform(x)
        return view1, view1


def load_dataset(args):
    read_start_time = time.time()

    if args.use_clip_tranform:
        transform_train_weak_aug, transform_test, transform_train_rand_aug = (
            getTransforms(mean=clip_mean, std=clip_std, size=args.img_size)
        )
    else:
        transform_train_weak_aug, transform_test, transform_train_rand_aug = (
            getTransforms(size=args.img_size)
        )
    if args.augment == "no":
        train_transform = transform_test
    elif args.augment == "weak":
        train_transform = transform_train_weak_aug
    elif args.augment == "strong":
        train_transform = transform_train_rand_aug
    else:
        raise NotImplementedError

    if args.dataset == "celeba":
        sensitive_attributes = 2  # gender 0 or 1
        num_classes = 2  # target 0 or 1
        root = os.path.join(args.data_path, "")
        base_dataset = datasets.CelebA(root=root, split="all", download=False)
        train_set = CelebaDataset(
            root=root,
            dataset=base_dataset,
            split="train",
            target=int(args.ta),
            transform=train_transform,
            seed=args.seed,
        )
        val_set = CelebaDataset(
            root=root,
            dataset=base_dataset,
            split="val",
            target=int(args.ta),
            transform=transform_test,
            seed=args.seed,
        )
        test_set = CelebaDataset(
            root=root,
            dataset=base_dataset,
            split="test",
            target=int(args.ta),
            transform=transform_test,
            seed=args.seed,
        )

    if args.dataset == "utk":
        if args.sa == "race" or args.sa == "ethnicity":
            sa = "ethnicity"
            ta = "gender"

        elif args.sa == "race_age" or args.sa == "ethnicity_age":
            sa = "ethnicity_age"
            ta = "gender"
        else:
            raise NotImplementedError
        assert args.ta == ta
        root = os.path.join(args.data_path, args.dataset)

        train_set = UTKDataset(
            root=root,
            target=ta,
            sensitive_attr=sa,
            split="train",
            load_memory=args.load_memory,
            transform=train_transform,
            seed=args.seed,
        )
        val_set = UTKDataset(
            root=root,
            target=ta,
            sensitive_attr=sa,
            split="val",
            load_memory=args.load_memory,
            transform=transform_test,
            seed=args.seed,
        )
        test_set = UTKDataset(
            root=root,
            target=ta,
            sensitive_attr=sa,
            split="test",
            load_memory=args.load_memory,
            transform=transform_test,
            seed=args.seed,
        )
        sensitive_attributes = train_set.num_sensitive_attributes
        num_classes = train_set.num_classes

    if args.dataset == "fairface":
        if args.sa == "race":
            sa = "race"
            ta = "gender"
        elif args.sa == "gender" or args.sa == "sex":
            ta = "race"
            sa = "gender"
        else:
            raise NotImplementedError
        assert args.ta == ta
        root = os.path.join(args.data_path, args.dataset)

        train_set = FairFaceDataset(
            root=root,
            target=ta,
            sensitive_attr=sa,
            split="train",
            transform=train_transform,
            seed=args.seed,
        )
        val_set = FairFaceDataset(
            root=root,
            target=ta,
            sensitive_attr=sa,
            split="val",
            transform=transform_test,
            seed=args.seed,
        )
        test_set = FairFaceDataset(
            root=root,
            target=ta,
            sensitive_attr=sa,
            split="test",
            transform=transform_test,
            seed=args.seed,
        )
        sensitive_attributes = train_set.num_sensitive_attributes
        num_classes = train_set.num_classes

    if args.dataset == "ham" or args.dataset == "ham10000":
        root = os.path.join(args.data_path, "ham/base")
        train_set = HAM10000Dataset(
            root=root,
            sensitive_attr=args.sa,
            split="train",
            transform=train_transform,
            seed=args.seed,
        )
        val_set = HAM10000Dataset(
            root=root,
            sensitive_attr=args.sa,
            split="val",
            transform=transform_test,
            seed=args.seed,
        )
        test_set = HAM10000Dataset(
            root=root,
            sensitive_attr=args.sa,
            split="test",
            transform=transform_test,
            seed=args.seed,
        )
        sensitive_attributes = 4 if args.sa == "sex_age" else 2
        num_classes = 2

    if args.dataset == "fitz" or args.dataset == "fitz17k":
        root = os.path.join(args.data_path, "fitz")
        train_set = FitzDataset(
            root=root,
            sensitive_attr=args.sa,
            split="train",
            transform=train_transform,
            seed=args.seed,
        )
        val_set = FitzDataset(
            root=root,
            sensitive_attr=args.sa,
            split="val",
            transform=transform_test,
            seed=args.seed,
        )
        test_set = FitzDataset(
            root=root,
            sensitive_attr=args.sa,
            split="test",
            transform=transform_test,
            seed=args.seed,
        )
        sensitive_attributes = train_set.num_sensitive_attributes
        num_classes = 2

    if args.dataset == "facet":
        root = os.path.join(args.data_path, args.dataset)
        train_set = FacetDataset(
            root=root,
            target=args.ta,
            split="train",
            transform=train_transform,
            seed=args.seed,
        )
        val_set = FacetDataset(
            root=root,
            target=args.ta,
            split="val",
            transform=transform_test,
            seed=args.seed,
        )
        test_set = FacetDataset(
            root=root,
            target=args.ta,
            split="test",
            transform=transform_test,
            seed=args.seed,
        )
        sensitive_attributes = 2
        num_classes = train_set.num_classes
    if args.dataset == "waterbirds":
        root = os.path.join(args.data_path, args.dataset)
        assert args.ta == "species"
        assert args.sa == "background"
        train_set = WaterbirdsDataset(
            root=root,
            split="train",
            transform=train_transform,
            load_memory=args.load_memory,
        )
        val_set = WaterbirdsDataset(
            root=root,
            split="val",
            transform=transform_test,
            load_memory=args.load_memory,
        )
        test_set = WaterbirdsDataset(
            root=root,
            split="test",
            transform=transform_test,
            load_memory=args.load_memory,
        )
        sensitive_attributes = 2
        num_classes = 2

    if "train_set" not in locals():
        raise ValueError(
            f"Unknown dataset: {args.dataset}. "
            f"Supported: celeba, utk, fairface, ham, fitz, facet, waterbirds."
        )

    print(f"Loading {args.dataset} dataset spend:{time.time() - read_start_time}")

    if args.method == "fis":
        train_set, unlabeled_trainset = train_set.split_labeled_unlabeled(
            args.fis_ratio
        )
        return (
            [train_set, unlabeled_trainset],
            val_set,
            test_set,
            sensitive_attributes,
            num_classes,
        )
    if args.method == "bm":
        train_set.bias_mimick()
        train_set.method = "bm"

    if args.method == "fscl":
        train_set.transform = TransformTwo(train_set.transform)
    return train_set, val_set, test_set, sensitive_attributes, num_classes
