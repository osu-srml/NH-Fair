import os

import torch
from torchvision import datasets

from .common import FairDataset, _train_val_test_split


class CelebaDataset(FairDataset):
    def __init__(
        self,
        root,
        dataset=None,
        split="train",
        target=31,
        sensitive_attr=20,
        transform=None,
        download=False,
        seed=42,
        return_idx=False,
    ):
        super().__init__(root, split, transform, seed, return_idx=return_idx)

        if dataset is None:
            self.dataset = datasets.CelebA(
                root=self.root, split="all", download=download
            )
        else:
            self.dataset = dataset
        self.seed = seed
        # Split indices
        if not os.path.exists(os.path.join(root, f"celeba/split_{seed}.pth")):
            self.train_indices, self.val_indices, self.test_indices = (
                _train_val_test_split(len(self.dataset), seed=seed)
            )
            torch.save(
                (self.train_indices, self.val_indices, self.test_indices),
                os.path.join(root, f"celeba/split_{seed}.pth"),
            )
        else:
            self.train_indices, self.val_indices, self.test_indices = torch.load(
                os.path.join(root, f"celeba/split_{seed}.pth")
            )

        if self.split == "train":
            self.indices = self.train_indices
        elif self.split == "val":
            self.indices = self.val_indices
        elif self.split == "test":
            self.indices = self.test_indices
        else:
            raise NotImplementedError

        self.attr = self.dataset.attr
        self.sensitive_attrs = self.dataset.attr[:, sensitive_attr]
        self.targets = self.dataset.attr[:, target]
        self.img_dir = os.path.join(
            self.dataset.root, self.dataset.base_folder, "img_align_celeba"
        )

        self.image_file_list = self.dataset.filename
        self.load_from_file = True

    def group_counts(self, resample_which="group"):

        A = self.sensitive_attrs[self.indices]
        Y = self.targets[self.indices]

        if resample_which == "group":
            group_array = A.tolist()
        elif resample_which == "balanced":
            num_labels = len(set(Y.tolist()))
            group_array = (A * num_labels + Y).tolist()
        else:
            raise ValueError("resample_which must be 'group' or 'balanced'")

        unique_groups = set(group_array)
        self._group_counts = [group_array.count(group) for group in unique_groups]
        print(self._group_counts)  # [52225, 42353, 9572, 57929]
        return group_array, self._group_counts
