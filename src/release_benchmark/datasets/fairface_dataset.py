import os
from typing import ClassVar

import pandas as pd
import torch

from .common import FairDataset, _train_val_split


class FairFaceDataset(FairDataset):
    gender_dict: ClassVar[dict[str, int]] = {"Male": 0, "Female": 1}
    race_dict: ClassVar[dict[str, int]] = {
        "White": 0,  # 18612
        "Black": 1,  # 13789
        "Latino_Hispanic": 2,  # 14990
        "East Asian": 3,  # 13837
        "Southeast Asian": 4,  # 12210
        "Indian": 5,  # 13835
        "Middle Eastern": 6,  # 10425
    }

    def __init__(
        self,
        root="./data/fairface",
        split="train",
        target="gender",
        sensitive_attr="race",
        transform=None,
        seed=42,
        return_idx=False,
    ):
        super().__init__(root, split, transform, seed, return_idx=return_idx)
        self.target = target
        self.sensitive_attr = sensitive_attr

        if split == "train" or split == "val":
            csv_file = os.path.join(self.root, "fairface_label_train.csv")
        elif split == "test":
            csv_file = os.path.join(self.root, "fairface_label_val.csv")

        self.data = pd.read_csv(csv_file)
        num_images = len(self.data)
        train_indices, val_indices = _train_val_split(num_images, seed=seed)

        if split == "train":
            self.indices = train_indices
            self.pandas_list = self.data.iloc[train_indices].reset_index(drop=True)
        elif split == "val":
            self.pandas_list = self.data.iloc[val_indices].reset_index(drop=True)
        elif split == "test":
            self.pandas_list = (
                self.data
            )  # another file 'fairface_label_val.csv', no need to split
        self.pandas_list = self.pandas_list.reset_index(drop=True)

        self.pandas_list["gender"] = self.pandas_list["gender"].map(self.gender_dict)
        self.pandas_list["race"] = self.pandas_list["race"].map(self.race_dict)

        self.ethnicity_list = self.pandas_list["race"].tolist()
        self.gender_list = self.pandas_list["gender"].tolist()

        self.image_file_list = self.pandas_list["file"].tolist()

        self.num_classes = {"gender": 2, "race": 7}.get(self.target)
        self.targets = {
            "gender": self.pandas_list["gender"].to_numpy(),
            "race": self.pandas_list["race"].to_numpy(),
        }.get(self.target)

        self.num_sensitive_attributes = {"gender": 2, "race": 7}.get(
            self.sensitive_attr
        )
        self.sensitive_attrs = {
            "gender": self.pandas_list["gender"].to_numpy(),
            "race": self.pandas_list["race"].to_numpy(),
        }.get(self.sensitive_attr)

        if self.num_classes is None or self.targets is None:
            raise ValueError("Invalid target. Choose from 'gender' or 'race'.")

        if self.num_sensitive_attributes is None or self.sensitive_attrs is None:
            raise ValueError(
                "Invalid sensitive attribute. Choose from 'gender' or 'race'."
            )

        assert self.target != self.sensitive_attr

        self.indices = list(range(len(self.image_file_list)))
        self.img_dir = self.root
        self.load_from_file = True

    def get_AY_proportions(self):
        A = self.sensitive_attrs[self.indices]
        Y = self.targets[self.indices]
        ttl = len(self.indices)

        AY_counts = {(a, y): 0 for a in set(A) for y in set(Y)}
        for a, y in zip(A, Y):
            AY_counts[(a, y)] += 1

        self.AY_proportion = {
            (a, y): count / ttl for (a, y), count in AY_counts.items()
        }
        return self.AY_proportion

    def get_A_proportions(self):
        AY = self.get_AY_proportions()
        A_counts = {}
        for (a, _y), proportion in AY.items():
            A_counts[a] = A_counts.get(a, 0) + proportion
        return A_counts

    def get_Y_proportions(self):
        AY = self.get_AY_proportions()
        Y_counts = {}
        for (_a, y), proportion in AY.items():
            Y_counts[y] = Y_counts.get(y, 0) + proportion
        return Y_counts

    def get_weights(self, resample_which):
        group_array, group_counts = self.group_counts(resample_which)
        group_weights = [1 / count for count in group_counts]
        sample_weights = [group_weights[group] for group in group_array]
        return sample_weights

    def group_counts(self, resample_which="group"):
        A = self.sensitive_attrs[self.indices]
        Y = self.targets[self.indices]

        if resample_which == "group":
            group_array = A.tolist()
        elif resample_which == "balanced":
            group_array = [a * len(set(Y)) + y for a, y in zip(A, Y)]
        else:
            raise ValueError("resample_which must be 'group' or 'balanced'")

        unique_groups = set(group_array)
        self._group_counts = [group_array.count(group) for group in unique_groups]
        print(self._group_counts)

        return group_array, self._group_counts

    def split_labeled_unlabeled(self, ratio, seed=42):
        torch.manual_seed(seed)
        num_labeled = int(len(self.indices) * ratio)

        labeled_idx = self.indices[:num_labeled]
        unlabeled_idx = self.indices[num_labeled:]

        self.indices = labeled_idx
        self.reset(self.indices)

        unlabeled_dataset = FairFaceDataset(
            self.root,
            self.split,
            self.target,
            self.sensitive_attr,
            self.transform,
            self.seed,
            return_idx=True,
        )
        unlabeled_dataset.reset(unlabeled_idx)

        print(f" Labeled={len(labeled_idx)}, Unlabeled={len(unlabeled_idx)}")

        return self, unlabeled_dataset
