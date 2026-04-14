import os

import pandas as pd
import torch

from .common import FairDataset, _train_val_test_split


class FacetDataset(FairDataset):
    def __init__(
        self,
        root,
        split="train",
        seed=42,
        target="class",
        transform=None,
        return_idx=False,
    ):  # class class_num visible_face
        super().__init__(
            root, split=split, transform=transform, seed=seed, return_idx=return_idx
        )
        # Load annotations and filter samples where class2 is not 'none'
        self.annotations = pd.read_csv(
            os.path.join(root, "annotations/annotations.csv")
        )
        self.img_dir = os.path.join(root, "img224")
        self.target = target

        self.annotations = self.annotations[
            (self.annotations["gender_presentation_masc"] == 1)
            | (self.annotations["gender_presentation_fem"] == 1)
        ]
        self.annotations = self.annotations.reset_index()

        self.attribute_type = "gender_presentation_masc"
        self.sensitive_attrs = self.annotations["gender_presentation_fem"].to_numpy()

        if "class" in target:
            self.annotations = self.annotations[
                self.annotations["class2"].isna()
                | (self.annotations["class2"] == "none")
            ]

        if target == "class":
            self.class_to_idx = {
                label: idx
                for idx, label in enumerate(self.annotations["class1"].unique())
            }
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            self.num_classes = len(self.class_to_idx)
            self.targets = self.annotations["class1"].map(self.class_to_idx).to_numpy()
        elif target == "visible_face":
            self.num_classes = 2
            self.targets = self.annotations["visible_face"].to_numpy()
        else:
            raise ValueError(
                f"Unknown target: {target}. Choose 'class' or 'visible_face'."
            )

        if not os.path.exists(os.path.join(root, f"split_{seed}.pth")):
            self.train_indices, self.val_indices, self.test_indices = (
                _train_val_test_split(len(self.annotations), seed=seed)
            )
            torch.save(
                (self.train_indices, self.val_indices, self.test_indices),
                os.path.join(root, f"split_{seed}.pth"),
            )
        else:
            self.train_indices, self.val_indices, self.test_indices = torch.load(
                os.path.join(root, f"split_{seed}.pth"), weights_only=False
            )

        if self.split == "train":
            self.indices = self.train_indices
        elif self.split == "val":
            self.indices = self.val_indices
        elif self.split == "test":
            self.indices = self.test_indices
        else:
            raise NotImplementedError

        self.image_file_list = self.annotations["filename"].to_numpy()
        self.load_from_file = True
