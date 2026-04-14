import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .common import FairDataset


def _train_val_test_split(all_meta, seed=42):
    np.random.seed(seed)

    sub_train, sub_val_test = train_test_split(
        all_meta["id"], test_size=0.2, random_state=seed
    )
    sub_val, sub_test = train_test_split(sub_val_test, test_size=0.5, random_state=seed)
    train_meta = all_meta[all_meta["id"].isin(sub_train)]
    val_meta = all_meta[all_meta["id"].isin(sub_val)]
    test_meta = all_meta[all_meta["id"].isin(sub_test)]
    return train_meta, val_meta, test_meta


resize = transforms.Resize((224, 224))


class FitzDataset(FairDataset):
    def __init__(
        self,
        root,
        split,
        sensitive_attr="skin_binary",
        transform=None,
        seed=42,
        return_idx=False,
    ):

        super().__init__(
            root, split=split, transform=transform, seed=seed, return_idx=return_idx
        )
        self.sensitive_attr = sensitive_attr

        if self.sensitive_attr == "skin_binary":
            self.num_sensitive_attributes = 2
        if self.sensitive_attr == "skin_type":
            self.num_sensitive_attributes = 6

        # preprocess
        path = os.path.join(root, "processed_fitz17k", "fitzpatrick17k.csv")
        annot_data = pd.read_csv(path)

        annot_data["id"] = range(len(annot_data))
        pathlist = annot_data["file"].values.tolist()
        paths = [os.path.join(root, "processed_fitz17k", "images", i) for i in pathlist]
        annot_data["Path"] = paths

        # Filter and preprocess
        annot_data = annot_data[annot_data["skin_type"] != -1]
        annot_data["binary_label"] = annot_data["three_partition_label"].map(
            lambda x: 1 if x == "malignant" else 0
        )
        annot_data["skin_type"] = annot_data["skin_type"] - 1
        skin_lists = annot_data["skin_type"].values.tolist()
        annot_data["skin_binary"] = [0 if x <= 2 else 1 for x in skin_lists]

        # Split dataset
        if not os.path.exists(os.path.join(root, f"split_{seed}.pth")):
            train_meta, val_meta, test_meta = _train_val_test_split(
                annot_data, seed=seed
            )
            torch.save(
                (train_meta, val_meta, test_meta),
                os.path.join(root, f"split_{seed}.pth"),
            )
        else:
            train_meta, val_meta, test_meta = torch.load(
                os.path.join(root, f"split_{seed}.pth"), weights_only=False
            )

        self.dataset = {"train": train_meta, "val": val_meta, "test": test_meta}

        self.data = self.dataset[split]

        if not os.path.exists(
            os.path.join(root, f"split_{seed}_loaded_data_{split}.pth")
        ):
            self.images = []
            self.targets = []
            self.sensitive_attrs = []

            for idx in tqdm(range(len(self.data))):
                meta = self.data.iloc[idx]
                img = Image.open(meta["Path"])
                img = img.convert("RGB")
                img = resize(img)
                self.images.append(img)
                label = meta["binary_label"]
                sens_att = meta[self.sensitive_attr]
                self.targets.append(label)
                self.sensitive_attrs.append(sens_att)
            torch.save(
                (self.images, self.targets, self.sensitive_attrs),
                os.path.join(root, f"split_{seed}_loaded_data_{split}.pth"),
            )
        else:
            self.images, self.targets, self.sensitive_attrs = torch.load(
                os.path.join(root, f"split_{seed}_loaded_data_{split}.pth"),
                weights_only=False,
            )

        self.sensitive_attrs = np.array(self.sensitive_attrs)
        self.targets = np.array(self.targets)

        self.return_idx = return_idx

        self.indices = list(range(len(self.data)))
