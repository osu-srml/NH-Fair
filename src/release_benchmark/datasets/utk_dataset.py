import os

import numpy as np
import torch

from .common import FairDataset, _train_val_test_split


class UTKDataset(FairDataset):
    def __init__(
        self,
        root,
        split,
        target,
        sensitive_attr,
        transform=None,
        load_memory=False,
        seed=42,
        return_idx=False,
    ):

        super().__init__(root, split, transform, seed, return_idx=return_idx)

        self.target = target
        self.sensitive_attr = sensitive_attr

        self.img_dir = os.path.join(self.root, "UTKface_inthewild")

        # Split indices
        if not os.path.exists(os.path.join(root, f"split_{seed}.pth")):
            self.img_list = os.listdir(self.img_dir)
            self.train_indices, self.val_indices, self.test_indices = (
                _train_val_test_split(len(self.img_list), seed=seed)
            )
            train_filenames = [self.img_list[idx] for idx in self.train_indices]
            val_filenames = [self.img_list[idx] for idx in self.val_indices]
            test_filenames = [self.img_list[idx] for idx in self.test_indices]
            torch.save(
                (train_filenames, val_filenames, test_filenames),
                os.path.join(root, f"split_{seed}.pth"),
            )
        else:
            train_filenames, val_filenames, test_filenames = torch.load(
                os.path.join(root, f"split_{seed}.pth")
            )

        if self.split == "train":
            self.image_file_list = train_filenames
        elif self.split == "val":
            self.image_file_list = val_filenames
        elif self.split == "test":
            self.image_file_list = test_filenames
        else:
            raise NotImplementedError
        self.load_memory = load_memory
        # if(load_memory):
        #     self.images = [Image.open(os.path.join(self.img_dir, file)).convert('RGB')  for file in tqdm(self.image_file_list)]

        self.age_list = [int(file.split("_")[0]) < 35 for file in self.image_file_list]
        self.ethnicity_list = [
            int(file.split("_")[2] == "0") for file in self.image_file_list
        ]
        self.gender_list = [int(file.split("_")[1]) for file in self.image_file_list]

        # Create age binary classification for 0-59 vs 60+
        self.age_binary_list = [
            0 if int(file.split("_")[0]) <= 60 else 1 for file in self.image_file_list
        ]

        # Create race x age combination: 4 groups
        # Group 0: race=0, age=0-59
        # Group 1: race=0, age=60+
        # Group 2: race=1, age=0-59
        # Group 3: race=1, age=60+
        self.ethnicity_age_list = [
            int(file.split("_")[2]) * 2 + (0 if int(file.split("_")[0]) <= 60 else 1)
            for file in self.image_file_list
        ]

        self.num_classes = {"gender": 2, "age": 117, "ethnicity": 2}.get(target)
        self.targets = {
            "gender": self.gender_list,
            "age": self.age_list,
            "ethnicity": self.ethnicity_list,
        }.get(target)
        self.num_sensitive_attributes = {
            "gender": 2,
            "ethnicity": 2,
            "ethnicity_age": 4,
        }.get(sensitive_attr)
        self.sensitive_attrs = {
            "gender": self.gender_list,
            "ethnicity": self.ethnicity_list,
            "ethnicity_age": self.ethnicity_age_list,
        }.get(sensitive_attr)

        self.sensitive_attrs = np.array(self.sensitive_attrs)
        self.targets = np.array(self.targets)

        if self.num_classes is None or self.num_sensitive_attributes is None:
            raise NotImplementedError("Invalid target or sensitive attribute")

        assert self.target != self.sensitive_attr
        self.load_from_file = True
        self.indices = list(range(len(self.image_file_list)))

    # def __getitem__(self, index):

    #     idx = self.indices[index]
    #     # if(self.load_memory):
    #     #     img=self.images[idx]
    #     # else:
    #     img = Image.open( os.path.join(self.img_dir, self.image_file_list[idx]))
    #     if img.mode != "RGB":
    #         img=img.convert('RGB')

    #     ta = self.targets[idx]
    #     sa = self.sensitive_attrs[idx]

    #     if self.transform:
    #         img = self.transform(img)

    #     if self.return_idx:
    #         return img, ta, sa, idx

    #     return img, ta, sa
