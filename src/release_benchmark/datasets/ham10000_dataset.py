import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .common import FairDataset


def _train_val_test_split(meta_data, seed=42):
    np.random.seed(seed)
    patient_ids = np.unique(meta_data["lesion_id"])
    train_ids, val_test_ids = train_test_split(
        patient_ids, test_size=0.2, random_state=seed
    )
    val_ids, test_ids = train_test_split(val_test_ids, test_size=0.5, random_state=seed)

    train_meta = meta_data[meta_data["lesion_id"].isin(train_ids)]
    val_meta = meta_data[meta_data["lesion_id"].isin(val_ids)]
    test_meta = meta_data[meta_data["lesion_id"].isin(test_ids)]

    return train_meta, val_meta, test_meta


resize = transforms.Resize((224, 224))


class HAM10000Dataset(FairDataset):
    def __init__(
        self,
        root,
        split="train",
        sensitive_attr="sex",
        transform=None,
        seed=42,
        return_idx=False,
    ):
        super().__init__(
            root, split=split, transform=transform, seed=seed, return_idx=return_idx
        )

        if sensitive_attr == "age":
            sensitive_attr = "Age_binary"
        if sensitive_attr == "gender":
            sensitive_attr = "sex"
        self.sensitive_attr = sensitive_attr

        meta_data = pd.read_csv(os.path.join(root, "HAM10000_metadata.csv"))
        pathlist = meta_data["image_id"].values.tolist()
        paths = [os.path.join(root, "HAM10000_images", i + ".jpg") for i in pathlist]
        meta_data["Path"] = paths
        meta_data = meta_data.dropna(subset=["age", "sex"])

        # Preprocess metadata
        meta_data["sex"] = meta_data["sex"].map({"male": 0, "female": 1})
        label_mapping = {
            "nv": 0,
            "bkl": 0,
            "mel": 1,
            "akiec": 1,
            "bcc": 0,
            "df": 0,
            "vasc": 0,
        }
        meta_data["label"] = meta_data["dx"].map(label_mapping)
        meta_data["Age_binary"] = (
            meta_data["age"].astype(int).apply(lambda x: 0 if x <= 60 else 1)
        )

        # Split the dataset
        if not os.path.exists(os.path.join(root, f"split_{seed}.pth")):
            train_meta, val_meta, test_meta = _train_val_test_split(
                meta_data, seed=seed
            )
            torch.save(
                (train_meta, val_meta, test_meta),
                os.path.join(root, f"split_{seed}.pth"),
            )
        else:
            train_meta, val_meta, test_meta = torch.load(
                os.path.join(root, f"split_{seed}.pth"), weights_only=False
            )

        # Create sex x age combination: 4 groups (AFTER loading split to ensure it's in all dataframes)
        # Group 0: sex=0 (male), age<=60
        # Group 1: sex=0 (male), age>60
        # Group 2: sex=1 (female), age<=60
        # Group 3: sex=1 (female), age>60
        for df in [train_meta, val_meta, test_meta]:
            if "sex_age" not in df.columns:
                df["sex_age"] = df["sex"] * 2 + df["Age_binary"]

        self.dataset = {"train": train_meta, "val": val_meta, "test": test_meta}

        self.data = self.dataset[split]
        # self.paths = self.data['Path'].values

        # Use sensitive_attr in filename to support multiple sensitive attributes
        # New format: split_{seed}_loaded_data_{split}_{sensitive_attr}.pth
        # Old format (for backward compatibility): split_{seed}_loaded_data_{split}.pth

        cache_file_new = os.path.join(
            root, f"split_{seed}_loaded_data_{split}_{sensitive_attr}.pth"
        )
        cache_file_old = os.path.join(root, f"split_{seed}_loaded_data_{split}.pth")

        # Try to load from cache (new format first, then old format for backward compatibility)
        cache_loaded = False
        if os.path.exists(cache_file_new):
            # Load from new format with sensitive_attr in filename
            self.images, self.targets, self.sensitive_attrs = torch.load(
                cache_file_new, weights_only=False
            )
            cache_loaded = True
            print(f"Loaded cached data from {cache_file_new}")
        elif os.path.exists(cache_file_old) and sensitive_attr in ["sex", "Age_binary"]:
            # Only use old format for old sensitive attributes (sex, Age_binary)
            # Don't use old cache for new attributes like sex_age
            self.images, self.targets, self.sensitive_attrs = torch.load(
                cache_file_old, weights_only=False
            )
            cache_loaded = True
            print(f"Loaded cached data from {cache_file_old} (legacy format)")

        if not cache_loaded:
            # Generate and cache the data
            print(
                f"Generating cache for {split} split with sensitive_attr={sensitive_attr}..."
            )
            self.images = []
            self.targets = []
            self.sensitive_attrs = []

            for idx in tqdm(range(len(self.data))):
                meta = self.data.iloc[idx]
                img = Image.open(meta["Path"])
                img = img.convert("RGB")
                img = resize(img)
                self.images.append(img)
                label = meta["label"]
                sens_att = meta[sensitive_attr]
                self.targets.append(label)
                self.sensitive_attrs.append(sens_att)

            # Save with new format (includes sensitive_attr in filename)
            torch.save(
                (self.images, self.targets, self.sensitive_attrs), cache_file_new
            )
            print(f"Saved cache to {cache_file_new}")

        self.sensitive_attrs = np.array(self.sensitive_attrs)
        self.targets = np.array(self.targets)

        self.return_idx = return_idx

        self.indices = list(range(len(self.data)))
