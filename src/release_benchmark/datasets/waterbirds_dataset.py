import os

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .common import FairDataset


class WaterbirdsDataset(FairDataset):
    def __init__(
        self, root, split="train", transform=None, load_memory=False, return_idx=False
    ):

        super().__init__(root, split=split, transform=transform, return_idx=return_idx)
        self.root = root
        self.split = split
        self.transform = transform

        metadata_path = os.path.join(root, "metadata.csv")
        self.metadata = pd.read_csv(metadata_path)
        split_mapping = {"train": 0, "val": 1, "test": 2}

        if split not in split_mapping:
            raise ValueError("Split must be one of: 'train', 'val', 'test'")

        self.metadata = self.metadata[
            self.metadata["split"] == split_mapping[split]
        ].reset_index(drop=True)

        self.sensitive_attrs = self.metadata["place"].to_numpy()
        self.targets = self.metadata["y"].to_numpy()
        self.image_file_list = self.metadata["img_filename"].to_list()
        self.load_memory = load_memory

        if self.load_memory:
            self.images = [
                Image.open(os.path.join(self.root, file)).convert("RGB")
                for file in tqdm(self.image_file_list)
            ]
            self.load_from_file = False
        else:
            self.load_from_file = True

        self.indices = list(range(len(self.metadata)))
        self.img_dir = self.root

    def split_data(self, mode="all", split_size=2900):
        """
        Split dataset for parallel processing
        Args:
            mode: 'all', 'first', or 'second'
            split_size: number of samples in each split
        """
        original_len = len(self.indices)

        if mode == "first":
            # Take first split_size samples
            self.indices = self.indices[:split_size]
            print(
                f"[WaterbirdsDataset] Using FIRST {split_size} samples: {original_len} -> {len(self.indices)}"
            )
        elif mode == "second":
            # Take last split_size samples
            # start_idx = max(0, len(self.indices) - split_size)
            self.indices = self.indices[split_size:]
            print(
                f"[WaterbirdsDataset] Using LAST {split_size} samples: {original_len} -> {len(self.indices)}"
            )
        else:
            print(f"[WaterbirdsDataset] Using ALL {original_len} samples")

        return self

    # def __getitem__(self, index):

    #     idx = self.indices[index]
    #     if self.load_memory:
    #         # Retrieve preloaded image
    #         img = self.images[idx]
    #     else:
    #         # Get metadata for the current index
    #         img_path = os.path.join(self.root, self.image_file_list[idx])
    #         img = Image.open(img_path).convert('RGB')

    #     sa = self.sensitive_attrs[idx]
    #     ta = self.targets[idx]
    #     if self.transform:
    #         img = self.transform(img)

    #     if self.return_idx:
    #         return img, ta, sa, idx

    #     return img, ta, sa
