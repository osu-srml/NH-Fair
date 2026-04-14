import copy
import os

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linprog
from torch.utils.data import Dataset


def _train_val_test_split(dataset_length, seed=42, train_ratio=0.8, val_ratio=0.1):
    np.random.seed(seed)
    indices = list(range(dataset_length))
    np.random.shuffle(indices)

    train_split = int(train_ratio * dataset_length)
    val_split = int((train_ratio + val_ratio) * dataset_length)

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    return train_indices, val_indices, test_indices


def _train_val_split(dataset_length, seed=42, train_ratio=0.9):
    np.random.seed(seed)
    indices = list(range(dataset_length))
    np.random.shuffle(indices)

    val_split = int(train_ratio * dataset_length)
    train_indices = indices[:val_split]
    val_indices = indices[val_split:]
    return train_indices, val_indices


def add_sampled_to_labeled(labeled_dataset, unlabeled_dataset, sampled_idx):
    """Move sampled_idx from unlabeled to labeled dataset by updating indices."""
    sampled_set = set(sampled_idx)

    labeled_dataset.indices.extend(sampled_set)
    unlabeled_dataset.indices = list(set(unlabeled_dataset.indices) - sampled_set)
    labeled_dataset.reset(labeled_dataset.indices)
    unlabeled_dataset.reset(unlabeled_dataset.indices)

    print(f"Added {len(sampled_idx)} samples to labeled dataset")
    print(
        f"Labeled={len(labeled_dataset.indices)}, Unlabeled={len(unlabeled_dataset.indices)}"
    )

    return labeled_dataset, unlabeled_dataset


class FairDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        transform=None,
        seed=42,
        load_from_file=False,
        return_idx=False,
    ):

        # Base Folder and Images Folder
        self.root = root
        self.img_dir = None

        self.split = split
        self.indices = []  # indice for splited dataset
        self.sensitive_attrs = np.array([])
        self.targets = np.array([])

        self.transform = transform

        # If need to load from files
        self.load_from_file = load_from_file
        self.image_file_list = []

        # If store images in memory
        self.images = []

        self.method = None
        self.AY_proportion = None
        self.seed = seed
        self.return_idx = return_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        if self.load_from_file:
            img_name = os.path.join(self.img_dir, self.image_file_list[idx])
            img = Image.open(img_name)
            if img.mode != "RGB":
                img = img.convert("RGB")
        else:
            img = self.images[idx]
        ta = self.targets[idx]
        sa = self.sensitive_attrs[idx]

        if self.transform:
            img = self.transform(img)

        if self.method == "bm":
            target_bin = self.targets_bin[index] if self.targets_bin is not None else -1
            gc = self.group_weights[index] if self.group_weights is not None else 1.0
            return img, ta, sa, target_bin, gc

        if self.return_idx:
            return img, ta, sa, idx
        return img, ta, sa

    def reset(self, indices):
        self.indices = indices

    # Resample
    def get_AY_proportions(self):
        if self.AY_proportion:
            return self.AY_proportion
        A = self.sensitive_attrs[self.indices]
        Y = self.targets[self.indices]
        ttl = len(self.indices)

        len_A0Y0 = len([ay for ay in zip(A, Y) if ay == (0, 0)])
        len_A0Y1 = len([ay for ay in zip(A, Y) if ay == (0, 1)])
        len_A1Y0 = len([ay for ay in zip(A, Y) if ay == (1, 0)])
        len_A1Y1 = len([ay for ay in zip(A, Y) if ay == (1, 1)])

        assert (len_A0Y0 + len_A0Y1 + len_A1Y0 + len_A1Y1) == ttl, (
            "Problem computing train set AY proportion."
        )
        A0Y0 = len_A0Y0 / ttl
        A0Y1 = len_A0Y1 / ttl
        A1Y0 = len_A1Y0 / ttl
        A1Y1 = len_A1Y1 / ttl

        self.AY_proportion = [[A0Y0, A0Y1], [A1Y0, A1Y1]]

        return self.AY_proportion

    def get_A_proportions(self):
        AY = self.get_AY_proportions()
        ret = [AY[0][0] + AY[0][1], AY[1][0] + AY[1][1]]
        np.testing.assert_almost_equal(np.sum(ret), 1.0)
        return ret

    def get_Y_proportions(self):
        AY = self.get_AY_proportions()
        ret = [AY[0][0] + AY[1][0], AY[0][1] + AY[1][1]]
        np.testing.assert_almost_equal(np.sum(ret), 1.0)
        return ret

    def get_weights(self, resample_which):
        sens_attr, group_num = self.group_counts(resample_which)
        group_weights = [1 / x for x in group_num]
        sample_weights = [group_weights[int(i)] for i in sens_attr]
        return sample_weights

    def group_counts(self, resample_which="group"):

        A = self.sensitive_attrs[self.indices]
        Y = self.targets[self.indices]

        if resample_which == "group":
            group_array = A.tolist()
        elif resample_which == "balanced":
            num_labels = len(set(Y))
            group_array = (A * num_labels + Y).tolist()
        else:
            raise ValueError("resample_which must be 'group' or 'balanced'")

        unique_groups = set(group_array)
        self._group_counts = [group_array.count(group) for group in unique_groups]
        print("group_counts:", self._group_counts)
        return group_array, self._group_counts

    def split_labeled_unlabeled(self, ratio, seed=42):
        """Split train set into labeled and unlabeled subsets by ratio."""
        num_labeled = int(len(self.indices) * ratio)

        labeled_idx = self.indices[:num_labeled]
        unlabeled_idx = self.indices[num_labeled:]

        unlabeled_dataset = copy.deepcopy(self)
        unlabeled_dataset.return_idx = True
        unlabeled_dataset.reset(unlabeled_idx)

        self.reset(labeled_idx)

        print(
            f"Split complete: Labeled={len(labeled_idx)}, Unlabeled={len(unlabeled_idx)}"
        )
        return self, unlabeled_dataset

    def get_target_distro(self, target):
        """Count (target, bias) combinations for a specific target class."""
        num_biases = len(np.unique(self.sensitive_attrs))
        if isinstance(self.targets, torch.Tensor):
            return [
                torch.sum(
                    (self.targets[self.indices] == target)
                    & (self.sensitive_attrs[self.indices] == bias)
                )
                for bias in range(num_biases)
            ]

        return [
            np.sum(
                (self.targets[self.indices] == target)
                & (self.sensitive_attrs[self.indices] == bias)
            )
            for bias in range(num_biases)
        ]

    def get_kept_indices(self, target, target_prime, target_prime_new_distro):
        """Retain sample indices according to the new distribution."""
        to_keep_indices = []
        for bias, bias_distro in enumerate(target_prime_new_distro):
            mask = (self.targets[self.indices] == target_prime) & (
                self.sensitive_attrs[self.indices] == bias
            )
            indices_bias = np.array(self.indices)[mask]
            if len(indices_bias) > 0:
                sample = np.random.choice(
                    list(indices_bias),
                    size=min(len(indices_bias), bias_distro),
                    replace=False,
                )
                to_keep_indices.extend(sample)
        return to_keep_indices

    def solve_linear_program(self, target_distro, target_prime_distro):
        """Compute new data distribution via linear programming."""
        num_biases = len(np.unique(self.sensitive_attrs))
        obj = [-1] * num_biases

        lhs_ineq = np.eye(num_biases)
        rhs_ineq = np.array(target_prime_distro)

        lhs_eq = []
        target_distro = np.array(target_distro) / sum(target_distro)
        for prob, bias in zip(target_distro, range(num_biases - 1)):
            eq = [-prob] * num_biases
            eq[bias] = 1 - prob
            lhs_eq.append(eq)

        rhs_eq = [0] * (num_biases - 1)
        bnd = [(0, float("inf"))] * num_biases

        opt = linprog(
            c=obj,
            A_ub=lhs_ineq,
            b_ub=rhs_ineq,
            A_eq=lhs_eq,
            b_eq=rhs_eq,
            bounds=bnd,
            method="revised simplex",
        )
        sol = np.maximum(opt.x.astype(int), 1)
        return sol.tolist()

    def set_dro_info(self):
        """Compute groups_idx (target * bias) index for each sample."""
        num_targets = len(np.unique(self.targets))
        len(np.unique(self.sensitive_attrs))

        self.groups_idx = np.zeros(len(self.indices), dtype=int)
        for i, (t, b) in enumerate(
            zip(self.targets[self.indices], self.sensitive_attrs[self.indices])
        ):
            self.groups_idx[i] = t + (b * num_targets)

    def group_counts_bm(self):
        """Count samples per (target, sensitive_attr) group."""
        num_groups = len(np.unique(self.groups_idx))
        counts = np.zeros(num_groups)
        for i in range(num_groups):
            counts[i] = np.sum(self.groups_idx == i)
        return counts

    def calculate_bias_weights(self):
        """Compute inverse-frequency group weights."""
        group_counts = self.group_counts_bm()
        normalized_counts = group_counts / np.sum(group_counts)
        group_weights = 1 / (normalized_counts + 1e-6)

        self.group_weights = np.array([group_weights[idx] for idx in self.groups_idx])

    def bias_mimick(self):
        """Adjust dataset bias to balance the distribution across groups."""
        num_targets = len(np.unique(self.targets))
        len(np.unique(self.sensitive_attrs))

        print("Applying bias mimicking...")

        to_keep_indices = []
        for target in range(num_targets):
            target_distro = self.get_target_distro(target)

            for target_prime in range(num_targets):
                if target_prime == target:
                    indices_target = np.array(self.indices)[
                        self.targets[self.indices] == target
                    ]
                    to_keep_indices.extend(indices_target)
                else:
                    target_prime_distro = self.get_target_distro(target_prime)
                    target_prime_new_distro = self.solve_linear_program(
                        target_distro, target_prime_distro
                    )
                    to_keep_indices.extend(
                        self.get_kept_indices(
                            target, target_prime, target_prime_new_distro
                        )
                    )

        # Update `indices`.
        self.set_to_keep(to_keep_indices)

        self.set_dro_info()
        self.calculate_bias_weights()

        self.targets_bin = torch.zeros(
            (len(self.indices), len(np.unique(self.targets)))
        )
        for i, t in enumerate(self.targets[self.indices]):
            self.targets_bin[i, t] = 1
        print("Bias mimicking completed.")

    def set_to_keep(self, to_keep_idx):
        """Update indices after bias mimicking."""
        self.indices = list(set(to_keep_idx))
