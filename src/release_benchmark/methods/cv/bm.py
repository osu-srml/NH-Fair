import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from release_benchmark.metrics.fairness_metrics import calculate_metrics
from release_benchmark.utils.common import AverageMeter

from .erm import erm


def get_samples_counts(all_labels_nb, all_bias):
    g_idxs = []
    g_counts = []
    full_idx = np.arange(len(all_bias))

    num_targets = len(np.unique(all_labels_nb))
    num_biases = len(np.unique(all_bias))

    for i in range(num_biases):
        for j in range(num_targets):
            g_idxs.append(full_idx[np.logical_and(all_bias == i, all_labels_nb == j)])
            g_counts.append(len(g_idxs[-1]))
    return g_idxs, g_counts


def under_sample_features(all_bias, all_feats, all_labels_nb):

    g_idxs, g_counts = get_samples_counts(all_labels_nb, all_bias)
    min_group = min(g_counts)

    to_keep_idx_all = []
    for _, group_idx in enumerate(g_idxs):
        to_keep_idx = np.random.choice(group_idx, min_group)
        to_keep_idx_all.extend(to_keep_idx)

    all_feats = all_feats[to_keep_idx_all]
    all_labels_nb = all_labels_nb[to_keep_idx_all]
    all_bias = all_bias[to_keep_idx_all]

    full_idx = np.arange(len(all_feats))
    np.random.shuffle(full_idx)

    all_feats = all_feats[full_idx]
    all_labels_nb = all_labels_nb[full_idx]
    all_bias = all_bias[full_idx]

    g_idxs, g_counts = get_samples_counts(all_labels_nb, all_bias)
    return all_feats, all_labels_nb


def over_sample_features(all_bias, all_feats, all_labels_nb):

    g_idxs, g_counts = get_samples_counts(all_labels_nb, all_bias)
    max_group = max(g_counts)

    for _idx, group_idx in enumerate(g_idxs):
        to_add = max_group - len(group_idx)
        to_add_idx = np.random.choice(group_idx, to_add)

        if to_add == 0:
            continue

        all_feats = np.concatenate((all_feats, all_feats[to_add_idx]), axis=0)
        all_labels_nb = np.concatenate(
            (all_labels_nb, all_labels_nb[to_add_idx]), axis=0
        )
        all_bias = np.concatenate((all_bias, all_bias[to_add_idx]), axis=0)

    full_idx = np.arange(len(all_feats))
    np.random.shuffle(full_idx)

    all_feats = all_feats[full_idx]
    all_labels_nb = all_labels_nb[full_idx]
    all_bias = all_bias[full_idx]

    g_idxs, g_counts = get_samples_counts(all_labels_nb, all_bias)

    return all_feats, all_labels_nb


class bm(erm):
    """Bias Mimicking (Qraitem et al., CVPR 2023).

    The original code declares a separate ``lr_layer`` for the prediction head,
    but the paper only reports a single learning rate for CelebA and does not
    discuss using different rates.  We therefore use the same ``args.lr`` for
    both the backbone and the prediction layer.
    """

    def __init__(self, args):
        super().__init__(args)

        self.pred = torch.nn.Linear(
            self.model.fc.in_features, self.model.fc.out_features
        ).to(args.device)
        self.criterion_bin = nn.BCEWithLogitsLoss(reduction="none")
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.optimizer_layer = self.setoptimizer_layer(self.pred, args)
        self.fe_scheduler_layer = None
        if args.StepLR_size > 1:
            self.fe_scheduler_layer = torch.optim.lr_scheduler.StepLR(
                self.optimizer_layer,
                step_size=args.StepLR_size,
                gamma=args.gamma,
            )

    def setoptimizer_layer(self, model, args):
        if args.optim == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-7
            )
        elif args.optim == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.weight_decay,
            )
        elif args.optim == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-7
            )
        return optimizer

    def train(self, train_loader, epoch, args):
        model = self.model
        device = args.device
        model.train()
        self.pred.train()
        if not args.no_progress:
            p_bar = tqdm(range(len(train_loader)))
        total_loss = AverageMeter()

        # BM
        all_outputs = []
        all_labels_nb = []
        all_gc = []
        all_feats = []
        all_bias = []
        sig = nn.Sigmoid()

        for batch_idx, (images, labels, biases, labels_bin, gc) in enumerate(
            train_loader
        ):
            labels, biases, labels_bin = (
                labels.to(device),
                biases.to(device),
                labels_bin.to(device),
            )

            images = images.to(device)
            logits, features = model.forward_return_feature(images)
            feat = F.normalize(features, dim=1)

            multi = torch.ones_like(labels_bin)
            multi[labels_bin == -1] = 0
            labels_bin[labels_bin == -1] = 0
            # Compute the binary classification loss.
            loss = self.criterion_bin(logits, labels_bin)
            loss = loss * multi
            loss = torch.sum(loss / torch.sum(multi))

            total_loss.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            all_outputs.append(sig(logits).cpu().detach().numpy())
            all_labels_nb.append(labels.cpu().detach().numpy())
            all_gc.append(gc.numpy())  # Sampling weights.
            all_bias.append(biases.cpu().detach().numpy())
            all_feats.append(feat.cpu().detach().numpy())

            if not args.no_progress:
                p_bar.set_description(
                    f"Train Epoch: {epoch + 1}/{args.epochs:4}. Iter: {batch_idx + 1:4}/{len(train_loader):4}. Loss: {total_loss.avg:.4f}.  "
                )
                p_bar.update()
        if not args.no_progress:
            p_bar.close()
        print(f"Average loss for epoch {epoch}: {total_loss.avg}")

        all_labels_nb = np.concatenate(all_labels_nb, axis=0)
        all_gc = np.concatenate(all_gc, axis=0)
        all_bias = np.concatenate(all_bias, axis=0)
        all_feats = np.concatenate(all_feats, axis=0)

        # Select the data resampling strategy.
        if args.bm_mode == "os":  # Oversampling.
            all_feats, all_labels_nb = over_sample_features(
                all_bias, all_feats, all_labels_nb
            )

        elif args.bm_mode == "us":  # Undersampling.
            all_feats, all_labels_nb = under_sample_features(
                all_bias, all_feats, all_labels_nb
            )

        batch_size = args.bs
        total_samples = len(all_labels_nb)
        num_batches = total_samples // batch_size

        # Generate shuffled random indices.
        all_idx = np.arange(total_samples)
        np.random.shuffle(all_idx)

        all_feats = all_feats[all_idx]
        all_labels_nb = all_labels_nb[all_idx]
        if args.bm_mode == "uw":  # Unweighted resampling mode.
            all_gc = all_gc[all_idx]

        # Train the prediction layer (linear layer).
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(total_samples, start + batch_size)

            feats = torch.from_numpy(all_feats[start:end]).to(device)
            labels = torch.from_numpy(all_labels_nb[start:end]).to(device)
            gc = torch.from_numpy(all_gc[start:end]).to(device)

            self.optimizer_layer.zero_grad()
            out_lr = self.pred(feats)

            # Compute the multiclass loss.
            if args.bm_mode == "uw":  # Weighted mode.
                loss = self.criterion(out_lr, labels) * gc
                loss = torch.mean(loss)
            else:  # Standard mode.
                loss = self.criterion(out_lr, labels)
                loss = torch.mean(loss)

            loss.backward()
            self.optimizer_layer.step()

        if self.fe_scheduler_layer is not None:
            self.fe_scheduler_layer.step()
        return total_loss.avg

    def validate(self, val_loader, epoch, args):
        model = self.model
        device = args.device
        model.eval()
        self.pred.eval()
        val_loss = AverageMeter()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        with torch.no_grad():
            for data, target, sensitive_attr in val_loader:
                data, target = data.to(device), target.to(device)
                # output= model(data)

                output, features = model.forward_return_feature(data)
                feat = F.normalize(features, dim=1)
                output = self.pred(feat)

                if self.num_classes == 1:
                    target = target.float()
                    output = output.squeeze()
                    prob = torch.sigmoid(output).flatten()
                else:
                    prob = F.softmax(output, dim=-1)
                loss = self.criterion(output, target)
                if loss.dim() > 0:
                    val_loss.update(loss.mean().item())
                else:
                    val_loss.update(loss.item())

                tol_output += prob.cpu().data.numpy().tolist()
                tol_target += target.cpu().data.numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()

        log_dict, _t_predictions, _aucs_subgroup = calculate_metrics(
            tol_output,
            tol_target,
            tol_sensitive,
            tol_index,
            args.sensitive_attributes,
            num_class=args.num_classes,
        )
        print(
            f"#####################################validation {epoch}#######################################"
        )
        print(log_dict, "\n")
        return val_loss.avg, log_dict

    def test(self, test_loader, epoch, args):
        model = self.model
        device = args.device
        model.eval()
        self.pred.eval()
        test_loss = AverageMeter()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        with torch.no_grad():
            for data, target, sensitive_attr in test_loader:
                data, target = data.to(device), target.to(device)

                output, features = model.forward_return_feature(data)
                feat = F.normalize(features, dim=1)
                output = self.pred(feat)

                if self.num_classes == 1:
                    target = target.float()
                    output = output.squeeze()
                    prob = torch.sigmoid(output).flatten()
                else:
                    prob = F.softmax(output, dim=-1)

                loss = self.criterion(output, target)
                if loss.dim() > 0:
                    test_loss.update(loss.mean().item())
                else:
                    test_loss.update(loss.item())

                tol_output += prob.cpu().data.numpy().tolist()
                tol_target += target.cpu().data.numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()

        log_dict, _t_predictions, _aucs_subgroup = calculate_metrics(
            tol_output,
            tol_target,
            tol_sensitive,
            tol_index,
            args.sensitive_attributes,
            num_class=args.num_classes,
        )
        print(
            "\n#####################################Test#######################################"
        )
        print(log_dict, "\n")
        return test_loss.avg, log_dict
