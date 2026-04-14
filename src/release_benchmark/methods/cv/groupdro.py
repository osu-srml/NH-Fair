import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from release_benchmark.metrics.fairness_metrics import calculate_metrics
from release_benchmark.utils.common import AverageMeter

from .erm import erm


class LossComputer:
    def __init__(
        self,
        criterion,
        is_robust,
        n_groups,
        group_count,
        alpha=None,
        gamma=0.1,
        adj=None,
        min_var_weight=0,
        step_size=0.01,
        normalize_loss=False,
        btl=False,
        device=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.criterion = criterion
        self.is_robust = is_robust
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl
        self.device = device

        self.n_groups = n_groups
        self.group_counts = torch.tensor(group_count).to(device)
        self.group_frac = self.group_counts / self.group_counts.sum()

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().to(device)
        else:
            self.adj = torch.zeros(self.n_groups).float().to(device)

        if is_robust:
            assert alpha, "alpha must be specified"

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).to(device) / self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).to(device)
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().to(device)

        self.reset_stats()

    def loss(self, yhat, y, group_idx=None, is_training=False):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg(
            (torch.argmax(yhat, 1) == y).float(), group_idx
        )

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        elif self.is_robust and self.btl:
            actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        return actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        if torch.all(self.adj > 0):
            adjusted_loss += self.adj / torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss / (adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss, group_count):
        adjusted_loss = self.exp_avg_loss + self.adj / torch.sqrt(self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0) <= self.alpha
        weights = mask.float() * sorted_frac / self.alpha
        last_idx = mask.sum()
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac * self.min_var_weight + weights * (
            1 - self.min_var_weight
        )

        robust_loss = sorted_loss @ weights

        # sort the weights back
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (
            group_idx == torch.arange(self.n_groups).unsqueeze(1).long().to(self.device)
        ).float()  # size: 2 x batch_size
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()  # avoid nans

        group_loss = (group_map @ losses.view(-1)) / group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma * (group_count > 0).float()) * (
            self.exp_avg_initialized > 0
        ).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized > 0) + (group_count > 0)

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups).to(self.device)
        self.update_data_counts = torch.zeros(self.n_groups).to(self.device)
        self.update_batch_counts = torch.zeros(self.n_groups).to(self.device)
        self.avg_group_loss = torch.zeros(self.n_groups).to(self.device)
        self.avg_group_acc = torch.zeros(self.n_groups).to(self.device)
        self.avg_per_sample_loss = 0.0
        self.avg_actual_loss = 0.0
        self.avg_acc = 0.0
        self.batch_count = 0.0

    def update_stats(
        self, actual_loss, group_loss, group_acc, group_count, weights=None
    ):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom == 0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom
        self.avg_group_loss = (
            prev_weight * self.avg_group_loss + curr_weight * group_loss
        )

        # avg group acc
        self.avg_group_acc = prev_weight * self.avg_group_acc + curr_weight * group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count / denom) * self.avg_actual_loss + (
            1 / denom
        ) * actual_loss

        # counts
        self.processed_data_counts += group_count
        if self.is_robust:
            self.update_data_counts += group_count * ((weights > 0).float())
            self.update_batch_counts += ((group_count * weights) > 0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count > 0).float()
        self.batch_count += 1

        # avg per-sample quantities
        group_frac = self.processed_data_counts / (self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc


class groupdro(erm):
    def __init__(self, args):
        super().__init__(args)

        if args.num_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        elif args.num_classes >= 2:
            self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.loss_computer = LossComputer(
            criterion=self.criterion,
            is_robust=True,
            n_groups=args.sensitive_attributes,
            group_count=args.groupdro_group_count,
            alpha=args.groupdro_alpha,
            gamma=args.groupdro_gamma,
            adj=None,
            step_size=0.01,
            normalize_loss=False,
            btl=args.groupdro_btl,
            min_var_weight=0,
        )

    def train(self, train_loader, epoch, args):
        model = self.model
        device = args.device

        model.train()
        if not args.no_progress:
            p_bar = tqdm(range(len(train_loader)))
        total_loss = AverageMeter()
        for batch_idx, (data, target, sensitive_attr) in enumerate(train_loader):
            data, target, sensitive_attr = (
                data.to(device),
                target.to(device),
                sensitive_attr.to(device),
            )
            output = model(data)
            self.optimizer.zero_grad()
            if args.num_classes == 1:
                target = target.float()
                output = output.squeeze()
            loss = self.loss_computer.loss(
                output, target, sensitive_attr, is_training=True
            )
            loss.backward()
            self.optimizer.step()
            total_loss.update(loss.item())
            if not args.no_progress:
                p_bar.set_description(
                    f"Train Epoch: {epoch + 1}/{args.epochs:4}. Iter: {batch_idx + 1:4}/{len(train_loader):4}. Loss: {total_loss.avg:.4f}.  "
                )
                p_bar.update()
        if not args.no_progress:
            p_bar.close()
        print(f"Average loss for epoch {epoch}: {total_loss.avg}")
        return total_loss.avg

    def validate(self, val_loader, epoch, args):
        model = self.model
        device = args.device
        model.eval()

        val_criterion = (
            nn.BCEWithLogitsLoss() if args.num_classes == 1 else nn.CrossEntropyLoss()
        )
        val_loss = AverageMeter()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        with torch.no_grad():
            for data, target, sensitive_attr in val_loader:
                data, target, sensitive_attr = (
                    data.to(device),
                    target.to(device),
                    sensitive_attr.to(device),
                )
                output = model(data)
                if args.num_classes == 1:
                    target = target.float()
                    output = output.squeeze()
                    prob = torch.sigmoid(output).flatten()
                else:
                    prob = F.softmax(output, dim=-1)
                loss = val_criterion(output, target)
                if loss.dim() > 0:
                    val_loss.update(loss.mean().item())
                else:
                    val_loss.update(loss.item())
                tol_output += prob.cpu().data.numpy().tolist()
                tol_target += target.cpu().data.numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()

        log_dict, _, _ = calculate_metrics(
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
