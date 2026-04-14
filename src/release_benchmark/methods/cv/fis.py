import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from release_benchmark.metrics.fairness_metrics import calculate_metrics
from release_benchmark.utils.common import AverageMeter

from .erm import erm

EPS = 1e-8


def constraints_dp(logits, attributes, labels, T=None, M=2, K=2):
    EPS = 1e-8
    prob = F.softmax(logits, dim=1)

    constraint = []
    H_noisy = torch.zeros((M, K), device=logits.device)
    H_cal = torch.zeros_like(H_noisy)

    for k in range(K):
        for i in range(M):
            mask = (attributes == i).float()
            H_noisy[i, k] = torch.sum(mask * prob[:, k]) / (torch.sum(mask) + EPS)

        if T is not None:
            H_cal[:, k] = torch.matmul(T[k], H_noisy[:, k].view(-1, 1)).view(-1)
        else:
            H_cal[:, k] = H_noisy[:, k]

    H_cal_clip = torch.clamp(H_cal, 1e-3, 1)
    H_cal_clip_final = H_cal_clip / H_cal_clip.sum(dim=1, keepdim=True)

    for i in range(M):
        for j in range(i + 1, M):
            for k in range(K):
                constraint.append(H_cal_clip_final[i, k] - H_cal_clip_final[j, k])

    constraint = torch.stack(constraint) / (M * (M - 1) * K * 1.0)
    return constraint


def constraints_eop(logits, attributes, labels):
    """PyTorch implementation of the Equal Opportunity (EOP) constraint."""
    prob = F.softmax(logits, dim=1)

    group_a1 = torch.sum((attributes == 0) * prob[:, 1] * (labels == 1)) / (
        torch.sum((labels == 1) * (attributes == 0).float()) + EPS
    )
    group_b1 = torch.sum((attributes == 1) * prob[:, 1] * (labels == 1)) / (
        torch.sum((labels == 1) * (attributes == 1).float()) + EPS
    )

    return group_a1 - group_b1


def constraints_eod(logits, attributes, labels):
    """PyTorch implementation of the Equalized Odds (EOD) constraint."""
    prob = F.softmax(logits, dim=1)

    group_a1 = torch.sum((attributes == 0) * prob[:, 1] * (labels == 1)) / (
        torch.sum((labels == 1) * (attributes == 0).float()) + EPS
    )
    group_b1 = torch.sum((attributes == 1) * prob[:, 1] * (labels == 1)) / (
        torch.sum((labels == 1) * (attributes == 1).float()) + EPS
    )

    group_a0 = torch.sum((attributes == 0) * prob[:, 1] * (labels == 0)) / (
        torch.sum((labels == 0) * (attributes == 0).float()) + EPS
    )
    group_b0 = torch.sum((attributes == 1) * prob[:, 1] * (labels == 0)) / (
        torch.sum((labels == 0) * (attributes == 1).float()) + EPS
    )

    return torch.stack([group_a0 - group_b0, group_a1 - group_b1])


from torch.func import functional_call, grad, vmap  # noqa: E402


class fis(erm):
    def __init__(self, args):
        super().__init__(args)
        self.unlabeled_train_loader = None
        self.constraints = None
        if args.fis_metric == "dp":
            self.constraints = constraints_dp
        if args.fis_metric == "eop":
            self.constraints = constraints_eop
        if args.fis_metric == "eod":
            self.constraints = constraints_eod
        self.mu = 1.0

    def validate(self, val_loader, epoch, args):
        model = self.model
        device = args.device
        model.eval()
        val_loss = AverageMeter()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        param_names = []
        flag = True
        grad_sum, grad_fair_sum = None, None
        num_samples = 0.0

        for _idx, (data, target, sensitive_attr) in enumerate(val_loader):
            data, target, sensitive_attr = (
                data.to(device),
                target.to(device),
                sensitive_attr.to(device),
            )
            if epoch < args.fis_warm:
                with torch.no_grad():
                    output = model(data)
                    if self.num_classes == 1:  # BCE Loss
                        target = target.float()
                        output = output.squeeze()
                        prob = F.sigmoid(output).flatten()
                    else:
                        prob = F.softmax(output, dim=-1)
                    loss = self.criterion(output, target)
                    try:
                        val_loss.update(loss.item())
                    except Exception:
                        val_loss.update(loss.mean().item())

                    tol_output += prob.cpu().data.numpy().tolist()
                    tol_target += target.cpu().data.numpy().tolist()
                    tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
            else:
                model.zero_grad()
                output = model(data)
                if self.num_classes == 1:  # BCE Loss
                    target = target.float()
                    output = output.squeeze()
                    prob = F.sigmoid(output).flatten()
                else:
                    prob = F.softmax(output, dim=-1)
                loss = self.criterion(output, target)
                loss.backward(retain_graph=True)

                grad_list = []

                for name, p in model.named_parameters():
                    if p.grad is not None:  # Ensure the parameter has a gradient.
                        grad_list.append(
                            p.grad.view(-1)
                        )  # Flatten the gradient and append it to grad_list.
                        if flag:
                            param_names.append(
                                name
                            )  # Save the corresponding parameter name.

                if flag:
                    param_names = set(param_names)
                flag = False
                grads = torch.cat(grad_list).detach()

                try:
                    val_loss.update(loss.item())
                except Exception:
                    val_loss.update(loss.mean().item())

                tol_output += prob.cpu().data.numpy().tolist()
                tol_target += target.cpu().data.numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()

                model.zero_grad()
                loss_reg = self.constraints(output, sensitive_attr, target)
                fair_loss = torch.sum((self.mu / 2) * loss_reg**2)
                fair_loss.backward()
                grads_fair = torch.cat(
                    [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
                ).detach()

                if grad_sum is None:
                    grad_sum = grads.clone()
                    grad_fair_sum = grads_fair.clone()
                else:
                    grad_sum += grads
                    grad_fair_sum += grads_fair

                num_samples += data.size(0)

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

        def compute_loss(params, buffers, sample, target):
            batch = sample.unsqueeze(0)
            targets = target.unsqueeze(0)

            predictions = functional_call(model, (params, buffers), (batch,))
            loss = F.cross_entropy(predictions, targets)
            return loss

        ft_compute_grad = grad(compute_loss)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

        if epoch >= args.fis_warm and len(self.unlabeled_train_loader) > 0:
            grad_avg = grad_sum / num_samples
            grad_fair = grad_fair_sum / num_samples
            scores, fair_scores, indices = [], [], []

            if not args.no_progress:
                p_bar = tqdm(range(len(self.unlabeled_train_loader)))
            for u_idx, (data, target, _sensitive_attr, index) in enumerate(
                self.unlabeled_train_loader
            ):  # Ensure the DataLoader returns indices here.
                data, target = data.to(device), target.to(device)

                model.zero_grad()

                params = {k: v.detach() for k, v in model.named_parameters()}
                buffers = {k: v.detach() for k, v in model.named_buffers()}

                ft_per_sample_grads = ft_compute_sample_grad(
                    params, buffers, data, target
                )

                batch_size = data.shape[0]  # Get the batch size.

                # Concatenate per-parameter gradients to build the per-sample gradient matrix.
                grads = torch.cat(
                    [
                        g.view(batch_size, -1)
                        for name, g in ft_per_sample_grads.items()
                        if name in param_names
                    ],
                    dim=-1,
                )

                infl = -torch.matmul(grads, grad_avg)
                infl_fair = -torch.matmul(
                    grads, grad_fair
                )  # Compute fairness influence.
                score_tmp = infl_fair.clone()
                score_tmp[infl_fair > 0] = 0.0
                score_tmp[infl > 0] = 0.0
                scores.extend(score_tmp.tolist())

                fair_scores.extend(infl_fair.tolist())
                indices.extend(index.tolist())
                if len(scores) >= num_samples * 100:
                    break

                if not args.no_progress:
                    p_bar.set_description(
                        f"Unlabel Epoch: {epoch + 1}/{args.epochs:4}. Iter: {u_idx + 1:4}/{len(self.unlabeled_train_loader):4}. Loss: {0:.4f}.  "
                    )
                    p_bar.update()
            if not args.no_progress:
                p_bar.close()

            num_samples = int(num_samples)
            scores = np.array(scores)
            sel_idx = np.argsort(scores)[:num_samples]
            max_score = scores[sel_idx[-1]]
            if max_score >= 0.0:
                sel_idx = np.arange(len(scores))[scores < 0.0]
            indices = np.array(indices)
            selected_indices = indices[sel_idx].tolist()

            log_dict["selected_indices"] = selected_indices

        return val_loss.avg, log_dict
