import torch
import torch.nn.functional as F
from tqdm import tqdm

from release_benchmark.metrics.fairness_metrics import calculate_metrics
from release_benchmark.model import Decoupled_ResNet18
from release_benchmark.utils.common import AverageMeter

from .erm import erm


class decoupled(erm):
    def setmodel(self, args):
        if "resnet" in args.model:
            model = Decoupled_ResNet18(
                num_classes=args.num_classes,
                pretrain=args.pretrain,
                sensitive_attributes=2,
                model=args.model,
            )
        elif args.model == "vit":
            raise NotImplementedError
        model = model.to(args.device)
        return model

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
            output = model(data, sensitive_attr)
            self.optimizer.zero_grad()

            if self.num_classes == 1:
                target = target.float()
            output = output.squeeze()
            loss = self.criterion(output, target)
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
        val_loss = AverageMeter()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        with torch.no_grad():
            for data, target, sensitive_attr in val_loader:
                data, target, sensitive_attr = (
                    data.to(device),
                    target.to(device),
                    sensitive_attr.to(device),
                )
                output = model(data, sensitive_attr)
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
        test_loss = AverageMeter()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        with torch.no_grad():
            for data, target, sensitive_attr in test_loader:
                data, target, sensitive_attr = (
                    data.to(device),
                    target.to(device),
                    sensitive_attr.to(device),
                )
                output = model(data, sensitive_attr)

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
