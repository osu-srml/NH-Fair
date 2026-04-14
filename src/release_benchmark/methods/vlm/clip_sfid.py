import clip as openclip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from release_benchmark.methods.vlm.vlm_utils import set_matching_prompt
from release_benchmark.metrics.fairness_metrics import calculate_metrics
from release_benchmark.utils.common import AverageMeter


class clip_sfid(nn.Module):
    def __init__(self, args):

        super().__init__()
        self.model = self.setmodel(args)
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = args.num_classes
        self.text_tokens = self.set_prompts(args)
        self.img_clf = RandomForestClassifier(n_estimators=100)
        self._clf_fitted = False
        self.img_important_indices = None
        self.img_mean_features_lowconfidence = None
        self.optimizer = None
        self.fe_scheduler = None

    def train(self, train_loader, epoch, args):
        """Fit the RF once on epoch 0; CLIP weights stay frozen (SFID is inference-time)."""
        if epoch == 0:
            self.fit(train_loader, args)
        else:
            raise RuntimeError(
                "SFID classifier already fitted. Call method.fit(train_loader, args) only once. Set epoch to 1"
            )
        return 0.0

    def set_prompts(self, args):
        texts = set_matching_prompt(args)
        return openclip.tokenize(texts).to(args.device)

    def setmodel(self, args):

        if "resnet" in args.model:
            model, _preprocess = openclip.load("RN50", device=args.device)
        elif "vitb16" in args.model:
            model, _preprocess = openclip.load("ViT-B/16", device=args.device)
        else:
            raise NotImplementedError
        return model

    def fit(self, train_loader, args):
        """Fit the RandomForest classifier on training data and precompute
        the important feature indices. The low-confidence mean is deferred
        to the first validate() call (calibration set), matching the reference."""
        self.model.eval()
        device = args.device
        embeddings = []
        sensitive_attrs = []
        with torch.no_grad():
            for data, _target, sensitive_attr in tqdm(
                train_loader, desc="SFID: extracting train features"
            ):
                data = data.to(device)
                image_features = self.model.encode_image(data)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                embeddings.append(image_features.cpu())
                sensitive_attrs.append(sensitive_attr.cpu())

        embeddings = torch.cat(embeddings)
        sensitive_attrs = torch.cat(sensitive_attrs)
        self.img_clf.fit(embeddings, sensitive_attrs)
        self._clf_fitted = True

        importances = self.img_clf.feature_importances_
        self.img_important_indices = torch.tensor(
            np.argsort(importances)[-args.sfid_image_prune_num :]
        ).to(device)

        print("SFID: RandomForest classifier fitted on training data.")

    def _calibrate(self, embeddings, args):
        """Compute the low-confidence mean from a held-out set (val embeddings).
        Called once during the first validate() call, matching the reference
        which uses a separate calibration split."""
        probabilities = self.img_clf.predict_proba(embeddings.cpu())
        max_probabilities = probabilities.max(axis=1)
        low_confidence_samples = embeddings[max_probabilities < args.sfid_threshold]
        self.img_mean_features_lowconfidence = torch.mean(
            low_confidence_samples.float(), dim=0
        ).to(args.device)
        print(
            f"SFID: calibrated on {len(embeddings)} samples "
            f"({len(low_confidence_samples)} low-confidence, threshold={args.sfid_threshold})."
        )

    def validate(self, val_loader, epoch, args):
        model = self.model
        model.eval()
        val_loss = AverageMeter()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        if epoch == -1:
            with torch.no_grad():
                for data, target, sensitive_attr in tqdm(val_loader):
                    data, target = data.to(args.device), target.to(args.device)
                    image_features = model.encode_image(data)
                    text_features = model.encode_text(self.text_tokens)
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )
                    text_features = text_features / text_features.norm(
                        dim=-1, keepdim=True
                    )

                    similarity = image_features @ text_features.T  # [batch_size, 2]
                    torch.argmax(
                        similarity, dim=1
                    )  # Return the predicted class for each image (0 or 1).
                    output = similarity.float()
                    if self.num_classes == 1:  # BCE Loss
                        target = target.float()
                        output = output.squeeze()
                        prob = torch.sigmoid(output).flatten()
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

        if not self._clf_fitted:
            raise RuntimeError(
                "SFID classifier not fitted. Call method.fit(train_loader, args) before validate/test."
            )

        Y_val = []
        embeddings = []
        with torch.no_grad():
            text_features = model.encode_text(self.text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            for data, target, sensitive_attr in tqdm(val_loader):
                data, target = data.to(args.device), target.to(args.device)
                image_features = model.encode_image(data)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                embeddings.append(image_features.cpu())
                Y_val.append(target.cpu())
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()

        embeddings = torch.cat(embeddings).float()
        Y_val = torch.cat(Y_val)

        if self.img_mean_features_lowconfidence is None:
            self._calibrate(embeddings, args)

        embeddings = embeddings.to(args.device)
        embeddings[:, self.img_important_indices] = (
            self.img_mean_features_lowconfidence[self.img_important_indices]
        )

        text_features = text_features.float()
        batch_size = args.bs
        similarity_all = []

        for i in range(0, embeddings.size(0), batch_size):
            batch_embed = embeddings[i : i + batch_size]
            with torch.no_grad():
                batch_sim = batch_embed @ text_features.T
            similarity_all.append(batch_sim.cpu())

        similarity = torch.cat(similarity_all, dim=0)
        output = similarity
        prob = F.softmax(output, dim=-1)
        loss = self.criterion(output, Y_val.to(output.device))
        val_loss.update(loss.item())

        tol_output = prob.cpu().numpy().tolist()
        tol_target = Y_val.cpu().numpy().tolist()
        tol_index = []  # Optional, if used

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
        model.eval()
        test_loss = AverageMeter()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []

        if not self._clf_fitted:
            raise RuntimeError(
                "SFID classifier not fitted. Call method.fit(train_loader, args) before validate/test."
            )

        Y_test = []
        embeddings = []

        with torch.no_grad():
            text_features = model.encode_text(self.text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            for data, target, sensitive_attr in test_loader:
                data, target = data.to(args.device), target.to(args.device)
                image_features = model.encode_image(data)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                embeddings.append(image_features.cpu())
                Y_test.append(target.cpu())
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()

        embeddings = torch.cat(embeddings).float().to(args.device)
        Y_test = torch.cat(Y_test)
        embeddings[:, self.img_important_indices] = (
            self.img_mean_features_lowconfidence[self.img_important_indices]
        )

        text_features = text_features.float()
        batch_size = args.bs
        similarity_all = []

        for i in range(0, embeddings.size(0), batch_size):
            batch_embed = embeddings[i : i + batch_size]
            with torch.no_grad():
                batch_sim = batch_embed @ text_features.T
            similarity_all.append(batch_sim.cpu())

        similarity = torch.cat(similarity_all, dim=0)
        output = similarity
        prob = F.softmax(output, dim=-1)
        loss = self.criterion(output, Y_test.to(output.device))
        test_loss.update(loss.item())

        tol_output = prob.cpu().numpy().tolist()
        tol_target = Y_test.cpu().numpy().tolist()
        tol_index = []

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
