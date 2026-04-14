import os

import clip as openclip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from tqdm import tqdm

from release_benchmark.methods.lib import kernels as kernels
from release_benchmark.methods.vlm.vlm_utils import set_matching_prompt
from release_benchmark.metrics.fairness_metrics import calculate_metrics
from release_benchmark.utils.common import AverageMeter


def mean_center(features, dim):
    """
    Return mean-centered features along a given dimension.
    """
    features = features.float()
    return features - torch.mean(features, dim=dim)


class KernelizedEncoderFull(nn.Module):
    def __init__(self, U, X, kernel):
        super().__init__()
        self.dtype = U.dtype
        self.X = X

        if isinstance(U, list):
            self.U = nn.Parameter(torch.zeros(*tuple(U)), requires_grad=False)
        else:
            self.U = nn.Parameter(U)

        self.kernel = kernel

    def forward(self, x):
        x = x.to(dtype=self.dtype)
        phi_x = self.kernel(x, self.X)
        z = torch.mm(phi_x, self.U)
        return z.float()


class KernelizedEncoder(nn.Module):
    def __init__(self, U, w, b):
        super().__init__()
        self.dtype = U.dtype

        if isinstance(U, list):
            self.U = Parameter(torch.zeros(*tuple(U)), requires_grad=False)
        else:
            self.U = Parameter(U)

        if isinstance(w, list):
            self.w = Parameter(torch.zeros(*tuple(w)), requires_grad=False)
        else:
            self.w = Parameter(w)

        if isinstance(b, list):
            self.b = Parameter(torch.zeros(*tuple(b)), requires_grad=False)
        else:
            self.b = Parameter(b)

    def forward(self, x):
        x = x.to(dtype=self.dtype)
        phi_x = torch.sqrt(
            torch.tensor([2.0 / len(self.w)], device=x.device)
        ) * torch.cos(
            torch.mm(x, self.w.t().to(dtype=self.dtype)) + self.b.to(dtype=self.dtype)
        )
        z = torch.mm(phi_x, self.U)
        return z.float()


class KernelMethodY:
    def __init__(self, opts, modal):
        self.hparams = opts
        if modal == "image":
            self.lambda_z = opts.clipfairer_tau_z_i
            self.lambda_s = opts.clipfairer_tau_i
            self.gamma = opts.clipfairer_gamma_i
        else:
            self.lambda_z = opts.clipfairer_tau_z_t
            self.lambda_s = opts.clipfairer_tau_t
            self.gamma = opts.clipfairer_gamma_t

        self.kernel_x = kernels.RFFGaussian(
            rff_dim=opts.clipfairer_rff_dim, sigma_numel_max=opts.clipfairer_sigma_max
        )
        self.kernel_y = kernels.RFFLinear()
        self.kernel_s = kernels.RFFLinear()
        self.kernel_z = kernels.RFFGaussian(
            rff_dim=opts.clipfairer_rff_dim, sigma_numel_max=opts.clipfairer_sigma_max
        )

    def solver(self, X, Y, S, Z=None):
        """
        Z = theta_D * R_xD
        """

        device = X.device
        dtype = torch.float

        s = S.to(dtype)

        n = len(X)
        rff_flag = getattr(self.hparams, "clipfairer_rff_flag", True)
        dim_z = getattr(self.hparams, "clipfairer_dim_z", 1)

        if rff_flag:
            R_x = self.kernel_x(X).to(dtype=dtype, device=device)
            R_x_c = mean_center(R_x, dim=0).to(dtype=dtype, device=device)
            dtype = R_x.dtype

            R_y = self.kernel_y(Y).to(dtype=dtype, device=device)
            R_y_c = mean_center(R_y, dim=0).to(dtype=dtype, device=device)

            R_s = self.kernel_s(s).to(dtype=dtype, device=device)
            R_s_c = mean_center(R_s, dim=0).to(dtype=dtype, device=device)

            if Z is not None:
                Z_k = self.kernel_z(Z).to(dtype=dtype, device=device)
                Z_k_c = mean_center(Z_k, dim=0).to(dtype=dtype, device=device)

            b_y = torch.mm(R_x.t(), R_y_c)
            b_y = torch.mm(b_y, b_y.t())

            if Z is not None:
                b_z = torch.mm(R_x.t(), Z_k_c)
                b_z = torch.mm(b_z, b_z.t())

            b_s = torch.mm(R_x.t(), R_s_c)
            b_s = torch.mm(b_s, b_s.t())

            norm2_b_y = torch.linalg.norm(b_y, 2)
            norm2_b_s = torch.linalg.norm(b_s, 2)

            if Z is not None:
                norm2_b_z = torch.linalg.norm(b_z, 2)
                b = (
                    b_y / norm2_b_y
                    + self.lambda_z / (1.0 - self.lambda_z) * b_z / norm2_b_z
                    - self.lambda_s / (1.0 - self.lambda_s) * b_s / norm2_b_s
                )
                self.norm2_b_z = norm2_b_z
            else:
                b = (
                    b_y / norm2_b_y
                    - self.lambda_s / (1.0 - self.lambda_s) * b_s / norm2_b_s
                )

            self.norm2_b_y = norm2_b_y
            self.norm2_b_s = norm2_b_s

            b = (b + b.t()) / 2

            c = torch.mm(R_x_c.t(), R_x_c) + n * self.gamma * torch.eye(
                R_x.shape[1], device=R_x.device
            )

            c = (c + c.t()) / 2

            eigs, V = torch.linalg.eig(torch.mm(torch.linalg.inv(c), b))
            eigs = torch.real(eigs)
            V = torch.real(V)

            sorted_eigs, indices = torch.sort(eigs, descending=True)

            U = V[:, indices[0:dim_z]]

            #########################################
            r0 = dim_z
            if self.lambda_s == 0:
                r = r0
            else:
                r1 = min((sorted_eigs > 0).sum(), r0)

                ###### Energy Thresholding ######
                if r1 > 0:
                    for k in range(1, r1 + 1):
                        if (
                            torch.linalg.norm(sorted_eigs[0:k]) ** 2
                            / torch.linalg.norm(sorted_eigs[0:r1]) ** 2
                            >= 0.95
                        ):
                            r = k
                            break
                else:
                    r = 0
            ######################################################
            if self.lambda_s >= 0.999999999:
                r = 0
            ######################################################
            U[:, r:dim_z] = 0

            encoder = KernelizedEncoder(
                U=n**0.5 * U, w=self.kernel_x.w, b=self.kernel_x.b
            )

            if Z is not None:
                Z_enc = encoder(X)
                if ((Z_enc / Z) < 0).sum() / Z.numel() > 0.5:
                    U *= -1
                    encoder = KernelizedEncoder(
                        U=n**0.5 * U, w=self.kernel_x.w, b=self.kernel_x.b
                    )

        else:
            R_x = self.kernel_x(X).to(dtype=dtype, device=device)
            R_x_c = mean_center(R_x, dim=0).to(dtype=dtype, device=device)
            dtype = R_x.dtype

            R_y = self.kernel_y(Y).to(dtype=dtype, device=device)
            R_y_c = mean_center(R_y, dim=0).to(dtype=dtype, device=device)
            R_y_c = mean_center(R_y_c.t(), dim=0).to(dtype=dtype, device=device)

            R_s = self.kernel_s(s).to(dtype=dtype, device=device)
            R_s_c = mean_center(R_s, dim=0).to(dtype=dtype, device=device)
            R_s_c = mean_center(R_s_c.t(), dim=0).to(dtype=dtype, device=device)

            if Z is not None:
                Z_k = self.kernel_z(Z).to(dtype=dtype, device=device)
                Z_k_c = mean_center(Z_k, dim=0).to(dtype=dtype, device=device)
                Z_k_c = mean_center(Z_k_c.t(), dim=0).to(dtype=dtype, device=device)

            b_y = torch.mm(torch.mm(R_x.t(), R_y_c), R_x)

            if Z is not None:
                b_z = torch.mm(torch.mm(R_x.t(), Z_k_c), R_x)

            b_s = torch.mm(torch.mm(R_x.t(), R_s_c), R_x)

            norm2_b_y = torch.linalg.norm(b_y, 2)
            norm2_b_s = torch.linalg.norm(b_s, 2)

            if Z is not None:
                norm2_b_z = torch.linalg.norm(b_z, 2)
                b = (
                    b_y / norm2_b_y
                    + self.lambda_z / (1.0 - self.lambda_z) * b_z / norm2_b_z
                    - self.lambda_s / (1.0 - self.lambda_s) * b_s / norm2_b_s
                )
                self.norm2_b_z = norm2_b_z
            else:
                b = (
                    b_y / norm2_b_y
                    - self.lambda_s / (1.0 - self.lambda_s) * b_s / norm2_b_s
                )

            self.norm2_b_y = norm2_b_y
            self.norm2_b_s = norm2_b_s

            b = (b + b.t()) / 2

            c = torch.mm(R_x_c.t(), R_x_c) + n * self.gamma * torch.eye(
                R_x.shape[1], device=R_x.device
            )

            c = (c + c.t()) / 2

            eigs, V = torch.linalg.eig(torch.mm(torch.linalg.inv(c), b))
            eigs = torch.real(eigs)
            V = torch.real(V)

            sorted_eigs, indices = torch.sort(eigs, descending=True)

            U = V[:, indices[0:dim_z]]

            #########################################
            r0 = dim_z
            if self.lambda_s == 0:
                r = r0
            else:
                r1 = min((sorted_eigs > 0).sum(), r0)

                ###### Energy Thresholding ######
                if r1 > 0:
                    for k in range(1, r1 + 1):
                        if (
                            torch.linalg.norm(sorted_eigs[0:k]) ** 2
                            / torch.linalg.norm(sorted_eigs[0:r1]) ** 2
                            >= 0.95
                        ):
                            r = k
                            break
                else:
                    r = 0
            ######################################################
            if self.lambda_s >= 0.999999999:
                r = 0
            ######################################################
            U[:, r:dim_z] = 0

            encoder = KernelizedEncoderFull(U=n**0.5 * U, kernel=self.kernel_x, X=X)

        self.encoder = encoder

        return encoder

    def encod(self, X):
        return self.encoder(X)


class AlternatingOptimizer:
    def __init__(self, opts):
        self.image_model = KernelMethodY(opts, "image")
        self.text_model = KernelMethodY(opts, "text")

    def main(
        self,
        X_I,
        Y_I,
        S_I,
        Y_D,
        S_D,
        text_embeddings,
        num_iters,
        get_zeroshot_predictions,
    ):

        self.image_model.solver(X=X_I, Y=Y_I, S=S_I, Z=None)

        y_binary = ((Y_I + 1) / 2)[:, 1].int()
        X_T = text_embeddings[y_binary]

        for iter in range(num_iters):
            # Updating the pseudo-labels
            if iter > 0:
                debias_image_train, _debias_text_train = self.get_feat(X_I, X_T)
                text_embeddings_debias = self.get_textfeat(text_embeddings)
                dataset_predictions_train = get_zeroshot_predictions(
                    debias_image_train, text_embeddings_debias, temperature=100.0
                )
                Y_D = (
                    torch.nn.functional.one_hot(
                        torch.from_numpy(dataset_predictions_train.astype(int)),
                        num_classes=2,
                    )
                ) * 2 - 1
                Y_I = Y_D

                y_binary = ((Y_I + 1) / 2)[:, 1].int()
                X_T = text_embeddings[y_binary]

            Z_I = self.image_model.encod(X_I)

            self.text_model.solver(X=X_T, Y=Y_D, S=S_D, Z=Z_I)

            Z_D = self.text_model.encod(X_T)

            self.image_model.solver(X=X_I, Y=Y_I, S=S_I, Z=Z_D)

            print(f"Training {iter + 1}/{num_iters} done!")

    def get_feat(self, X_I, X_D):

        Z_D = self.text_model.encod(X_D)
        Z_I = self.image_model.encod(X_I)

        return Z_I, Z_D

    def get_textfeat(self, X_D):

        Z_D = self.text_model.encod(X_D)

        return Z_D


def get_zeroshot_predictions(image_embeddings, text_embeddings, temperature=100.0):

    with torch.no_grad():
        _image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )

        _text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        cross = _image_embeddings @ _text_embeddings.T
        text_probs = (temperature * cross).softmax(dim=-1)
        _, predicted = torch.max(text_probs.data, 1)

    return predicted.cpu().numpy()


class clip_fairer(nn.Module):
    def __init__(self, args):

        super().__init__()
        self.model = self.setmodel(args)
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = args.num_classes
        self.text_tokens = self.set_prompts(args)
        self.optimizer, self.fe_scheduler = None, None
        self.model_debias = AlternatingOptimizer(args)

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

    def train(self, train_loader, epoch, args):
        self.model.eval()
        device = args.device

        save_dir = os.path.join(args.data_path, args.dataset)
        save_path = os.path.join(save_dir, f"features_{args.seed}.pt")

        if os.path.exists(save_path):
            print("Loading saved features...")
            saved_data = torch.load(save_path)
            embeddings = saved_data["embeddings"]
            Y_train_onehot = saved_data["Y_train_onehot"]
            sensitive_attrs_onehot = saved_data["sensitive_attrs_onehot"]
            text_features = saved_data["text_features"]
        else:
            print("Saving new features...")
            Y_train = []
            embeddings = []
            sensitive_attrs = []
            with torch.no_grad():
                text_features = self.model.encode_text(self.text_tokens).float()
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                for data, target, sensitive_attr in tqdm(train_loader):
                    data, target = data.to(device), target.to(device)
                    image_features = self.model.encode_image(data)
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )
                    embeddings.append(image_features.float())
                    Y_train.append(target)
                    sensitive_attrs.append(sensitive_attr)

            embeddings = torch.cat(embeddings)
            Y_train = torch.cat(Y_train)
            sensitive_attrs = torch.cat(sensitive_attrs)
            Y_train_onehot = F.one_hot(Y_train) * 2 - 1
            sensitive_attrs_onehot = F.one_hot(sensitive_attrs) * 2 - 1

            saved_data = {
                "embeddings": embeddings,
                "Y_train_onehot": Y_train_onehot,
                "sensitive_attrs_onehot": sensitive_attrs_onehot,
                "text_features": text_features,
            }
            os.makedirs(save_dir, exist_ok=True)
            print(f"Saving features to {save_path}")
            torch.save(saved_data, save_path)

        num_iters = getattr(args, "clipfairer_iters", 1)
        self.model_debias.main(
            embeddings,
            Y_train_onehot,
            sensitive_attrs_onehot,
            Y_train_onehot,
            sensitive_attrs_onehot,
            text_features,
            num_iters,
            get_zeroshot_predictions,
        )

        return 0.0

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

                    # print(text_features)
                    similarity = image_features @ text_features.T  # [batch_size, 2]
                    torch.argmax(
                        similarity, dim=1
                    )  # Return the predicted class for each image (0 or 1).
                    output = similarity.float()
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

        with torch.no_grad():
            text_features = model.encode_text(self.text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            for data, target, sensitive_attr in tqdm(val_loader):
                data, target = data.to(args.device), target.to(args.device)
                image_features = model.encode_image(data)

                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                image_features_debias, text_features_debias = (
                    self.model_debias.get_feat(image_features, text_features)
                )

                # print(text_features)
                similarity = (
                    image_features_debias @ text_features_debias.T
                )  # [batch_size, 2]
                torch.argmax(
                    similarity, dim=1
                )  # Return the predicted class for each image (0 or 1).
                output = similarity.float()

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

    def test(self, test_loader, epoch, args):
        model = self.model
        device = args.device
        model.eval()
        test_loss = AverageMeter()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        with torch.no_grad():
            text_features = model.encode_text(self.text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            for data, target, sensitive_attr in test_loader:
                data = data.to(device)
                target = target.to(device)
                image_features = model.encode_image(data)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                image_features_debias, text_features_debias = (
                    self.model_debias.get_feat(image_features, text_features)
                )

                similarity = image_features_debias @ text_features_debias.T
                output = similarity.float()

                if self.num_classes == 1:  # BCE Loss
                    target = target.float()
                    output = output.squeeze()
                    prob = F.sigmoid(output).flatten()
                else:
                    prob = F.softmax(output, dim=-1)
                loss = self.criterion(output, target)
                try:
                    test_loss.update(loss.item())
                except Exception:
                    test_loss.update(loss.mean().item())
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
