import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from release_benchmark.metrics.fairness_metrics import calculate_metrics

from .erm import erm


class dfr(erm):
    def __init__(self, args):
        super().__init__(args)
        self.model.eval()

        self.C_OPTIONS = [3.0, 1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01, 0.003]
        CLASS_WEIGHT_OPTIONS = [1.0, 2.0, 3.0, 10.0, 100.0, 300.0, 1000.0]
        self.CLASS_WEIGHT_OPTIONS = [{0: 1, 1: w} for w in CLASS_WEIGHT_OPTIONS] + [
            {0: w, 1: 1} for w in CLASS_WEIGHT_OPTIONS
        ]

        self.REG = args.dfr_ref
        self.notrain_dfr_val = args.dfr_notrain_val
        self.mode = args.dfr_mode
        self.tune_class_weights_dfr_train = args.dfr_tune_class_weights_train

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _balance_groups(x, y, g):
        """Undersample all groups to the size of the smallest group."""
        n_groups = np.max(g) + 1
        g_idx = [np.where(g == gi)[0] for gi in range(n_groups)]
        min_g = min(len(gi) for gi in g_idx)
        for gi in g_idx:
            np.random.shuffle(gi)
        x = np.concatenate([x[gi[:min_g]] for gi in g_idx])
        y = np.concatenate([y[gi[:min_g]] for gi in g_idx])
        g = np.concatenate([g[gi[:min_g]] for gi in g_idx])
        return x, y, g, n_groups

    def _build_averaged_logreg(self, coefs, intercepts, x_train, y_train):
        """Create a LogisticRegression whose weights are the mean of *coefs* / *intercepts*."""
        logreg = LogisticRegression(penalty=self.REG, C=1.0, solver="liblinear")
        n_classes = np.max(y_train) + 1
        logreg.fit(x_train[:n_classes], np.arange(n_classes))
        logreg.coef_ = np.mean(coefs, axis=0)
        logreg.intercept_ = np.mean(intercepts, axis=0)
        return logreg

    def _compute_metrics(self, preds, y, g):
        return calculate_metrics(
            preds,
            y,
            g,
            None,
            self.args.sensitive_attributes,
            num_class=self.args.num_classes,
        )

    def _evaluate_logreg(
        self,
        logreg,
        scaler,
        preprocess,
        all_embeddings,
        all_y,
        all_g,
        x_train,
        y_train,
        g_train,
    ):
        """Run the averaged logreg on test and (last-iteration) train data."""
        x_test = all_embeddings["test"]
        y_test = all_y["test"]
        g_test = all_g["test"]

        if preprocess:
            x_test = scaler.transform(x_test)

        preds_test = logreg.predict_proba(x_test)
        preds_train = logreg.predict_proba(x_train)

        test_log_dict, test_preds, test_aucs = self._compute_metrics(
            preds_test,
            y_test,
            g_test,
        )
        train_log_dict, train_preds, train_aucs = self._compute_metrics(
            preds_train,
            y_train,
            g_train,
        )
        return (
            test_log_dict,
            test_preds,
            test_aucs,
            train_log_dict,
            train_preds,
            train_aucs,
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def get_features(self, loader, args):
        self.model.eval()
        device = args.device
        all_embs, all_targets, all_sens = [], [], []
        with torch.no_grad():
            for data, target, sens in loader:
                data = data.to(device)
                _, feats = self.model.forward_return_feature(data)
                all_embs.append(feats.cpu().numpy())
                all_targets.append(target.numpy())
                all_sens.append(sens.numpy())

        return (
            np.concatenate(all_embs),
            np.concatenate(all_targets),
            np.concatenate(all_sens),
        )

    # ------------------------------------------------------------------
    # DFR on validation
    # ------------------------------------------------------------------

    def dfr_on_validation_tune(
        self,
        all_embeddings,
        all_y,
        all_g,
        preprocess=True,
        balance_val=False,
        add_train=True,
        num_retrains=1,
    ):
        worst_accs = {}
        for i in range(num_retrains):
            x_val = all_embeddings["val"]
            y_val = all_y["val"]
            g_val = all_g["val"]

            n_val = len(x_val) // 2
            idx = np.arange(len(x_val))
            np.random.shuffle(idx)

            x_valtrain = x_val[idx[n_val:]]
            y_valtrain = y_val[idx[n_val:]]
            g_valtrain = g_val[idx[n_val:]]

            if balance_val:
                x_valtrain, y_valtrain, g_valtrain, n_groups = self._balance_groups(
                    x_valtrain, y_valtrain, g_valtrain
                )
            else:
                n_groups = np.max(g_valtrain) + 1

            x_val = x_val[idx[:n_val]]
            y_val = y_val[idx[:n_val]]
            g_val = g_val[idx[:n_val]]

            n_train = len(x_valtrain) if add_train else 0

            x_train = np.concatenate([all_embeddings["train"][:n_train], x_valtrain])
            y_train = np.concatenate([all_y["train"][:n_train], y_valtrain])
            g_train = np.concatenate([all_g["train"][:n_train], g_valtrain])
            print(np.bincount(g_train))
            if preprocess:
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_val = scaler.transform(x_val)

            cls_w_options = (
                [{0: 1.0, 1: 1.0}]
                if (balance_val and not add_train)
                else self.CLASS_WEIGHT_OPTIONS
            )
            for c in self.C_OPTIONS:
                for class_weight in cls_w_options:
                    logreg = LogisticRegression(
                        penalty=self.REG,
                        C=c,
                        solver="liblinear",
                        class_weight=class_weight,
                    )
                    logreg.fit(x_train, y_train)
                    preds_val = logreg.predict(x_val)
                    group_accs = np.array(
                        [
                            (preds_val == y_val)[g_val == g].mean()
                            for g in range(n_groups)
                        ]
                    )
                    worst_acc = np.min(group_accs)
                    key = (c, class_weight[0], class_weight[1])
                    if i == 0:
                        worst_accs[key] = worst_acc
                    else:
                        worst_accs[key] += worst_acc

        ks, vs = list(worst_accs.keys()), list(worst_accs.values())
        return ks[np.argmax(vs)]

    def dfr_on_validation_eval(
        self,
        c,
        w1,
        w2,
        all_embeddings,
        all_y,
        all_g,
        num_retrains=20,
        preprocess=True,
        balance_val=False,
        add_train=True,
    ):
        coefs, intercepts = [], []
        scaler = None
        if preprocess:
            scaler = StandardScaler()
            scaler.fit(all_embeddings["train"])

        for _i in range(num_retrains):
            x_val = all_embeddings["val"]
            y_val = all_y["val"]
            g_val = all_g["val"]

            if balance_val:
                x_val, y_val, g_val, _ = self._balance_groups(x_val, y_val, g_val)

            n_train = len(x_val) if add_train else 0
            train_idx = np.arange(len(all_embeddings["train"]))
            np.random.shuffle(train_idx)
            train_idx = train_idx[:n_train]

            x_train = np.concatenate([all_embeddings["train"][train_idx], x_val])
            y_train = np.concatenate([all_y["train"][train_idx], y_val])
            g_train = np.concatenate([all_g["train"][train_idx], g_val])
            print(np.bincount(g_train))
            if preprocess:
                x_train = scaler.transform(x_train)

            logreg = LogisticRegression(
                penalty=self.REG, C=c, solver="liblinear", class_weight={0: w1, 1: w2}
            )
            logreg.fit(x_train, y_train)
            coefs.append(logreg.coef_)
            intercepts.append(logreg.intercept_)

        logreg = self._build_averaged_logreg(coefs, intercepts, x_train, y_train)
        return self._evaluate_logreg(
            logreg,
            scaler,
            preprocess,
            all_embeddings,
            all_y,
            all_g,
            x_train,
            y_train,
            g_train,
        )

    # ------------------------------------------------------------------
    # DFR on train subset
    # ------------------------------------------------------------------

    def dfr_train_subset_tune(
        self, all_embeddings, all_y, all_g, preprocess=True, learn_class_weights=False
    ):
        x_val = all_embeddings["val"]
        y_val = all_y["val"]
        g_val = all_g["val"]

        x_train = all_embeddings["train"]
        y_train = all_y["train"]
        g_train = all_g["train"]

        scaler = None
        if preprocess:
            scaler = StandardScaler()
            scaler.fit(x_train)

        x_train, y_train, g_train, n_groups = self._balance_groups(
            x_train, y_train, g_train
        )

        if preprocess:
            x_train = scaler.transform(x_train)
            x_val = scaler.transform(x_val)

        worst_accs = {}
        cls_w_options = (
            self.CLASS_WEIGHT_OPTIONS if learn_class_weights else [{0: 1.0, 1: 1.0}]
        )
        for c in self.C_OPTIONS:
            for class_weight in cls_w_options:
                logreg = LogisticRegression(
                    penalty=self.REG,
                    C=c,
                    solver="liblinear",
                    class_weight=class_weight,
                    max_iter=20,
                )
                logreg.fit(x_train, y_train)
                preds_val = logreg.predict(x_val)
                group_accs = np.array(
                    [(preds_val == y_val)[g_val == g].mean() for g in range(n_groups)]
                )
                worst_acc = np.min(group_accs)
                worst_accs[c, class_weight[0], class_weight[1]] = worst_acc
                print(c, class_weight, worst_acc, group_accs)

        ks, vs = list(worst_accs.keys()), list(worst_accs.values())
        return ks[np.argmax(vs)]

    def dfr_train_subset_eval(
        self, c, w1, w2, all_embeddings, all_y, all_g, num_retrains=10, preprocess=True
    ):
        coefs, intercepts = [], []
        scaler = None
        if preprocess:
            scaler = StandardScaler()
            scaler.fit(all_embeddings["train"])

        for _i in range(num_retrains):
            x_train, y_train, g_train, _ = self._balance_groups(
                all_embeddings["train"], all_y["train"], all_g["train"]
            )

            if preprocess:
                x_train = scaler.transform(x_train)

            logreg = LogisticRegression(
                penalty=self.REG, C=c, solver="liblinear", class_weight={0: w1, 1: w2}
            )
            logreg.fit(x_train, y_train)
            coefs.append(logreg.coef_)
            intercepts.append(logreg.intercept_)

        logreg = self._build_averaged_logreg(coefs, intercepts, x_train, y_train)
        return self._evaluate_logreg(
            logreg,
            scaler,
            preprocess,
            all_embeddings,
            all_y,
            all_g,
            x_train,
            y_train,
            g_train,
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def train(self, loaders, epoch, args):
        assert epoch == 0, "DFR is a post-processing method; set args.epochs = 1"
        train_loader, val_loader, test_loader = loaders

        if args.ckpt_path == "none":
            raise ValueError(
                "DFR requires --ckpt_path to point to a pretrained ERM checkpoint."
            )

        print(f"Loading checkpoint from: {args.ckpt_path}")
        self.model.load_state_dict(
            torch.load(args.ckpt_path, map_location=args.device)["model"]
        )
        self.model.eval()

        X_train, Y_train, S_train = self.get_features(train_loader, args)
        X_val, Y_val, S_val = self.get_features(val_loader, args)
        X_test, Y_test, S_test = self.get_features(test_loader, args)

        all_embeddings = {"train": X_train, "val": X_val, "test": X_test}
        all_y = {"train": Y_train, "val": Y_val, "test": Y_test}
        all_g = {"train": S_train, "val": S_val, "test": S_test}

        if self.mode == "validation":
            c, w1, w2 = self.dfr_on_validation_tune(
                all_embeddings,
                all_y,
                all_g,
                balance_val=True,
                add_train=not self.notrain_dfr_val,
            )
            (
                test_log_dict,
                _,
                _,
                train_log_dict,
                _,
                _,
            ) = self.dfr_on_validation_eval(
                c,
                w1,
                w2,
                all_embeddings,
                all_y,
                all_g,
                balance_val=True,
                add_train=not self.notrain_dfr_val,
            )

        elif self.mode == "train":
            c, w1, w2 = self.dfr_train_subset_tune(
                all_embeddings,
                all_y,
                all_g,
                learn_class_weights=self.tune_class_weights_dfr_train,
            )
            (
                test_log_dict,
                _,
                _,
                train_log_dict,
                _,
                _,
            ) = self.dfr_train_subset_eval(c, w1, w2, all_embeddings, all_y, all_g)

        else:
            raise NotImplementedError(f"Unknown dfr_mode: {self.mode!r}")

        val_log_dict = train_log_dict

        print(
            f"#####################################validation {epoch}#######################################"
        )
        print(val_log_dict, "\n")

        print(
            "#####################################test#######################################"
        )
        print(test_log_dict, "\n")

        return -1, -1, val_log_dict, -1, test_log_dict
