import numpy as np
import torch
import torch.nn.functional as F

from release_benchmark.metrics.fairness_metrics import calculate_metrics

from .erm import erm

# Import oxonfair
try:
    import oxonfair
    from oxonfair.utils import group_metrics as gm

    OXONFAIR_AVAILABLE = True
except ImportError:
    OXONFAIR_AVAILABLE = False
    print(
        "Warning: oxonfair package not available. Please install with: pip install oxonfair"
    )


class oxonfair_method(erm):
    def __init__(self, args):
        if not OXONFAIR_AVAILABLE:
            raise ImportError(
                "oxonfair package is required but not installed. Please install with: pip install oxonfair"
            )
        super().__init__(args)
        self.model.eval()

    def get_model_outputs(self, loader, args):
        """Get model outputs (logits) and convert to probabilities"""
        self.model.eval()
        device = args.device
        all_outputs, all_targets, all_sens = [], [], []

        with torch.no_grad():
            for data, target, sens in loader:
                data = data.to(device)
                logits = self.model(data)  # Get raw logits
                # print(logits[0:2])

                # Convert logits to probabilities using softmax
                probs = F.softmax(logits, dim=1)

                all_outputs.append(probs.cpu().numpy())
                all_targets.append(target.numpy())
                all_sens.append(sens.numpy())

        outputs = np.concatenate(all_outputs, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        sens = np.concatenate(all_sens, axis=0)

        return outputs, targets, sens

    def prepare_oxonfair_data(self, outputs, targets, sens):
        if outputs.shape[1] == 2:
            Y_pred = outputs
        else:
            raise ValueError(f"Unsupported number of classes: {outputs.shape}")

        Y_true = targets.astype(int)
        S_true = sens.astype(int)

        return Y_pred, Y_true, S_true

    def evaluate_oxonfair_predictor(
        self, predictor, test_outputs, test_targets, test_sens
    ):
        """Evaluate the oxonfair predictor"""

        # Prepare test data
        test_pred, test_true, test_group = self.prepare_oxonfair_data(
            test_outputs, test_targets, test_sens
        )
        # test_pred = test_pred.reshape(-1, 1)
        test_dict = {"data": test_pred, "target": test_true, "groups": test_group}

        # Get updated predictions from oxonfair
        print(predictor.evaluate_groups(test_dict))
        corrected_test_pred = predictor.predict_proba(test_dict)

        return corrected_test_pred

    def train(self, loaders, epoch, args):
        """Main training method for oxonfair post-processing"""
        assert (
            epoch == 0
        )  # no need to run multiple rounds for post-processing. Set args.epochs = 1 for oxonfair_method since it is a post-processing method

        model = self.model
        _train_loader, val_loader, test_loader = loaders

        if args.ckpt_path == "none":
            raise ValueError(
                "oxonfair_method requires --ckpt_path to point to a pretrained ERM checkpoint."
            )

        print(f"Loading checkpoint from: {args.ckpt_path}")
        model.load_state_dict(
            torch.load(args.ckpt_path, map_location=args.device)["model"]
        )
        model.eval()

        self.validate(val_loader, epoch, args)
        self.test(test_loader, epoch, args)

        # Get model outputs (probabilities) from all splits
        print("Getting model outputs...")
        val_outputs, val_targets, val_sens = self.get_model_outputs(val_loader, args)
        test_outputs, test_targets, test_sens = self.get_model_outputs(
            test_loader, args
        )

        print(f"Output shapes -  Val: {val_outputs.shape}, Test: {test_outputs.shape}")
        print(
            f"Label distributions -  Val: {np.bincount(val_targets)}, Test: {np.bincount(test_targets)}"
        )
        print(
            f"Sensitive attribute distributions - Val: {np.bincount(val_sens)}, Test: {np.bincount(test_sens)}"
        )

        # Train oxonfair predictor
        print("Training oxonfair predictor...")

        val_pred, val_true, val_group = self.prepare_oxonfair_data(
            val_outputs, val_targets, val_sens
        )
        predictor = oxonfair.FairPredictor(
            predictor=None,
            validation_data={"data": val_pred, "target": val_true, "groups": val_group},
            groups=val_group,
        )
        mode = getattr(self.args, "oxonfair_mode", "accuracy_noharm")
        if mode == "accuracy_noharm":
            predictor.fit(
                gm.accuracy,
                gm.accuracy.diff,
                greater_is_better_obj=True,
                greater_is_better_const=False,
            )
        elif mode == "minmax_fairness":
            predictor.fit(
                gm.accuracy.min,
                gm.accuracy,
                greater_is_better_obj=True,
                greater_is_better_const=True,
            )
        elif mode == "diff_min":
            predictor.fit(
                gm.accuracy.diff,
                gm.accuracy.min,
                greater_is_better_obj=False,
                greater_is_better_const=True,
            )
        elif mode == "dp_acc":
            predictor.fit(gm.demographic_parity, gm.accuracy)
        elif mode == "eqodd_acc":
            predictor.fit(gm.equalized_odds, gm.accuracy)
        elif mode == "accuracy_noharm_balanced":
            predictor.fit(
                gm.balanced_accuracy,
                gm.balanced_accuracy.diff,
                greater_is_better_obj=True,
                greater_is_better_const=False,
            )
        elif mode == "minmax_fairness_balanced":
            predictor.fit(
                gm.balanced_accuracy.min,
                gm.balanced_accuracy,
                greater_is_better_obj=True,
                greater_is_better_const=True,
            )
        elif mode == "diff_min_balanced":
            predictor.fit(
                gm.balanced_accuracy.diff,
                gm.balanced_accuracy.min,
                greater_is_better_obj=False,
                greater_is_better_const=True,
            )
        elif mode == "dp_acc_balanced":
            predictor.fit(gm.demographic_parity, gm.balanced_accuracy)
        elif mode == "eqodd_acc_balanced":
            predictor.fit(gm.equalized_odds, gm.balanced_accuracy)
        else:
            raise ValueError(f"Invalid oxonfair mode: {mode}")

        predictor.groups = None

        # Evaluate on test set
        print("Evaluating on test set...")
        val_predictions = self.evaluate_oxonfair_predictor(
            predictor, val_outputs, val_targets, val_sens
        )
        test_predictions = self.evaluate_oxonfair_predictor(
            predictor, test_outputs, test_targets, test_sens
        )

        # test_predictions = test_predictions.reshape(-1, 1)
        # # Calculate metrics using the existing evaluation framework
        val_log_dict, _val_t_predictions, _val_aucs_subgroup = calculate_metrics(
            val_predictions,
            val_targets,
            val_sens,
            None,
            self.args.sensitive_attributes,
            num_class=self.args.num_classes,
        )
        test_log_dict, _test_t_predictions, _test_aucs_subgroup = calculate_metrics(
            test_predictions,
            test_targets,
            test_sens,
            None,
            self.args.sensitive_attributes,
            num_class=self.args.num_classes,
        )

        print(
            "#####################################test#######################################"
        )
        print(test_log_dict, "\n")

        # Set dummy losses (not used in post-processing)
        train_loss = -1
        val_loss = -1
        test_loss = -1

        return train_loss, val_loss, val_log_dict, test_loss, test_log_dict
