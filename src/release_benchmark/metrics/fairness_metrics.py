"""Fairness metrics for the NH-Fair benchmark. Adapted from MedFair: https://github.com/ys-zong/MEDFAIR

Computes accuracy, AUC, DP, EqOpp, and EqOdd for binary and multi-class
classification with binary or multi-group sensitive attributes.
"""

import numpy as np
import sklearn.metrics as sklm
from sklearn.metrics import roc_auc_score

from .evaluation import (
    conditional_errors_binary,
    conditional_errors_multi,
    find_threshold,
)

# ---------------------------------------------------------------------------
# AUC helpers
# ---------------------------------------------------------------------------


def calculate_auc(prediction, labels):
    fpr, tpr, _ = sklm.roc_curve(labels, prediction, pos_label=1)
    return sklm.auc(fpr, tpr)


def calculate_multiclass_auc(predictions, labels):
    if predictions.shape[1] == 2:
        predictions = predictions[:, 1]
    return roc_auc_score(labels, predictions, average="macro", multi_class="ovr")


def conditional_AUC_binary(preds, labels, attrs, num_class):
    preds, labels, attrs = np.asarray(preds), np.asarray(labels), np.asarray(attrs)
    idx = attrs == 0
    auc_fn = calculate_auc if num_class == 1 else calculate_multiclass_auc
    return auc_fn(preds[idx], labels[idx]), auc_fn(preds[~idx], labels[~idx])


def conditional_AUC_multi(preds, labels, attrs, sens_classes, num_class):
    auc_fn = calculate_auc if num_class == 1 else calculate_multiclass_auc
    return [auc_fn(preds[attrs == i], labels[attrs == i]) for i in range(sens_classes)]


# ---------------------------------------------------------------------------
# Equalized-odds helper (binary sensitive attr)
# ---------------------------------------------------------------------------


def cal_eqodd_binary(tol_predicted, labels, attrs):
    sens_idx = attrs == 0
    target_idx = labels == 0
    cond_00 = np.mean(tol_predicted[np.logical_and(sens_idx, target_idx)])
    cond_10 = np.mean(tol_predicted[np.logical_and(~sens_idx, target_idx)])
    cond_01 = np.mean(tol_predicted[np.logical_and(sens_idx, ~target_idx)])
    cond_11 = np.mean(tol_predicted[np.logical_and(~sens_idx, ~target_idx)])
    return 1 - 0.5 * (np.abs(cond_00 - cond_10) + np.abs(cond_01 - cond_11))


# ---------------------------------------------------------------------------
# Main metric dispatcher
# ---------------------------------------------------------------------------


def calculate_metrics(
    tol_output,
    tol_target,
    tol_sensitive,
    tol_index,
    sens_classes,
    num_class=1,
    skip_auc=False,
):
    """Compute all fairness and performance metrics.

    Returns (log_dict, predictions, aucs_per_group).
    """
    correct = 0
    if num_class == 1:
        threshold = find_threshold(tol_output, tol_target)
        tol_predicted = (tol_output > threshold).astype("float")
        correct += (tol_predicted == tol_target).sum()
    else:
        tol_output = np.asarray(tol_output)
        tol_predicted = np.argmax(tol_output, 1, keepdims=False)
        correct += (tol_predicted == tol_target).sum()

    acc = 100 * correct / len(tol_target)

    if skip_auc:
        auc = None
    elif num_class == 1:
        auc = calculate_auc(tol_output, tol_target)
    else:
        auc = calculate_multiclass_auc(tol_output, tol_target)

    tol_predicted = np.asarray(tol_predicted)
    tol_output = np.asarray(tol_output)
    tol_target = np.asarray(tol_target).squeeze()
    tol_sensitive = np.asarray(tol_sensitive)

    # ----- Binary sensitive attr + binary/binary-equiv target -----
    if sens_classes == 2 and num_class <= 2:
        sens_idx = tol_sensitive == 0
        target_idx = tol_target == 0

        cls_error, error_0, error_1 = conditional_errors_binary(
            tol_predicted, tol_target, tol_sensitive
        )
        pred_0 = np.mean(tol_predicted[sens_idx])
        pred_1 = np.mean(tol_predicted[~sens_idx])

        cond_00 = np.mean(tol_predicted[np.logical_and(sens_idx, target_idx)])
        cond_10 = np.mean(tol_predicted[np.logical_and(~sens_idx, target_idx)])
        cond_01 = np.mean(tol_predicted[np.logical_and(sens_idx, ~target_idx)])
        cond_11 = np.mean(tol_predicted[np.logical_and(~sens_idx, ~target_idx)])

        eqodd_thresh = cal_eqodd_binary(tol_predicted, tol_target, tol_sensitive)
        log_dict = {
            "Overall Acc": 1 - cls_error,
            "acc-group_0": 1 - error_0,
            "acc-group_1": 1 - error_1,
            "worst_acc": min(1 - error_0, 1 - error_1),
            "gap_acc": abs((1 - error_0) - (1 - error_1)),
            "DP": 1 - np.abs(pred_0 - pred_1),
            "EqOpp1": 1 - np.abs(cond_00 - cond_10),
            "EqOpp0": 1 - np.abs(cond_01 - cond_11),
            "EqOdd": 1 - 0.5 * (np.abs(cond_00 - cond_10) + np.abs(cond_01 - cond_11)),
            "EqOdd_0.5": eqodd_thresh,
        }
        if skip_auc:
            aucs = [0, 0]
        else:
            auc0, auc1 = conditional_AUC_binary(
                tol_output, tol_target, tol_sensitive, num_class
            )
            log_dict["Overall AUC"] = auc
            log_dict["auc-group_0"] = auc0
            log_dict["auc-group_1"] = auc1
            log_dict["worst_AUC"] = min(auc0, auc1)
            log_dict["gap_AUC"] = abs(auc0 - auc1)
            aucs = [auc0, auc1]

    # ----- Binary sensitive attr + multi-class target -----
    elif sens_classes == 2 and num_class > 2:
        log_dict = {"Overall Acc": acc}
        total_dp, total_eqopp, total_eqodd = 0.0, 0.0, 0.0
        total_correct = 0
        group_correct = {0: 0, 1: 0}
        group_total = {0: 0, 1: 0}

        for c in range(num_class):
            sens_idx = tol_sensitive == 0
            non_sens_idx = ~sens_idx
            target_idx = tol_target == c

            pred_c_sens = np.mean(tol_predicted[sens_idx] == c)
            pred_c_non_sens = np.mean(tol_predicted[non_sens_idx] == c)
            dp_c = 1 - abs(pred_c_sens - pred_c_non_sens)

            tpr_c_sens = np.mean(
                tol_predicted[np.logical_and(sens_idx, target_idx)] == c
            )
            tpr_c_non_sens = np.mean(
                tol_predicted[np.logical_and(non_sens_idx, target_idx)] == c
            )
            eqopp_c = 1 - abs(tpr_c_sens - tpr_c_non_sens)

            fpr_c_sens = np.mean(
                tol_predicted[np.logical_and(sens_idx, ~target_idx)] == c
            )
            fpr_c_non_sens = np.mean(
                tol_predicted[np.logical_and(non_sens_idx, ~target_idx)] == c
            )
            eqodd_c = 1 - 0.5 * (
                abs(tpr_c_sens - tpr_c_non_sens) + abs(fpr_c_sens - fpr_c_non_sens)
            )

            log_dict[f"DP_class_{c}"] = dp_c
            log_dict[f"EqOpp_class_{c}"] = eqopp_c
            log_dict[f"EqOdd_class_{c}"] = eqodd_c
            total_dp += dp_c
            total_eqopp += eqopp_c
            total_eqodd += eqodd_c

        log_dict["DP"] = total_dp / num_class
        log_dict["EqOppAvg"] = total_eqopp / num_class
        log_dict["EqOdd"] = total_eqodd / num_class

        for i in range(len(tol_target)):
            group = tol_sensitive[i]
            group_total[group] += 1
            if tol_predicted[i] == tol_target[i]:
                group_correct[group] += 1
                total_correct += 1

        group_acc = {g: group_correct[g] / group_total[g] for g in group_total}
        log_dict["Total_Accuracy"] = total_correct / len(tol_target)
        for g, a in group_acc.items():
            log_dict[f"Accuracy_group_{g}"] = a
        log_dict["worst_acc"] = min(group_acc.values())
        log_dict["gap_acc"] = max(group_acc.values()) - min(group_acc.values())
        aucs = [0, 0]

    # ----- Multi-group sensitive attr + binary target -----
    elif sens_classes > 2 and num_class <= 2:
        cls_error, errors = conditional_errors_multi(
            tol_predicted, tol_target, tol_sensitive, sens_classes
        )

        log_dict = {"Overall Acc": 1 - cls_error}
        if not skip_auc:
            log_dict["Overall AUC"] = auc
        group_accs = []

        if skip_auc:
            aucs = [0] * sens_classes
            for i, error in enumerate(errors):
                ga = 1 - error
                log_dict[f"acc-group_{i}"] = ga
                group_accs.append(ga)
        else:
            aucs = conditional_AUC_multi(
                tol_output, tol_target, tol_sensitive, sens_classes, num_class
            )
            for i, (error, auc_i) in enumerate(zip(errors, aucs)):
                ga = 1 - error
                log_dict[f"acc-group_{i}"] = ga
                log_dict[f"auc-group_{i}"] = auc_i
                group_accs.append(ga)

        log_dict["worst_acc"] = min(group_accs)
        log_dict["gap_acc"] = max(group_accs) - min(group_accs)
        if not skip_auc:
            log_dict["worst_AUC"] = min(aucs)
            log_dict["gap_AUC"] = max(aucs) - min(aucs)

        # DP
        pred_rates = [
            np.mean(tol_predicted[tol_sensitive == i]) for i in range(sens_classes)
        ]
        log_dict["DP"] = 1 - (max(pred_rates) - min(pred_rates))

        # EqOpp
        if num_class == 1:
            tprs = []
            for i in range(sens_classes):
                combined = np.logical_and(tol_sensitive == i, tol_target == 1)
                tprs.append(
                    np.mean(tol_predicted[combined]) if np.sum(combined) > 0 else 0.0
                )
            eqopp_diff = max(tprs) - min(tprs) if tprs else 0
            log_dict["EqOpp1"] = 1 - eqopp_diff
            log_dict["EqOppAvg"] = 1 - eqopp_diff
        else:
            log_dict["EqOppAvg"] = log_dict["DP"]

        # EqOdd
        if num_class == 1:
            fprs = []
            for i in range(sens_classes):
                combined = np.logical_and(tol_sensitive == i, tol_target == 0)
                fprs.append(
                    np.mean(tol_predicted[combined]) if np.sum(combined) > 0 else 0.0
                )
            tpr_diff = max(tprs) - min(tprs) if tprs else 0
            fpr_diff = max(fprs) - min(fprs) if fprs else 0
            log_dict["EqOdd"] = 1 - 0.5 * (tpr_diff + fpr_diff)
        else:
            log_dict["EqOdd"] = log_dict["DP"]

    else:
        raise ValueError(
            f"Unsupported combination: sens_classes={sens_classes}, num_class={num_class}"
        )
    log_dict = {k: float(v) for k, v in log_dict.items()}
    return log_dict, tol_predicted, aucs
