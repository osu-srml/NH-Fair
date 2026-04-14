import warnings

import numpy as np
import sklearn.metrics as sklm

warnings.filterwarnings("ignore")


# adapted from https://github.com/LalehSeyyed/Underdiagnosis_NatMed/blob/main/CXP/classification/predictions.py
# and https://github.com/MLforHealth/CXR_Fairness/blob/master/cxr_fairness/metrics.py
def find_threshold(tol_output, tol_target):
    # to find this thresold, first we get the precision and recall without this, from there we calculate f1 score,
    # using f1score, we found this theresold which has best precsision and recall.  Then this threshold activation
    # are used to calculate our binary output.

    p, r, t = sklm.precision_recall_curve(tol_target, tol_output)
    # Choose the best threshold based on the highest F1 measure
    f1 = np.multiply(2, np.divide(np.multiply(p, r), np.add(r, p)))
    bestthr = t[np.where(f1 == max(f1))]

    return bestthr[0]


def conditional_errors_binary(preds, labels, attrs):
    """
    Compute the conditional errors of A = 0/1. All the arguments need to be one-dimensional vectors.
    :param preds: The predicted label given by a model.
    :param labels: The groundtruth label.
    :param attrs: The label of sensitive attribute.
    :return: Overall classification error, error | A = 0, error | A = 1.
    """

    assert preds.shape == labels.shape and labels.shape == attrs.shape
    cls_error = 1 - np.mean((preds == labels).astype("float"))
    idx = attrs == 0
    error_0 = 1 - np.mean((preds[idx] == labels[idx]).astype("float"))
    error_1 = 1 - np.mean((preds[~idx] == labels[~idx]).astype("float"))
    return cls_error, error_0, error_1


def conditional_errors_multi(preds, labels, attrs, sens_classes):
    """
    Compute the conditional errors of A with multiple values (0, 1, 2, ...). All the arguments need to be one-dimensional vectors.
    :param preds: The predicted label given by a model.
    :param labels: The groundtruth label.
    :param attrs: The label of sensitive attribute.
    :return: Overall classification error, error | A = 1, 2, n.
    """
    assert preds.shape == labels.shape and labels.shape == attrs.shape
    cls_error = 1 - np.mean((preds == labels).astype("float"))

    errors = []
    for i in range(sens_classes):
        idx = attrs == i
        error = 1 - np.mean((preds[idx] == labels[idx]).astype("float"))
        errors.append(error.item())
    return cls_error, errors


def threshold_metric_fn(
    labels, pred_probs, sample_weight=None, threshold=0.5, metric_generator_fn=None
):
    """
    Function that generates threshold metric functions.
    Calls a metric_generator_fn for customization
    """
    if metric_generator_fn is None:
        raise ValueError("metric_generator_fn must not be None")

    metric_fn = metric_generator_fn(
        threshold=threshold,
    )
    if sample_weight is None:
        return metric_fn(pred_probs, labels)
    else:
        return metric_fn(pred_probs, labels, sample_weight=sample_weight)


def get_worst_auc(log_dict):
    auc_dict = {}
    for key in log_dict:
        if "auc-" in key:
            auc_dict[key] = log_dict[key]

    worst_auc = 1.0
    group = -1
    for key, value in auc_dict.items():
        if value <= worst_auc:
            worst_auc = value
            group = key
    log_dict["worst_auc"] = worst_auc
    log_dict["worst_group"] = group
    return log_dict
