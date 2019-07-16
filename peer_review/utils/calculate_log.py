# Original code is from https://github.com/ShiyuLiang/odin-pytorch/blob/master/code/calMetric.py
import numpy as np


def auroc(x, y):
    end = np.max([np.max(x), np.max(y)])
    start = np.min([np.min(x), np.min(y)])
    gap = (end - start) / 200000
    auroc_base = 0.
    fpr_temp = 1.

    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(x >= delta)) / np.float(len(x))
        fpr = np.sum(np.sum(y > delta)) / np.float(len(y))
        auroc_base += (fpr_temp - fpr) * tpr
        fpr_temp = fpr

    auroc_base += fpr * tpr
    return auroc_base


def aupr_in(x, y):
    end = np.max([np.max(x), np.max(y)])
    start = np.min([np.min(x), np.min(y)])
    gap = (end - start) / 200000
    aupr_base = 0.
    recall_temp = 1.

    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(x >= delta)) / np.float(len(x))
        fp = np.sum(np.sum(y >= delta)) / np.float(len(y))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp
        aupr_base += (recall_temp - recall) * precision
        recall_temp = recall

    aupr_base += recall * precision
    return aupr_base


def aupr_out(x, y):
    end = np.max([np.max(x), np.max(y)])
    start = np.min([np.min(x), np.min(y)])
    gap = (end - start) / 200000
    aupr_base = 0.
    recall_temp = 1.

    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(x < delta)) / np.float(len(x))
        tp = np.sum(np.sum(y < delta)) / np.float(len(y))
        if tp + fp == 0:
            break
        precision = tp / (tp + fp)
        recall = tp
        aupr_base += (recall_temp - recall) * precision
        recall_temp = recall

    aupr_base += recall * precision
    return aupr_base


def det_acc(x, y):
    end = np.max([np.max(x), np.max(y)])
    start = np.min([np.min(x), np.min(y)])
    gap = (end - start) / 200000
    error_base = 1.

    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(x < delta)) / np.float(len(x))
        error = np.sum(np.sum(y > delta)) / np.float(len(y))
        error_base = np.minimum(error_base, (tpr + error) / 2.0)

    return 1 - error_base


def tnr_at_tpr_95(x, y):
    end = np.max([np.max(x), np.max(y)])
    start = np.min([np.min(x), np.min(y)])
    gap = (end - start) / 200000
    fpr = 0.
    total = 0.

    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(x >= delta)) / np.float(len(x))
        error = np.sum(np.sum(y > delta)) / np.float(len(y))
        if 0.94 <= tpr <= 0.96:
            fpr += error
            total += 1

    if total == 0:  # Corner case
        fpr_base = 1
    else:
        fpr_base = fpr / total

    return 1 - fpr_base
