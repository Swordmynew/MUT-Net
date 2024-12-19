import torch
import numpy as np
from collections import OrderedDict
import pandas as pd
import os
from tqdm import tqdm
import cv2
from utils.misc import split_np_imgrid


def cal_ber(tn, tp, fn, fp):
    return 0.5 * (fp / (tn + fp) + fn / (fn + tp))


def cal_acc(tn, tp, fn, fp):
    return (tp + tn) / (tp + tn + fp + fn)


def get_binary_classification_metrics(pred, gt, threshold=None):
    if threshold is not None:
        gt = (gt > threshold)
        pred = (pred > threshold)
    TP = np.logical_and(gt, pred).sum()
    TN = np.logical_and(np.logical_not(gt), np.logical_not(pred)).sum()
    FN = np.logical_and(gt, np.logical_not(pred)).sum()
    FP = np.logical_and(np.logical_not(gt), pred).sum()
    BER = cal_ber(TN, TP, FN, FP)
    ACC = cal_acc(TN, TP, FN, FP)
    return OrderedDict([('TP', TP),
                        ('TN', TN),
                        ('FP', FP),
                        ('FN', FN),
                        ('BER', BER),
                        ('ACC', ACC)]
                       )

def evaluate(pred_root, gt_root, nimg, nrow):
    pred_img_names = os.listdir(pred_root)
    gt_img_names = os.listdir(gt_root)
    img_names = list(set(pred_img_names) & set(gt_img_names))
    score_dict = OrderedDict()

    for img_name in tqdm(img_names, disable=False):
        pred_im_path = os.path.join(pred_root, img_name)
        gt_im_path = os.path.join(gt_root, img_name)
        pred_im = cv2.imread(pred_im_path)
        gt_im = cv2.imread(gt_im_path)
        score_dict[img_name] = get_binary_classification_metrics(pred_im,
                                                                 gt_im,
                                                                 125)
    print(score_dict)
    df = pd.DataFrame(score_dict)
    df['ave'] = df.mean(axis=1)

    tn = df['ave']['TN']
    tp = df['ave']['TP']
    fn = df['ave']['FN']
    fp = df['ave']['FP']

    pos_err = (1 - tp / (tp + fn)) * 100
    neg_err = (1 - tn / (tn + fp)) * 100
    ber = (pos_err + neg_err) / 2
    acc = (tn + tp) / (tn + tp + fn + fp)

    return pos_err, neg_err, ber, acc, df


