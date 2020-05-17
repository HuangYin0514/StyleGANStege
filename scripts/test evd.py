
# %%
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from sklearn.metrics import average_precision_score

from utils import util
from dataloader import getDataLoader
from models import build_model
from metrics.distance import compute_distance_matrix
from metrics.rank import evaluate_rank


# ---------------------- Evaluation ----------------------
def evaluate(query_features, query_labels, query_cams, gallery_features, gallery_labels, gallery_cams):
    """Evaluate the CMC and mAP

    Arguments:
        query_features {np.ndarray of size NxC} -- Features of probe images
        query_labels {np.ndarray of query size N} -- Labels of probe images
        query_cams {np.ndarray of query size N} -- Cameras of probe images
        gallery_features {np.ndarray of size N'xC} -- Features of gallery images
        gallery_labels {np.ndarray of gallery size N'} -- Lables of gallery images
        gallery_cams {np.ndarray of gallery size N'} -- Cameras of gallery images

    Returns:
        (torch.IntTensor, float) -- CMC list, mAP
    """

    CMC = torch.IntTensor(len(gallery_labels)).zero_()
    AP = 0
    sorted_index_list, sorted_y_true_list, junk_index_list = [], [], []

    for i in range(len(query_labels)):
        query_feature = query_features[i]
        query_label = query_labels[i]
        query_cam = query_cams[i]

        # Prediction score
        score = np.dot(gallery_features, query_feature)

        match_query_index = np.argwhere(gallery_labels == query_label)
        same_camera_index = np.argwhere(gallery_cams == query_cam)

        # Positive index is the matched indexs at different camera i.e. the desired result
        positive_index = np.setdiff1d(
            match_query_index, same_camera_index, assume_unique=True)

        # Junk index is the indexs at the same camera or the unlabeled image
        junk_index = np.append(
            np.argwhere(gallery_labels == -1),
            np.intersect1d(match_query_index, same_camera_index))  # .flatten()

        index = np.arange(len(gallery_labels))
        # Remove all the junk indexs
        sufficient_index = np.setdiff1d(index, junk_index)

        # compute AP
        y_true = np.in1d(sufficient_index, positive_index)
        y_score = score[sufficient_index]
        if not np.any(y_true):
            # this condition is true when query identity does not appear in gallery
            continue
        AP += average_precision_score(y_true, y_score)

        # Compute CMC
        # Sort the sufficient index by their scores, from large to small
        sorted_index = np.argsort(y_score)[::-1]
        sorted_y_true = y_true[sorted_index]
        match_index = np.argwhere(sorted_y_true == True)

        if match_index.size > 0:
            first_match_index = match_index.flatten()[0]
            CMC[first_match_index:] += 1

        # keep with junk index, for using the index to show the img from dataloader
        all_sorted_index = np.argsort(score)[::-1]
        all_y_true = np.in1d(index, match_query_index)
        all_sorted_y_true = all_y_true[all_sorted_index]

        sorted_index_list.append(all_sorted_index)
        sorted_y_true_list.append(all_sorted_y_true)
        junk_index_list.append(junk_index)

    CMC = CMC.float()
    CMC = CMC / len(query_labels) * 100  # average CMC
    mAP = AP / len(query_labels) * 100

    return CMC, mAP, (sorted_index_list, sorted_y_true_list, junk_index_list)


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


# %%
qf = torch.Tensor(np.array([[1, 2, 3], [2, 4, 5], [1, 5, 5]]))
gf = torch.Tensor(np.array([[1, 2, 3], [2, 4, 5], [1, 5, 5],[1, 2, 3]]))

q_pids = np.array([0, 1, 2])
g_pids = np.array([0, 1, 2,1])

q_cams = np.array([1, 2, 3])
g_cams = np.array([7777, 12, 555, 4])

# norm feature
fnorm = qf.norm(p=2, dim=1)
features_q = qf.div(fnorm.unsqueeze(dim=1))
fnorm = gf.norm(p=2, dim=1)
features_g = gf.div(fnorm.unsqueeze(dim=1))
print(evaluate(features_q, q_pids, q_cams, features_g, g_pids, g_cams))

distmat = compute_distance_matrix(
    features_q, features_g, metric='cosine')
distmat = distmat.numpy()
print()
print(eval_market1501(
    distmat,
    q_pids,
    g_pids,
    q_cams,
    g_cams,
    max_rank=5
))
# %%
