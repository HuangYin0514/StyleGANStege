
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


# ---------------------- Extract features ----------------------
def extract_feature(model, inputs, requires_norm, vectorize, requires_grad=False):

    # Move to model's device
    inputs = inputs.to(next(model.parameters()).device)

    with torch.set_grad_enabled(requires_grad):
        features = model(inputs)

    size = features.shape

    if requires_norm:
        # [N, C*H]
        features = features.view(size[0], -1)

        # norm feature
        fnorm = features.norm(p=2, dim=1)
        features = features.div(fnorm.unsqueeze(dim=1))

    if vectorize:
        features = features.view(size[0], -1)
    else:
        # Back to [N, C, H=S]
        features = features.view(size)

    return features


# ---------------------- Start testing ----------------------
def test(model, dataset, dataset_path, batch_size, max_rank=100):
    model.eval()
    # test dataloader------------------------------------------------------------
    gallery_dataloader = getDataLoader(
        dataset, batch_size, dataset_path, 'gallery', shuffle=False, augment=False)
    query_dataloader = getDataLoader(
        dataset, batch_size, dataset_path, 'query', shuffle=False, augment=False)

    # image information------------------------------------------------------------
    gallery_cams, gallery_pids = [], []
    query_cams, query_pids = [], []
    gallery_features = []
    query_features = []

    # gallery_dataloader ------------------------------------------------------------
    for inputs, pids, camids in gallery_dataloader:
        gallery_features.append(extract_feature(
            model, inputs, requires_norm=True, vectorize=True).cpu().data)
        gallery_pids.extend(np.array(pids))
        gallery_cams.extend(np.array(camids))
    gallery_features = torch.cat(gallery_features, dim=0)
    gallery_pids = np.asarray(gallery_pids)
    gallery_cams = np.asarray(gallery_cams)

    # query_dataloader ------------------------------------------------------------
    for inputs, pids, camids in query_dataloader:
        query_features.append(extract_feature(
            model, inputs, requires_norm=True, vectorize=True).cpu().data)
        query_pids.extend(np.array(pids))
        query_cams.extend(np.array(camids))
    query_features = torch.cat(query_features, dim=0)
    query_pids = np.asarray(query_pids)
    query_cams = np.asarray(query_cams)

    # compute cmc and map ------------------------------------------------------------
    distmat = compute_distance_matrix(
        query_features, gallery_features, metric='cosine')
    distmat = distmat.numpy()

    print('Computing CMC and mAP ...')
    cmc, mAP = evaluate_rank(
        distmat,
        query_pids,
        gallery_pids,
        query_cams,
        gallery_cams,
    )

    return cmc, mAP


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing arguments')
    parser.add_argument('--experiment', type=str, default='PCB_p6')
    parser.add_argument('--save_path', type=str, default='./experiments')
    parser.add_argument('--which_epoch', default='final', type=str, help='0,1,2,3...or final')
    parser.add_argument('--dataset', type=str, default='Market1501')
    parser.add_argument('--dataset_path', type=str, default='/home/hy/vscode/reid-custom/data/Market-1501-v15.09.15')
    parser.add_argument('--checkpoint', type=str, default='/home/hy/vscode/reid-custom/experiments/Market1501')
    parser.add_argument('--batch_size', default=512, type=int, help='batchsize')
    parser.add_argument('--share_conv', default=False, action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make saving directory
    save_dir_path = os.path.join(args.save_path, args.dataset)
    os.makedirs(save_dir_path, exist_ok=True)

    logger = util.Logger(save_dir_path)
    logger.info(vars(args))

    model = build_model(args.experiment, num_classes=1, hare_conv=args.share_conv)
    model = util.load_network(model, args.checkpoint, args.which_epoch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    CMC, mAP = test(model, args.dataset, args.dataset_path, args.batch_size)

    logger.info('Testing: top1:%.2f top5:%.2f top10:%.2f mAP:%.2f' % (CMC[0], CMC[4], CMC[9], mAP))

    # torch.cuda.empty_cache()
