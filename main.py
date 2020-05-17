
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from dataloader import getDataLoader
from models import build_model
from train import *

# -----------------------------parameters setting --------------------------------
data = '../../gan/custom_dataset',
results_dir = './GoodResult/results',
models_dir = './GoodResult/models',
log_dir = './GoodResult/logs',
name = 'mytest',
new = False,
load_from = 10,
network_capacity = 16,
transparent = False,
batch_size = 3,
gradient_accumulate_every = 5,
num_train_steps = 120000,
learning_rate = 2e-4,
num_workers = None,
save_every = 10000,
generate = False,
num_image_tiles = 8,
trunc_psi = 0.6

parser = argparse.ArgumentParser(description='stegan stylegan')

"""
System parameters
"""
parser.add_argument('--nThread', type=int, default=4, help='number of threads for data loading')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs')
parser.add_argument('--save_path', type=str, default='./experiments')

"""
Data parameters
"""
parser.add_argument('--dataset', type=str, default='Celeba')
parser.add_argument('--dataset_path', type=str, default='/home/hy/vscode/reid-custom/data/Market-1501-v15.09.15')
parser.add_argument('--batch_size', default=3, type=int, help='batch_size')

"""
Model parameters
"""
parser.add_argument('--experiment', type=str, default='StyleGAN2')
parser.add_argument('--image_size', default=64)
parser.add_argument('--gradient_accumulate_every', default=5)
parser.add_argument('--mixed_prob', default=0.9)

"""
Train parameters
"""
parser.add_argument('--num_train_steps', type=int, default=3)
parser.add_argument('--test_every', type=int, default=1)

"""
Optimizer parameters
"""
parser.add_argument('--lr', type=float, default=0.1)

args = parser.parse_args()


if __name__ == "__main__":
    # devie---------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix random seed---------------------------------------------------------------------------
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    # dataset------------------------------------------------------------------------------------
    train_dataloader = getDataLoader(args.dataset, args.batch_size, args.dataset_path)

    # model------------------------------------------------------------------------------------
    model = build_model(args.experiment, image_size=args.image_size, lr=args.lr)
    model = model.to(device)

    # save_dir_path-----------------------------------------------------------------------------------
    save_dir_path = os.path.join(args.save_path, args.dataset)
    os.makedirs(save_dir_path, exist_ok=True)

    # train -----------------------------------------------------------------------------------
    train(train_dataloader, model, device, save_dir_path, args)
