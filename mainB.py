
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
from dataloader import getDataLoader
from models import build_model
from trainB import train
from utils import checkpointNet


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
parser.add_argument('--batch_size', default=100, type=int, help='batch_size')

"""
Model parameters
"""
parser.add_argument('--experiment', type=str, default='ExtractNet')
parser.add_argument('--image_size', default=64)
parser.add_argument('--gradient_accumulate_every', default=5)
parser.add_argument('--mixed_prob', default=0.9)

parser.add_argument('--which_epoch', default='final', type=str, help='0,1,2,3...or final')
parser.add_argument('--checkpoint_GAN', type=str, default='/home/hy/vscode/StyleGANStege/experiments/Celeba')
parser.add_argument('--checkpoint_E', type=str, default='/home/hy/vscode/StyleGANStege/experiments/ExtractNet')

"""
Train parameters
"""
parser.add_argument('--num_train_steps', type=int, default=25000)
parser.add_argument('--test_every', type=int, default=10000)

"""
Optimizer parameters
"""
parser.add_argument('--lr', type=float, default=2e-4)

args = parser.parse_args()


if __name__ == "__main__":
    # devie---------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix random seed---------------------------------------------------------------------------
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    # model------------------------------------------------------------------------------------
    stylegan = build_model('StyleGAN2', image_size=args.image_size, lr=args.lr)
    stylegan = checkpointNet.load_part_network(stylegan, args.checkpoint_GAN, args.which_epoch)
    stylegan = stylegan.to(device)

    extractNet = build_model('ExtractNet', image_size=args.image_size, lr=args.lr)
    extractNet = checkpointNet.load_part_network(extractNet, args.checkpoint_E, args.which_epoch)
    extractNet = extractNet.to(device)

    # criterion-----------------------------------------------------------------------------------
    criterion = nn.MSELoss()

    # optimizer-----------------------------------------------------------------------------------
    for p in stylegan.parameters():
        p.requires_grad = True
    param_groups = [{'params': stylegan.NE.parameters(), 'lr': args.lr}]
    optimizer = torch.optim.Adam(param_groups, lr=args.lr, betas=(0.5, 0.999))

    # scheduler-----------------------------------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=8, verbose=True, threshold=0.0001)

    # save_dir_path-----------------------------------------------------------------------------------
    save_dir_path = os.path.join(args.save_path, args.experiment)
    os.makedirs(save_dir_path, exist_ok=True)

    # train -----------------------------------------------------------------------------------
    train(stylegan, extractNet, criterion, optimizer, scheduler, device, save_dir_path, args)
