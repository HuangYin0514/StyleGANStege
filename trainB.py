import argparse
import os
import time
from random import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from pathlib import Path

from utils import checkpointNet, util_logger
from utils.util import *
from utils.BER import *

from test import test

# speed up
cudnn.benchmark = True


def sample_StyleGAN_input_data(stylegan, args):
    batch_size = args.batch_size
    latent_dim = stylegan.G.latent_dim
    num_layers = stylegan.G.num_layers
    # w
    get_latents_fn = mixed_list if random() < args.mixed_prob else noise_list
    style = get_latents_fn(batch_size, num_layers, latent_dim)
    w_space = latent_to_w(stylegan.S, style)
    w_styles = styles_def_to_tensor(w_space)
    # noise
    noise = custom_image_nosie(batch_size, 100)
    noise_styles = latent_to_nosie(stylegan.N, noise)
    secret = noise
    return w_styles, noise_styles, secret


def plt_ber_curve(ber1_list, ber2_list, ber3_list, save_dir_path):
    plt.figure(figsize=(5, 4), dpi=80)
    plt.subplot(1, 1, 1)
    plt.plot(ber1_list, label='sample_acc1 ', marker='^', color='black', linewidth=1)
    plt.plot(ber2_list, label='sample_acc2 ', marker='x', color='black', linewidth=1)
    plt.plot(ber3_list, label='sample_acc3 ', marker='o', color='black', linewidth=1)
    # plt.xlim((-1, 12))
    plt.ylim((0, 0.5))
    plt.xticks(fontsize=10.5)
    plt.yticks(fontsize=10.5)
    plt.legend()
    savepath = 'experiments/BerCuver/ber_curve.jpg'
    plt.savefig(savepath)


# ---------------------- Train function ----------------------
def train(stylegan, criterion, optimizer, scheduler, device, save_dir_path, args):
    '''
        train
    '''
    # start_time -----------------------------------------------
    start_time = time.time()

    # Logger instance--------------------------------------------
    logger = util_logger.Logger(save_dir_path)
    logger.info('-' * 10)
    logger.info(vars(args))

    # globe parameter--------------------------------------------
    num_layers = stylegan.GE.num_layers
    latent_dim = stylegan.GE.latent_dim
    get_latents_fn = mixed_list if random() < args.mixed_prob else noise_list
    style = get_latents_fn(args.batch_size, num_layers, latent_dim)
    noise = custom_image_nosie(args.batch_size, 100)
    secret = noise

    BER_1_list, BER_2_list, BER_3_list = [], [], []

    # train----------------------------------------------
    for step in range(args.num_train_steps):
        # info ----------------------------------------------------
        # stylegan.train()

        # noise-----------------------------
        noise_styles = latent_to_nosie(stylegan.NE, noise)
        
        # w-----------------------------
        w_space = latent_to_w(stylegan.SE, style)
        w_styles = styles_def_to_tensor(w_space)

        # loss-----------------------------
        generated_images = stylegan.GE(w_styles, noise_styles)
        decode_msg = stylegan.E(generated_images)
        secret_loss = criterion(decode_msg, secret)
        divergence = args.batch_size * (30*secret_loss)
        E_loss = divergence

        # clear grad-----------------------------
        optimizer.zero_grad()
        E_loss.backward()
        optimizer.step()
        scheduler.step(E_loss.item())

        # BER {1,2,3}------------------------------------------
        BER_1 = compute_BER(decode_msg.detach().unsqueeze(0), secret, sigma=1)
        BER_2 = compute_BER(decode_msg.detach().unsqueeze(0), secret, sigma=2)
        BER_3 = compute_BER(decode_msg.detach().unsqueeze(0), secret, sigma=3)
        E_loss = float(divergence.detach().item())

        if step % 10 == 0:
            BER_1_list.append(BER_1)
            BER_2_list.append(BER_2)
            BER_3_list.append(BER_3)
            torchvision.utils.save_image(generated_images, str(Path(save_dir_path) / f'{str(step)}.jpg'))

        # logger ------------------------------------------
        if step % 10 == 0:
            logger.info('step {}/{}'.format(step + 1, args.num_train_steps))
            logger.info('E_loss:{}'.format(E_loss))
            logger.info('BER_1:{:.4f} BER_2:{} BER_3:{}'.format(BER_1, BER_2, BER_3))
            logger.info('-' * 10)

    plt_ber_curve(BER_1_list, BER_2_list, BER_3_list, save_dir_path)
    # stop time ---------------------------------------------------
    time_elapsed = time.time() - start_time
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return BER_1
    # Save final model weights-----------------------------------
    # checkpointNet.save_network(model, save_dir_path, 'final')
