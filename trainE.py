import argparse
import os
import time
from random import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

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


# ---------------------- Train function ----------------------
def train(model, stylegan, criterion, optimizer, scheduler, device, save_dir_path, args):
    '''
        train
    '''
    # start_time -----------------------------------------------
    start_time = time.time()

    # Logger instance--------------------------------------------
    logger = util_logger.Logger(save_dir_path)
    logger.info('-' * 10)
    logger.info(vars(args))
    # logger.info(model)

    # train----------------------------------------------
    for step in range(args.num_train_steps):
        # info ----------------------------------------------------
        model.train()

        # clear grad-----------------
        model.zero_grad()
        # prepare data -------------
        w_styles, noise_styles, secret = sample_StyleGAN_input_data(stylegan, args)
        generated_images = stylegan.G(w_styles.to(device), noise_styles)
        decode_msg = model(generated_images.clone().detach())
        # loss----------------------
        divergence = args.batch_size * criterion(decode_msg, secret)
        E_loss = divergence
        E_loss.register_hook(raise_if_nan)
        # update grad--------------
        E_loss.backward()
        optimizer.step()

        # BER {1,2,3}------------------------------------------
        BER_1 = compute_BER(decode_msg.detach(), secret, sigma=1)
        BER_2 = compute_BER(decode_msg.detach(), secret, sigma=2)
        BER_3 = compute_BER(decode_msg.detach(), secret, sigma=3)
        E_loss = float(divergence.detach().item())

        if step % 20 == 0:
            logger.info('step {}/{}'.format(step + 1, args.num_train_steps))
            logger.info('E_loss:{}'.format(E_loss))
            logger.info('BER_1:{} BER_2:{} BER_3:{}'.format(BER_1, BER_2, BER_3))
            logger.info('-' * 10)

    # stop time ---------------------------------------------------
    time_elapsed = time.time() - start_time
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # Save final model weights-----------------------------------
    checkpointNet.save_network(model, save_dir_path, 'final')
