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
from test import test
from utils.BER import *
import matplotlib.pyplot as plt

# speed up
cudnn.benchmark = True


# ---------------------- Train function ----------------------
def train(train_dataloader, model, device, save_dir_path, args):
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

    # init gloabel parameters ----------------------------------------
    EPS = 1e-8

    # +++++++++++++++++++++++++++++++++start++++++++++++++++++++++++++++++++++++++++
    E_loss_list = []
    ber_1_list =[]
    for step in range(args.start_steps, args.num_train_steps):

        model.train()

        # init parameters ----------------------------------------
        batch_size = args.batch_size
        latent_dim = model.G.latent_dim
        num_layers = model.G.num_layers

        # train E************************************
        model.E_opt.zero_grad()
        E_accumulate_loss = 0
        

        # w--------------------------------
        get_latents_fn = mixed_list if random() < args.mixed_prob else noise_list
        style = get_latents_fn(batch_size, num_layers, latent_dim)
        w_space = latent_to_w(model.S, style)
        w_styles = styles_def_to_tensor(w_space)

        # noise--------------------------------
        noise = custom_image_nosie(batch_size, 100)
        noise_styles = latent_to_nosie(model.N, noise)

        generated_images = model.G(w_styles, noise_styles)
        decode_msg = model.E(generated_images.clone().detach())
        # loss----------------------
        divergence = nn.MSELoss()(decode_msg, noise)
        E_loss = divergence*100
        E_accumulate_loss += E_loss.clone().detach()

        E_loss.backward()

        model.E_opt.step()
        model.E_opt_scheduler.step()
        
        E_loss_list.append(E_accumulate_loss/args.gradient_accumulate_every)

        # BER {1,2,3}------------------------------------------
        BER_1 = compute_BER(decode_msg.detach(), noise, sigma=1)
        BER_2 = compute_BER(decode_msg.detach(), noise, sigma=2)
        BER_3 = compute_BER(decode_msg.detach(), noise, sigma=3)
        ber_1_list.append(BER_1)

        # logging-----------------------------------------------------------------
        if step % 10 == 0:
            logger.info('step {}/{}'.format(step + 1, args.num_train_steps))
            logger.info('e_loss: {:.4f}'.format(E_loss.clone().detach()))
            logger.info('BER_1:{} BER_2:{} BER_3:{}'.format(BER_1, BER_2, BER_3))
            logger.info('-' * 10)

        # Testing / Validating-----------------------------------
        if (step + 1) % args.test_every == 0 or step + 1 == args.num_train_steps:
            torch.cuda.empty_cache()
            logger.info('step {}/{}'.format(step + 1, args.num_train_steps))
            logger.info('-------------------test--------------------')
            test(model, save_dir_path, args, num=step)

        # for kaggle time to stop (14400 sce about 4 hours)-----------
        if time.time() - start_time > 14400:
            break

        # plot----------------------------------------------------------------
        if step % 500 == 0:
            plt.figure(figsize=(5, 4), dpi=80)
            plt.subplot(1, 1, 1)
            plt.plot(E_loss_list, label='e_loss ', marker='^', color='black', linewidth=1)
            plt.savefig(f'{save_dir_path}/plot_cuver_{step}.png')

            plt.figure(figsize=(5, 4), dpi=80)
            plt.subplot(1, 1, 1)
            plt.plot(ber_1_list, label='ber ', marker='^', color='black', linewidth=1)
            plt.savefig(f'{save_dir_path}/plot_cuver_ber_{step}.png')

    # stop time -------------------------------------------------------------
    time_elapsed = time.time() - start_time
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # Save final model weights----------------------------------------------
    checkpointNet.save_network(model, save_dir_path, '_finetune_final')
