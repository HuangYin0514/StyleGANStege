import argparse
import os
import time
from random import random

import torch
import torch.nn as nn

from utils import util_logger
from utils.util import *
import torch.nn.functional as F


# speed up
import torch.backends.cudnn as cudnn
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
    logger.info(model)

    # init gloabel parameters ----------------------------------------
    pl_mean = 0
    last_gp_loss = 0
    d_loss = 0
    g_loss = 0
    EPS = 1e-8

    # +++++++++++++++++++++++++++++++++start++++++++++++++++++++++++++++++++++++++++
    for step in range(args.num_train_steps):

        model.train()

        logger.info('step {}/{}'.format(step + 1, args.num_train_steps))

        # init parameters ----------------------------------------
        total_disc_loss = torch.tensor(0.).to(device)
        total_gen_loss = torch.tensor(0.).to(device)

        batch_size = args.batch_size
        image_size = model.G.image_size
        latent_dim = model.G.latent_dim
        num_layers = model.G.num_layers

        apply_gradient_penalty = step % 4 == 0
        apply_path_penalty = step % 32 == 0

        # train discriminator************************************
        avg_pl_length = pl_mean
        model.D_opt.zero_grad()
        for i in range(args.gradient_accumulate_every):
            # w--------------------------------
            get_latents_fn = mixed_list if random() < args.mixed_prob else noise_list
            style = get_latents_fn(batch_size, num_layers, latent_dim)
            w_space = latent_to_w(model.S, style)
            w_styles = styles_def_to_tensor(w_space)

            # noise--------------------------------
            noise = custom_image_nosie(batch_size, 100)
            noise_styles = latent_to_nosie(model.N, noise)

            # trian fake--------------------------------
            generated_images = model.G(w_styles, noise_styles)
            fake_output = model.D(generated_images.clone().detach())

            # trian real--------------------------------
            image_batch = next(train_dataloader).to(device)
            image_batch.requires_grad_()
            real_output = model.D(image_batch)

            # loss--------------------------------
            divergence = (F.relu(1 + real_output) + F.relu(1 - fake_output)).mean()
            disc_loss = divergence

            # auxilur loss--------------------------------
            if apply_gradient_penalty:
                gp = gradient_penalty(image_batch, real_output)
                last_gp_loss = gp.clone().detach().item()
                disc_loss = disc_loss + gp
            disc_loss = disc_loss / args.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            disc_loss.backward()

            # record total loss--------------------------------
            total_disc_loss += divergence.detach().item() / args.gradient_accumulate_every
        # d_loss = float(total_disc_loss)
        # model.D_opt.step()

        # # train generator************************************
        # model.G_opt.zero_grad()
        # for i in range(args.gradient_accumulate_every):
        #     # w--------------------------------
        #     style = get_latents_fn(batch_size, num_layers, latent_dim)
        #     w_space = latent_to_w(model.S, style)
        #     w_styles = styles_def_to_tensor(w_space)

        #     # noise--------------------------------
        #     noise = custom_image_nosie(batch_size, 100)
        #     noise_styles = latent_to_nosie(model.N, noise)

        #     # fake--------------------------------
        #     generated_images = model.G(w_styles, noise_styles)
        #     fake_output = model.D(generated_images)
        #     loss = fake_output.mean()
        #     gen_loss = loss

        #     # pl loss--------------------------------
        #     if apply_path_penalty:
        #         std = 0.1 / (w_styles.std(dim=0, keepdim=True) + EPS)
        #         w_styles_2 = w_styles + torch.randn(w_styles.shape).to(device) / (std + EPS)
        #         pl_images = model.G(w_styles_2, noise_styles)
        #         pl_lengths = ((pl_images - generated_images)**2).mean(dim=(1, 2, 3))
        #         avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

        #         if pl_mean is not None:
        #             pl_loss = ((pl_lengths - pl_mean)**2).mean()
        #             if not torch.isnan(pl_loss):
        #                 gen_loss = gen_loss + pl_loss

        #     gen_loss = gen_loss / args.gradient_accumulate_every
        #     gen_loss.register_hook(raise_if_nan)
        #     gen_loss.backward()

        #     # total loss--------------------------------
        #     total_gen_loss += loss.detach().item() / args.gradient_accumulate_every
        # g_loss = float(total_gen_loss)
        # model.G_opt.step()


        # #calculate moving averages ---------------------------------------
        # if apply_path_penalty and not np.isnan(avg_pl_length):
        #     pl_mean = self.pl_length_ma.update_average(pl_mean, avg_pl_length)
        # if self.steps % 10 == 0 and self.steps > 20000:
        #     self.GAN.EMA()
        # if self.steps <=  and self.steps % 1000 == 2:
        #     self.GAN.reset_parameter_averaging()

        print(g_loss , d_loss)

        break
