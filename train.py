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
    pl_mean = 0
    last_gp_loss = 0
    d_loss = 0
    g_loss = 0
    EPS = 1e-8
    pl_length_ma = EMA(0.99)

    # +++++++++++++++++++++++++++++++++start++++++++++++++++++++++++++++++++++++++++
    for step in range(args.start_steps, args.num_train_steps):

        model.train()

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
        d_loss = float(total_disc_loss)
        model.D_opt.step()

        # train generator************************************
        model.G_opt.zero_grad()
        for i in range(args.gradient_accumulate_every):
            # w--------------------------------
            style = get_latents_fn(batch_size, num_layers, latent_dim)
            w_space = latent_to_w(model.S, style)
            w_styles = styles_def_to_tensor(w_space)

            # noise--------------------------------
            noise = custom_image_nosie(batch_size, 100)
            noise_styles = latent_to_nosie(model.N, noise)

            # fake--------------------------------
            generated_images = model.G(w_styles, noise_styles)
            fake_output = model.D(generated_images)
            loss = fake_output.mean()
            gen_loss = loss

            # pl loss--------------------------------
            if apply_path_penalty:
                std = 0.1 / (w_styles.std(dim=0, keepdim=True) + EPS)
                w_styles_2 = w_styles + torch.randn(w_styles.shape).to(device) / (std + EPS)
                pl_images = model.G(w_styles_2, noise_styles)
                pl_lengths = ((pl_images - generated_images)**2).mean(dim=(1, 2, 3))
                avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                if pl_mean is not None:
                    pl_loss = ((pl_lengths - pl_mean)**2).mean()
                    if not torch.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss

            gen_loss = gen_loss / args.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)
            gen_loss.backward()

            # total loss--------------------------------
            total_gen_loss += loss.detach().item() / args.gradient_accumulate_every

        g_loss = float(total_gen_loss)
        model.G_opt.step()

        # train N************************************
        model.N_opt.zero_grad()
        for i in range(args.gradient_accumulate_every):
            # w--------------------------------
            get_latents_fn = mixed_list if random() < args.mixed_prob else noise_list
            style = get_latents_fn(batch_size, num_layers, latent_dim)
            w_space = latent_to_w(model.S, style)
            w_styles = styles_def_to_tensor(w_space)

            # noise--------------------------------
            noise = custom_image_nosie(batch_size, 100)
            noise_styles = latent_to_nosie(model.N, noise)

            generated_images = model.G(w_styles, noise_styles)
            decode_msg = model.E(generated_images)
            # loss----------------------
            divergence = nn.MSELoss()(decode_msg, noise)
            E_loss = divergence
            E_loss.backward()
        model.N_opt.step()

        # train E************************************
        model.E_opt.zero_grad()
        for i in range(args.gradient_accumulate_every):
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
            E_loss = divergence
            E_loss.backward()

        model.E_opt.step()

        # BER {1,2,3}------------------------------------------
        BER_1 = compute_BER(decode_msg.detach(), noise, sigma=1)
        BER_2 = compute_BER(decode_msg.detach(), noise, sigma=2)
        BER_3 = compute_BER(decode_msg.detach(), noise, sigma=3)

        # calculate moving averages ---------------------------------------
        if apply_path_penalty and not np.isnan(avg_pl_length):
            pl_mean = pl_length_ma.update_average(pl_mean, avg_pl_length)
        if step % 10 == 0 and step > 20000:
            model.EMA()
        if step <= 25000 and step % 1000 == 2:
            model.reset_parameter_averaging()

        if step % 10 == 0:
            logger.info('step {}/{}'.format(step + 1, args.num_train_steps))
            logger.info('g_loss: {:.4f}, d_loss {:.4f}'.format(g_loss, d_loss))
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

    # stop time -------------------------------------------------------------
    time_elapsed = time.time() - start_time
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # Save final model weights----------------------------------------------
    checkpointNet.save_network(model, save_dir_path, 'final')
