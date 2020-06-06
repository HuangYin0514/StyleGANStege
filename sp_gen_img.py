
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
from pathlib import Path

from dataloader import getDataLoader
from models import build_model
from utils.util import *
from utils import checkpointNet


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
    return torch.index_select(a, dim, order_index)

# generate_images --------------------------------------------
@torch.no_grad()
def generate_images(stylizer, generator, latents, noise, args):
    w = latent_to_w(stylizer, latents)
    w_styles = styles_def_to_tensor(w)
    generated_images = evaluate_in_chunks(args.batch_size, generator,
                                          w_styles, noise)
    generated_images.clamp_(0., 1.)
    return generated_images

# generate_truncated --------------------------------------------
@torch.no_grad()
def generate_truncated(S, G, style, noi, av, batch_size, trunc_psi=0.6, num_image_tiles=8):
    latent_dim = G.latent_dim

    if av is None:
        z = noise(2000, latent_dim)
        samples = evaluate_in_chunks(batch_size, S, z).cpu().numpy()
        av = np.mean(samples, axis=0)
        av = np.expand_dims(av, axis=0)

    w_space = []
    for tensor, num_layers in style:
        tmp = S(tensor)
        av_torch = torch.from_numpy(av).to(device)
        tmp = trunc_psi * (tmp - av_torch) + av_torch
        w_space.append((tmp, num_layers))

    w_styles = styles_def_to_tensor(w_space)
    generated_images = evaluate_in_chunks(batch_size, G, w_styles, noi)
    return generated_images.clamp_(0., 1.)


def genimg(model):
    # parameters --------------------------------
    ext = 'jpg'
    num_image_tiles = 10
    num_rows = num_image_tiles
    latent_dim = model.G.latent_dim
    image_size = model.G.image_size
    num_layers = model.G.num_layers
    batch_size = 3
    av = None
    num = 'gen'

    # noise-------------------------------------------
    noise_ = custom_image_nosie(num_rows**2, 100)
    n = latent_to_nosie(model.N, noise_)

    # diff noise============================================================================================
    latents_W = noise(1, latent_dim)
    latents_W = latents_W.repeat(num_rows**2, 1)
    latents = [(latents_W, num_layers)]
    generated_images = generate_truncated(model.SE, model.GE, latents, n, av, batch_size)
    for index, img in enumerate(generated_images):
        torchvision.utils.save_image(img, str(Path('experiments/sp/dn') / f'{str(num)+str(index)}-dn.{ext}'))
    torchvision.utils.save_image(generated_images, str(Path('experiments/sp/') / f'{str(num)+str(0)}-dn-all.{ext}'))

    # diff w============================================================================================
    latents = noise_list(num_rows**2, num_layers, latent_dim)
    noise_ = custom_image_nosie(1, 100)
    noise_ = noise_.repeat(num_rows**2, 1)
    ndn = latent_to_nosie(model.N, noise_)
    generated_images = generate_truncated(model.SE, model.GE, latents, ndn, av, batch_size)
    for index, img in enumerate(generated_images):
        torchvision.utils.save_image(img, str(Path('experiments/sp/dw') / f'{str(num)+str(index)}-dw.{ext}'))
    torchvision.utils.save_image(generated_images, str(Path('experiments/sp/') / f'{str(num)+str(0)}-dw-all.{ext}'))

    # dcgan============================================================================================
    latents = noise_list(num_rows**2, num_layers, latent_dim)
    noise_ = custom_image_nosie(num_rows**2, 100)
    n = latent_to_nosie(model.N, noise_)
    generated_images = generate_truncated(model.SE, model.GE, latents, n, av, batch_size)
    for index, img in enumerate(generated_images):
        torchvision.utils.save_image(img, str(Path('experiments/sp/dcgan') / f'{str(num)+str(index)}-dcgan.{ext}'))
    torchvision.utils.save_image(generated_images, str(Path('experiments/sp/') / f'{str(num)+str(0)}-dcgan-all.{ext}'))


if __name__ == "__main__":
    # Fix random seed---------------------------------------------------------------------------
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    # model------------------------------------------------------------------------------------
    checkpoint = '/home/hy/vscode/StyleGANStege/experiments/Celeba'
    which_epoch = 'final'
    experiment = 'StyleGAN2'
    model = build_model(experiment, image_size=64, lr=1)
    model = checkpointNet.load_part_network(model, checkpoint, which_epoch)
    model = model.to(device)

    genimg(model)
