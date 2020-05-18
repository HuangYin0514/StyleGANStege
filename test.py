
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
def generate_truncated(S, G, style, noi, av, args, trunc_psi=0.6, num_image_tiles=8):
    latent_dim = G.latent_dim

    if av is None:
        z = noise(2000, latent_dim)
        samples = evaluate_in_chunks(args.batch_size, S, z).cpu().numpy()
        av = np.mean(samples, axis=0)
        av = np.expand_dims(av, axis=0)

    w_space = []
    for tensor, num_layers in style:
        tmp = S(tensor)
        av_torch = torch.from_numpy(av).to(device)
        tmp = trunc_psi * (tmp - av_torch) + av_torch
        w_space.append((tmp, num_layers))

    w_styles = styles_def_to_tensor(w_space)
    generated_images = evaluate_in_chunks(args.batch_size, G, w_styles, noi)
    return generated_images.clamp_(0., 1.)


# ---------------------- Start testing ----------------------
def test(model, save_dir_path, args, num=0, num_image_tiles=8):
    model.eval()

    # parameters --------------------------------
    ext = 'jpg'
    num_rows = num_image_tiles
    latent_dim = model.G.latent_dim
    image_size = model.G.image_size
    num_layers = model.G.num_layers
    av = None

    # w----------------------------------------------
    latents = noise_list(num_rows**2, num_layers, latent_dim)

    # noise-------------------------------------------
    noise_ = custom_image_nosie(num_rows**2, 100)
    n = latent_to_nosie(model.N, noise_)

    # regular-------------------------------------------
    generated_images = generate_images(model.S, model.G, latents, n, args)
    torchvision.utils.save_image(generated_images, str(Path(save_dir_path) / f'{str(num)}.{ext}'), nrow=num_rows)

    # moving averages-------------------------------------------
    generated_images = generate_truncated(model.SE, model.GE, latents, n, av, args)
    torchvision.utils.save_image(generated_images, str(Path(save_dir_path) / f'{str(num)}-ema.{ext}'), nrow=num_rows)

    # mixing regularities--------------------------------------------
    nn = noise(num_rows, latent_dim)
    tmp1 = tile(nn, 0, num_rows)
    tmp2 = nn.repeat(num_rows, 1)
    tt = int(num_layers / 2)
    mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]
    generated_images = generate_truncated(model.SE, model.GE, mixed_latents, n, av, args)
    torchvision.utils.save_image(generated_images, str(Path(save_dir_path) / f'{str(num)}-mr.{ext}'), nrow=num_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing arguments')

    parser.add_argument('--experiment', type=str, default='StyleGAN2')

    parser.add_argument('--save_path', type=str, default='./experiments')
    parser.add_argument('--which_epoch', default='final', type=str, help='0,1,2,3...or final')
    parser.add_argument('--checkpoint', type=str, default='/home/hy/vscode/StyleGANStege/experiments/Celeba')

    parser.add_argument('--dataset', type=str, default='Celeba')
    parser.add_argument('--batch_size', default=3, type=int, help='batch_size')
    parser.add_argument('--image_size', default=64)

    parser.add_argument('--lr', type=float, default=0.1)

    args = parser.parse_args()

    # devie---------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save_dir_path-----------------------------------------------------------------------------------
    save_dir_path = os.path.join(args.save_path, args.dataset)
    os.makedirs(save_dir_path, exist_ok=True)

    # model------------------------------------------------------------------------------------
    model = build_model(args.experiment, image_size=args.image_size, lr=args.lr)
    model = checkpointNet.load_part_network(model, args.checkpoint, args.which_epoch)
    model = model.to(device)

    test(model, save_dir_path, args)

    # torch.cuda.empty_cache()
