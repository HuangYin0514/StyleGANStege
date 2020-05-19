
import sys
sys.path.append('/home/hy/vscode/StyleGANStege')

import torch
from models import build_model
from utils import checkpointNet



if __name__ == "__main__":

    checkpoint = '/home/hy/vscode/StyleGANStege/experiments/Celeba'
    model = build_model('StyleGAN2', image_size=64)
    model = checkpointNet.load_part_network(model, checkpoint, 'final')
    print(model)
