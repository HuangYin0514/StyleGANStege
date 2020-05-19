
import sys
sys.path.append('/home/hy/vscode/StyleGANStege')

import torch
from models import build_model
from utils import checkpointNet



if __name__ == "__main__":

    checkpoint = '/home/hy/vscode/StyleGANStege/experiments/ExtractNet'
    model = build_model('ExtractNet', image_size=64)
    model = checkpointNet.load_part_network(model, checkpoint, 'final')
    print(model)
