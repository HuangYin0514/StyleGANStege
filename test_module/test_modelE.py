
import sys
sys.path.append('/home/hy/vscode/StyleGANStege')

import torch
from models import build_model



if __name__ == "__main__":

    model = build_model('ExtractNet', image_size=64)
    print(model)

    inp = torch.randn(3, 3, 64, 64)
    out = model.E(inp)

    print(inp.shape)
    print(out.shape)

    print('complete check.')
