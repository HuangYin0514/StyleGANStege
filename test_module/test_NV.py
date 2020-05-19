
import sys
sys.path.append('/home/hy/vscode/StyleGANStege')

import torch
from models.NoiseVectorizer import *



if __name__ == "__main__":
    m=NoiseVectorizer(100)
    inp = torch.randn(3,100)
    print(inp.shape)