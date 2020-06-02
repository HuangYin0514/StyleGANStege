
import sys
sys.path.append('/home/hy/vscode/StyleGANStege')
from models.NoiseVectorizer import *
import torch

    



if __name__ == "__main__":

    m = NoiseVectorizer(100)
    inp = torch.randn(3, 100)
    # inp = torch.FloatTensor([[1,2,3,4],[5,6,7,8]])
    out = m(inp)
    for _ in range(5):
        print(out[_].shape)
    print('complete check.')

