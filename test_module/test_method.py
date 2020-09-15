import torch
from torch import nn

if __name__ == "__main__":
    m = nn.AdaptiveMaxPool2d((64,64))
    x= torch.randn(3,3,96,96)
    x = m(x)
    print(x.shape)