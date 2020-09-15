import torch
from torch import nn

if __name__ == "__main__":
    m = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4, stride=3, padding=0, bias=False)
    x= torch.randn(3,3,96,96)
    x = m(x)
    print(x.shape)