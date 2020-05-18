import torch

from utils.util import *
from torch import nn
from .NoiseVectorizer import *


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# ExtractModule--------------------------------------------------------------------------------
class ExtractModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=64 * 2,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False)
        self.batch_norm2 = nn.BatchNorm2d(64 * 2)
        self.conv3 = nn.Conv2d(in_channels=64 * 2,
                               out_channels=64 * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False)
        self.batch_norm3 = nn.BatchNorm2d(64 * 4)
        self.conv4 = nn.Conv2d(in_channels=64 * 4,
                               out_channels=64 * 8,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False)
        self.batch_norm4 = nn.BatchNorm2d(64 * 8)
        self.conv5 = nn.Conv2d(in_channels=64 * 8,
                               out_channels=1,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False)
        self.linear1 = nn.Linear(in_features=64 * 8 * 4 * 4,
                                 out_features=100,
                                 bias=True)
        self.tanh = nn.Tanh()
        self.apply(weights_init)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.batch_norm3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.batch_norm4(out)
        out = self.relu(out)
        out = out.view(in_size, -1)
        out = self.linear1(out)
        # out = self.tanh(out)
        return out


# ExtractNet------------------------------------------------------------------------
class ExtractNet(nn.Module):

    def __init__(self, lr=2e-4, steps=1, beta1=0.5, beta2=0.999, **kwargs):
        super().__init__()
        self.lr = lr
        self.steps = steps

        self.beta1 = beta1
        self.beta2 = beta2

        self.E = ExtractModule()
        self.N = NoiseVectorizer(100)

        generator_params = list(self.E.parameters()) + list(self.N.parameters())
        self.E_opt = torch.optim.Adam(generator_params, lr=self.lr, betas=(self.beta1, self.beta2))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(
                    m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return x
