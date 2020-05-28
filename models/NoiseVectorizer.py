from torch import nn
from utils.util import leaky_relu
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoiseVectorizer(nn.Module):
    def __init__(self, emb):
        super().__init__()

        self.layer_channel = [512, 256, 128, 64, 32]
        layer_h_w = [4, 8, 16, 32, 64]

        # --------------------------------------------------
        layer_index = 0
        self.layer0 = nn.Sequential(
            nn.Linear(emb, self.layer_channel[layer_index]),
            nn.BatchNorm1d(self.layer_channel[layer_index]),
            nn.Sigmoid()
        )
        self.layer0_pool = nn.AdaptiveAvgPool2d((layer_h_w[layer_index], layer_h_w[layer_index]))

        # --------------------------------------------------
        layer_index = 1
        self.layer1 = nn.Sequential(
            nn.Linear(emb, self.layer_channel[layer_index]),
            nn.BatchNorm1d(self.layer_channel[layer_index]),
            nn.Sigmoid()
        )
        self.layer1_pool = nn.AdaptiveAvgPool2d((layer_h_w[layer_index], layer_h_w[layer_index]))

        # --------------------------------------------------
        layer_index = 2
        self.layer2 = nn.Sequential(
            nn.Linear(emb, self.layer_channel[layer_index]),
            nn.BatchNorm1d(self.layer_channel[layer_index]),
            nn.Sigmoid()
        )
        self.layer2_pool = nn.AdaptiveAvgPool2d((layer_h_w[layer_index], layer_h_w[layer_index]))

        # --------------------------------------------------
        layer_index = 3
        self.layer3 = nn.Sequential(
            nn.Linear(emb, self.layer_channel[layer_index]),
            nn.BatchNorm1d(self.layer_channel[layer_index]),
            nn.Sigmoid()
        )
        self.layer3_pool = nn.AdaptiveAvgPool2d((layer_h_w[layer_index], layer_h_w[layer_index]))

        # --------------------------------------------------
        layer_index = 4
        self.layer4 = nn.Sequential(
            nn.Linear(emb, self.layer_channel[layer_index]),
            nn.BatchNorm1d(self.layer_channel[layer_index]),
            nn.Sigmoid()
        )
        self.layer4_pool = nn.AdaptiveAvgPool2d((layer_h_w[layer_index], layer_h_w[layer_index]))

    def forward(self, x):
        x0 = self.layer0(x).reshape(-1, self.layer_channel[0], 1, 1)
        x0 = self.layer0_pool(x0)

        x1 = self.layer1(x).reshape(-1, self.layer_channel[1], 1, 1)
        x1 = self.layer1_pool(x1)

        x2 = self.layer2(x).reshape(-1, self.layer_channel[2], 1, 1)
        x2 = self.layer2_pool(x2)

        x3 = self.layer3(x).reshape(-1, self.layer_channel[3], 1, 1)
        x3 = self.layer3_pool(x3)

        x4 = self.layer4(x).reshape(-1, self.layer_channel[4], 1, 1)
        x4 = self.layer4_pool(x4)

        return [x0, x1, x2, x3, x4]
