from torch import nn
from utils.util import leaky_relu
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoiseVectorizer(nn.Module):
    def __init__(self, emb):
        super().__init__()

        # self.layer_channel = [512, 256, 128, 64, 32]
        self.layer_h_w = [4**2, 8**2, 16**2, 32**2, 64**2]

        # --------------------------------------------------
        layer_index = 0
        self.layer0 = nn.Sequential(
            nn.Linear(emb, self.layer_h_w[layer_index]),
            nn.BatchNorm1d(self.layer_h_w[layer_index]),
            nn.Sigmoid()
        )
        # --------------------------------------------------
        layer_index = 1
        self.layer1 = nn.Sequential(
            nn.Linear(self.layer_h_w[layer_index-1], self.layer_h_w[layer_index]),
            nn.BatchNorm1d(self.layer_h_w[layer_index]),
            nn.Sigmoid()
        )
        # --------------------------------------------------
        layer_index = 2
        self.layer2 = nn.Sequential(
            nn.Linear(self.layer_h_w[layer_index-1], self.layer_h_w[layer_index]),
            nn.BatchNorm1d(self.layer_h_w[layer_index]),
            nn.Sigmoid()
        )
        # --------------------------------------------------
        layer_index = 3
        self.layer3 = nn.Sequential(
            nn.Linear(self.layer_h_w[layer_index-1], self.layer_h_w[layer_index]),
            nn.BatchNorm1d(self.layer_h_w[layer_index]),
            nn.Sigmoid()
        )
        # --------------------------------------------------
        layer_index = 4
        self.layer4 = nn.Sequential(
            nn.Linear(self.layer_h_w[layer_index-1], self.layer_h_w[layer_index]),
            nn.BatchNorm1d(self.layer_h_w[layer_index]),
            nn.Sigmoid()
        )

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x0 = x0.reshape(-1, 4, 4, 1)
        x1 = x1.reshape(-1, 8, 8, 1)
        x2 = x2.reshape(-1, 16, 16, 1)
        x3 = x3.reshape(-1, 32, 32, 1)
        x4 = x4.reshape(-1, 64, 64, 1)

        return [0.5*x0, 0.5*x1, 0.5*x2, 0.5*x3, 0.5*x4]
