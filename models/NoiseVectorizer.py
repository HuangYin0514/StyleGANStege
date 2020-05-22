from torch import nn
from utils.util import leaky_relu
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoiseVectorizer(nn.Module):
    def __init__(self, emb):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(emb, emb),
            leaky_relu(0.2),
            nn.Linear(emb, 512),
            leaky_relu(0.2),
            nn.Linear(512, 1024),
            leaky_relu(0.2),
            nn.Linear(1024, 2048),
            leaky_relu(0.2),
            nn.Linear(2048, 4096),
            # nn.BatchNorm1d(4096),
            nn.Sigmoid()
        )

    def forward(self, x):

        # return torch.FloatTensor(x.size(0), 64, 64, 1).uniform_(0., 1.).to(device)
        return self.net(x).reshape(-1, 64, 64, 1)

