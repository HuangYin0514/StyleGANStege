from torch import nn
from torch_optimizer import DiffGrad

from utils.util import *
from .StyleVectorizer import *
from .NoiseVectorizer import *
from .Generator import *
from .Discriminator import *
from .ExtractNetSimilarE import ExtractNetSimilarE


class FineTuneStylegan(nn.Module):

    def __init__(self, image_size, latent_dim=512, noise_dim=100, style_depth=8, network_capacity=16, transparent=False, steps=1, lr=2e-4):
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)

        self.S = StyleVectorizer(latent_dim, style_depth)
        self.N = NoiseVectorizer(noise_dim)
        self.G = Generator(image_size, latent_dim, network_capacity, transparent=transparent)
        self.D = Discriminator(image_size, network_capacity, transparent=transparent)
        ###########################################
        self.E = ExtractNetSimilarE(64)

        self.SE = StyleVectorizer(latent_dim, style_depth)
        self.NE = NoiseVectorizer(noise_dim)
        self.GE = Generator(image_size, latent_dim, network_capacity, transparent=transparent)

        set_requires_grad(self.SE, False)
        set_requires_grad(self.NE, False)
        set_requires_grad(self.GE, False)

        generator_params = list(self.G.parameters())
        self.G_opt = DiffGrad(generator_params, lr=self.lr, betas=(0.5, 0.9))
        self.D_opt = DiffGrad(self.D.parameters(), lr=self.lr, betas=(0.5, 0.9))
        ###############################################
        # E_params = list(self.E.parameters())+list(self.G.downsample.parameters())
        E_params = list(self.E.parameters())+list(self.N.parameters())+list(self.G.parameters())

        # E_params = list(self.E.to_logit.parameters())
        # base_param_ids = set(map(id, self.E.to_logit.parameters()))
        # new_params = [p for p in self.E.parameters() if id(p) not in base_param_ids]
        # E_param_groups = [{'params': self.E.parameters(), 'lr': self.lr},
        #                 #   {'params': self.G.parameters(), 'lr': self.lr},
        #                 #   {'params': new_params, 'lr': self.lr},  # other E layers
        #                   {'params': self.N.parameters(), 'lr': self.lr}
        #                   ]

        self.E_opt = DiffGrad(E_params, lr=self.lr, betas=(0.5, 0.9))

        self.E_opt_scheduler = torch.optim.lr_scheduler.StepLR(self.E_opt, step_size=200, gamma=0.1)

        N_params = list(self.N.parameters())
        self.N_opt = DiffGrad(N_params, lr=self.lr, betas=(0.5, 0.9))

        self._init_weights()
        self.reset_parameter_averaging()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.NE, self.N)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.NE.load_state_dict(self.N.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x
