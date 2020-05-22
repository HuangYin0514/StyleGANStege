
import torch

from .Stylegan import StyleGAN2
from .ExtractNet import ExtractNet
from .ExtractNetSimilarE import ExtractNetSimilarE

__model_factory = {
    'StyleGAN2': StyleGAN2,
    'ExtractNet': ExtractNet,
    'ExtractNetSimilarE':ExtractNetSimilarE
}


def build_model(name, image_size, **kwargs):
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(name, avai_models))
    return __model_factory[name](image_size=image_size, **kwargs)
