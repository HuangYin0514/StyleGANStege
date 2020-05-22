import random
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import torch
from collections import OrderedDict
import matplotlib
import warnings

matplotlib.use('agg')


# ---------------------- Helper functions ----------------------
def save_network(network, path, epoch_label):
    file_path = os.path.join(path, 'net_%s.pth' % epoch_label)
    torch.save(network.state_dict(), file_path)


def load_network(network, path, epoch_label):
    file_path = os.path.join(path, 'net_%s.pth' % epoch_label)

    # Original saved file with DataParallel
    state_dict = torch.load(
        file_path, map_location=lambda storage, loc: storage)

    # If the model saved with DataParallel, the keys in state_dict contains 'module'
    if list(state_dict.keys())[0][:6] == 'module':
        # Create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            key_name = k[7:]  # remove `module.`
            new_state_dict[key_name] = v

        state_dict = new_state_dict

    # # ------------- PCB specific -------------
    # # Load PCB from another dataset, change the fc_list parameters' shape
    # for name in state_dict.keys():
    #     if name[0:7] == 'fc_list':
    #         desired_shape = network.state_dict()[name].shape
    #         if desired_shape != state_dict[name].shape:
    #             state_dict[name] = torch.randn(desired_shape)
    # # ------------------------------------------------

    network.load_state_dict(state_dict)

    return network


# load part network from checkpoint -------------------------------------------------------------
def load_part_network(network, path, epoch_label):

    # devie---------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # file name -----------------------------------------------------------------------------
    file_path = os.path.join(path, 'net_%s.pth' % epoch_label)

    # Original saved file with DataParallel-------------------------------------------------------
    state_dict = torch.load(file_path, map_location=torch.device(device))

    # state dict--------------------------------------------------------------------------
    model_dict = network.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    # load model state ---->{matched_layers, discarded_layers}------------------------------------
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    network.load_state_dict(model_dict)

    # assert model state ------------------------------------------------------------------------
    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
            format(path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )

    return network
