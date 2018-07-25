import torch
import torch.nn as nn


registry = {}

class MagpieModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def __init_subclass__(cls, prefix, **kwargs):
        registry[prefix] = cls


def make_ortho_weight(input_size, output_size):
    return nn.init.orthogonal_(torch.empty(output_size, input_size))


def find_model(model_name):
    return registry[model_name.split(".")[0]]
