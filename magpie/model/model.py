import torch
import torch.nn as nn


registry = {}

class MagpieModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def __init_subclass__(cls, prefix, **kwargs):
        registry[prefix] = cls


def find_model(model_name):
    return registry[model_name.split(".")[0]]
