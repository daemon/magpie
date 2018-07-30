import torch
import torch.nn as nn


registry = {}

class MagpieModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def __init_subclass__(cls, prefix, **kwargs):
        registry[prefix] = cls

def find_rnn(rnn_type):
    if rnn_type.upper() == "LSTM":
        return nn.LSTM
    elif rnn_type.upper() == "GRU":
        return nn.GRU
    else:
        raise ValueError("RNN type not found")

def make_ortho_weight(input_size, output_size):
    return nn.init.orthogonal_(torch.empty(output_size, input_size))


def find_model(model_name):
    return registry[model_name.split(".")[0]]
