from collections import namedtuple
import enum

import torch.utils.data as data


registry = {}


class DatasetType(enum.Enum):

    TRAINING = enum.auto()
    VALIDATION = enum.auto()
    TEST = enum.auto()


class MagpieDataset(data.Dataset):

    def __init_subclass__(cls, name, **kwargs):
        registry[name] = cls


DatasetDescriptor = namedtuple("DatasetDescriptor", "length, mean, mean2")


def find_dataset(name):
    return registry[name]