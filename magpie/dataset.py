from collections import namedtuple
import enum
import gc

import torch
import torch.utils.data as data


class DatasetType(enum.Enum):

    TRAINING = enum.auto()
    VALIDATION = enum.auto()
    TEST = enum.auto()


DatasetDescriptor = namedtuple("DatasetDescriptor", "length, mean, mean2")


class RawAudioDataset(data.Dataset):

    def __init__(self, config, set_type, descriptor, audio, labels=None):
        self.set_type = set_type
        self.audio = audio
        self.descriptor = descriptor
        self.labels = labels

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.audio[idx]
        else:
            return self.audio[idx], self.labels[idx]

    def free(self):
        self.audio = None
        self.labels = None
        gc.collect()

    @classmethod
    def splits(cls, config, data_dict):
        ds = data_dict["descriptor"]
        audio = data_dict["audio"]
        labels = data_dict.get("labels")

        train_pct = config["train_pct"]
        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]
        tot_pct = train_pct + dev_pct + test_pct
        audio_len = audio.size(0)
        train_idx = int(train_pct / tot_pct * audio_len)
        dev_idx = train_idx + int(dev_pct / tot_pct * audio_len)

        train_audio = audio[:train_idx]
        dev_audio = audio[train_idx:dev_idx]
        test_audio = audio[dev_idx:]
        audio_splits = (train_audio, dev_audio, test_audio)

        train_labels = labels[:train_idx] if labels else None
        dev_labels = labels[train_idx:dev_idx] if labels else None
        test_labels = labels[dev_idx:] if labels else None
        label_splits = (train_labels, dev_labels, test_labels)

        ds_types = (DatasetType.TRAINING, DatasetType.VALIDATION, DatasetType.TEST)
        datasets = [cls(config, ds_type, ds, audio, labels=labels) for 
            audio, labels, ds_type in zip(audio_splits, label_splits, ds_type)]
        return datasets
