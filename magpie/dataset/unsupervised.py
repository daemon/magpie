import gc

import torch

from . import dataset as ds


class UnlabeledRawAudioDataset(ds.MagpieDataset, name="unlabeled-raw"):

    def __init__(self, config, set_type, descriptor, audio):
        super().__init__()
        self.set_type = set_type
        self.audio = audio
        self.descriptor = descriptor

    def __len__(self):
        return self.audio.size(0)

    def __getitem__(self, idx):
        return self.audio[idx], self.audio[idx]

    def free(self):
        self.audio = None
        gc.collect()

    @classmethod
    def splits(cls, config, data_dict):
        descriptor = data_dict["descriptor"]
        audio = data_dict["audio"]

        train_pct = config["train_pct"]
        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]
        tot_pct = train_pct + dev_pct + test_pct
        audio = torch.cat(audio)
        chunk_len = config["chunk_length"]
        audio = audio[:(audio.size(0) // chunk_len) * chunk_len].view(-1, chunk_len)
        audio_len = audio.size(0)
        train_idx = int(train_pct / tot_pct * audio_len)
        dev_idx = train_idx + int(dev_pct / tot_pct * audio_len)

        train_audio = audio[:train_idx]
        dev_audio = audio[train_idx:dev_idx]
        test_audio = audio[dev_idx:]
        audio_splits = (train_audio, dev_audio, test_audio)

        ds_types = (ds.DatasetType.TRAINING, ds.DatasetType.VALIDATION, ds.DatasetType.TEST)
        datasets = [cls(config, ds_t, descriptor, audio) for audio, ds_t in zip(audio_splits, ds_types)]
        return datasets