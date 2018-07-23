import gc

from . import dataset as ds


class LabeledRawAudioDataset(ds.MagpieDataset, name="labeled-raw"):

    def __init__(self, config, set_type, descriptor, audio, labels):
        super().__init__()
        self.set_type = set_type
        self.audio = audio
        self.descriptor = descriptor
        self.labels = labels

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        return self.audio[idx], self.labels[idx]

    def free(self):
        self.audio = None
        self.labels = None
        gc.collect()

    @classmethod
    def splits(cls, config, data_dict):
        descriptor = data_dict["descriptor"]
        audio = data_dict["audio"]
        labels = data_dict.get("labels")

        train_pct = config["train_pct"]
        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]
        tot_pct = train_pct + dev_pct + test_pct
        audio_len = len(audio)
        train_idx = int(train_pct / tot_pct * audio_len)
        dev_idx = train_idx + int(dev_pct / tot_pct * audio_len)

        train_audio = audio[:train_idx]
        dev_audio = audio[train_idx:dev_idx]
        test_audio = audio[dev_idx:]
        audio_splits = (train_audio, dev_audio, test_audio)

        train_labels = labels[:train_idx]
        dev_labels = labels[train_idx:dev_idx]
        test_labels = labels[dev_idx:]
        label_splits = (train_labels, dev_labels, test_labels)

        ds_types = (ds.DatasetType.TRAINING, ds.DatasetType.VALIDATION, ds.DatasetType.TEST)
        datasets = [cls(config, ds_type, descriptor, audio, labels) for 
            audio, labels, ds_type in zip(audio_splits, label_splits, ds_type)]
        return datasets
