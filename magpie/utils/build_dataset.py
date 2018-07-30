from sys import stdin, stderr
import argparse
import gc
import os

from tqdm import tqdm
import librosa
import torch
import numpy as np

from magpie.utils.audio import MelSpectrogram
import magpie.dataset as ds


def process_audio(proc_idx, files, out_filename):
    try:
        os.makedirs(os.path.dirname(out_filename))
    except FileExistsError:
        pass
    audio_data = []
    mel_data = []
    mel = MelSpectrogram().cuda()
    for filename in tqdm(files, file=stderr, position=proc_idx):
        audio = torch.from_numpy(librosa.core.load(filename, sr=None)[0])
        log_mel = mel(audio.cuda()).cpu().squeeze(0)
        audio_data.append(audio)
        mel_data.append(log_mel)

    torch.save((audio_data, mel_data), out_filename)


def main():
    description = "Creates a speech dataset from stdin."
    epilog = "Usage:\npython -m magpie.utils.build_dataset --output_prefix dataset"
    parser = argparse.ArgumentParser(description, epilog)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--output_prefix", type=str, default="dataset")
    parser.add_argument("--tmp_prefix", type=str, default="tmp/tmp-")
    args = parser.parse_args()
    
    files_list = [[] for _ in range(args.chunks)]
    out_filenames = [f"{args.tmp_prefix}{idx}" for idx in range(len(files_list))]
    labels_list = [[] for _ in range(len(files_list))]
    has_labels = True
    
    for idx, line in enumerate(stdin):
        splits = line.split("\t")
        files_list[idx % len(files_list)].append(splits[0])
        if has_labels and len(splits) < 2:
            has_labels = False
            continue
        labels_list[idx % len(labels_list)].append(splits[1].strip())

    if has_labels:
        label_ds = []
        list(map(label_ds.extend, labels_list))

    for idx, (files, out_filename) in enumerate(zip(files_list, out_filenames)):
        process_audio(idx, files, out_filename)
        gc.collect()

    for idx, filename in enumerate(out_filenames):
        audio_ds, logmel_ds = torch.load(filename)
        audio_concat = torch.cat(audio_ds)
        lm_concat = torch.cat(logmel_ds)
        descriptor_raw = ds.DatasetDescriptor(mean=audio_concat.mean(-1), 
            mean2=(audio_concat**2).mean(-1), length=audio_concat.size(0))
        descriptor_lm = ds.DatasetDescriptor(mean=lm_concat.mean(1), 
            mean2=(lm_concat**2).mean(1), length=lm_concat.size(0))

        data_dict = dict(audio=audio_ds, logmel=logmel_ds)
        if has_labels:
            data_dict["labels"] = label_ds
        data_file = f"{args.output_prefix}.data.{idx}.pt"
        descriptor_file = f"{args.output_prefix}.desc.{idx}.pt"
        print(f"Saving to {data_file}...", file=stderr)
        torch.save(data_dict, data_file)
        print(f"Saving to {descriptor_file}...", file=stderr)
        torch.save((descriptor_raw, descriptor_lm), descriptor_file)
        print(f"Deleting temporary files...", file=stderr)
        os.unlink(filename)
        gc.collect()


if __name__ == "__main__":
    main()