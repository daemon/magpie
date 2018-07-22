from sys import stdin, stderr
import argparse
import multiprocessing as mp
import os

from tqdm import tqdm
import librosa
import torch
import numpy as np

import magpie.dataset as ds


def process_audio(proc_idx, files, out_filename):
    try:
        os.makedirs(os.path.dirname(out_filename))
    except FileExistsError:
        pass
    audio_data = []
    for filename in tqdm(files, file=stderr, position=proc_idx):
        audio = torch.from_numpy(librosa.core.load(filename, sr=None)[0])
        audio_data.append(audio)

    torch.save(audio_data, out_filename)


def main():
    description = "Creates a speech dataset from stdin."
    epilog = "Usage:\npython -m magpie.utils.build_dataset --output_file dataset.pt"
    parser = argparse.ArgumentParser(description, epilog)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--tmp_prefix", type=str, default="tmp/tmp-")
    args = parser.parse_args()
    
    files_list = [[] for _ in range(mp.cpu_count())]
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

    print(f"Spinning up {mp.cpu_count()} processes...", file=stderr)
    processes = [mp.Process(target=process_audio, args=args) for args in zip(range(mp.cpu_count()), 
        files_list, out_filenames)]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()

    data_list = [torch.load(filename) for filename in out_filenames]
    audio_ds = []
    list(map(audio_ds.extend, data_list))
    all_audio = torch.cat(audio_ds)
    descriptor = ds.DatasetDescriptor(mean=all_audio.mean(-1), 
        mean2=(all_audio**2).mean(-1), length=all_audio.size(0))


    data_dict = dict(audio=audio_ds, descriptor=descriptor)
    if has_labels:
        data_dict["labels"] = label_ds
    print(f"Saving to {args.output_file}...", file=stderr)
    torch.save(data_dict, args.output_file)
    print(f"Deleting temporary files...", file=stderr)
    for filename in out_filenames:
        os.unlink(filename)


if __name__ == "__main__":
    main()