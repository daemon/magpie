import argparse
import glob
import os
import random


def print_gsc_dataset(input_folder, shuffle=False):
    wav_files = glob.glob(os.path.join(input_folder, "*", "*.wav"))
    print_strings = []
    for wav_file in wav_files:
        dirname = os.path.dirname(wav_file)
        label = os.path.basename(dirname)
        print_strings.append("\t".join((wav_file, label)))
    if shuffle:
        random.shuffle(print_strings)
    for print_str in print_strings:
        print(print_str)


def main():
    description = "Discovers and prints audio and label files from a folder."
    epilog = "Usage:\npython -m magpie.utils.print_audio --format gsc --folder speech_dataset"
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("--format", type=str, default="gsc", choices=["gsc", "comcast"])
    parser.add_argument("--folder", type=str)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    if args.format == "gsc":
        print_gsc_dataset(args.folder, shuffle=args.shuffle)


if __name__ == "__main__":
    main()