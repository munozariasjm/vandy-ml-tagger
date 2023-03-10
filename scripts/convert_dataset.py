import argparse
import glob
import os
import re
from vandy_taggers.utils.dataset_conversion import PartData

def parse_args():
    parser = argparse.ArgumentParser("Convert root to pickle datasets")
    parser.add_argument(
        "-c",
        "--data_class",
        required=True,
        help="Class of dataset.",
        default="Part",
        )
    parser.add_argument(
        "-i",
        "--inputdir",
        required=True,
        help="Directory of input numpy files."
    )
    parser.add_argument(
        "-o",
        "--outputdir",
        required=True,
        help="Output directory."
    )
    parser.add_argument(
        "--train-test-split", default=0.9, help="Training / testing split fraction."
    )
    return parser.parse_args()


def natural_sort(l):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(l, key=alphanum_key)


if __name__ == "__main__":
    args = parse_args()

    if args.data_class == "ParT":
        convert = PartData.convert

    sources = natural_sort(glob.glob(os.path.join(args.inputdir, "*.root")))
    assert len(sources) > 0, "No files found in {}".format(args.inputdir)
    n_train = int(args.train_test_split * len(sources))

    train_sources = sources[:n_train]
    test_sources = sources[n_train:]

    convert(train_sources, destdir=args.outputdir, basename="train_file")
    convert(test_sources, destdir=args.outputdir, basename="test_file")
