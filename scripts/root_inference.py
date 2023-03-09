import argparse
import glob
import os
import re
from src.utils.dataset_conversion import PartData
from src.utils.data_loader import CustomLoader
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser("Run inference with  Particle Transformer")
    parser.add_argument(
        "-i",
        "--inputdir",
        required=True,
        help="Directory of the converted .*pkl files to test... can use glob hints"
    )
    parser.add_argument(
        "-m",
        "--modelpath",
        required=True
        help="Path to the model to use for inference."
    )
    parser.add_argument(
        "-o",
        "--outputdir",
        required=True,
        help="Output directory where to save the predictions."
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        required=False,
        default=128,
        help="Batch size."
    )
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    model = torch.load(args.modelpath)
    print("Loading data...")
    OUT_CONVERTED_PATH = args.inputdir
    BS = int(args.batch_size)
    test_loader = CustomLoader(OUT_CONVERTED_PATH,
                             BS,
                             shuffle=False,
                             ).get_loader()
    print("Running inference...")
    all_predictions = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for (X, y) in test_loader:
            X = X.to(device)
            y = y.to(device)
            predictions = model(X)
            all_predictions.append(predictions)
            all_targets.append(y)
    print("Saving predictions...")
    predictions = torch.cat(all_predictions)

    # Save predictions
