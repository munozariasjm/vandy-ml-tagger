import glob
import os
import torch
import joblib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from alt_taggers.modules.models.part import ParticleTransformer, PartTrainer
from alt_taggers.utils.data_loader import DataLoader
from alt_taggers.utils.pt_ranger import Ranger

def parse_args():
    import argparse

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


if __name__ == "__main__":

    arguments = parse_args()
    model = torch.load(arguments.modelpath)

    print("Loading data...")
    OUT_CONVERTED_PATH = arguments.inputdir
    BS = int(arguments.batch_size)
    test_loader = DataLoader(OUT_CONVERTED_PATH,
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
    # Convert to numpy
    predictions = predictions.cpu().numpy()
    the_targets = torch.cat(all_targets)
    the_targets = the_targets.cpu().numpy()
    data = {"predictions": predictions, "targets": the_targets}

    if not os.path.exists(arguments.outputdir):
        os.makedirs(arguments.outputdir)

    with open(arguments.outputdir, "wb") as f:
        joblib.dump(data, f)
