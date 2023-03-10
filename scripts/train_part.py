import glob
import sys
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from vandy_taggers.modules.models.part import ParticleTransformer, PartTrainer
from vandy_taggers.utils.data_loader import DataLoader
from vandy_taggers.utils.pt_ranger import Ranger


def parse_args():
    import argparse

    parser = argparse.ArgumentParser("Train Particle Transformer")
    parser.add_argument(
        "-i",
        "--inputdir",
        required=True,
        help="Directory of the converted .*pkl files to use for training"
    )
    parser.add_argument(
        "-o",
        "--outputdir",
        required=True,
        help="Output directory where to save the trained models."
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        required=False,
        default=128,
        help="Batch size."
    )
    parser.add_argument(
        "-e",
        "--epochs",
        required=False,
        default=30,
        help="Number of epochs."
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        required=False,
        default=1e-3,
        help="Learning rate."
    )
    return parser.parse_args()

if __name__ == "__main__":

    arguments = parse_args()

    print("Loading data...")
    OUT_CONVERTED_PATH = arguments.inputdir
    BS = int(arguments.batch_size)
    train_loader = DataLoader(OUT_CONVERTED_PATH, BS).get_loader()
    val_loader = DataLoader(OUT_CONVERTED_PATH, BS).get_loader()

    print("Building model...")
    num_epochs = int(arguments.epochs)
    lr_epochs = max(1, int(num_epochs * 0.3))
    lr_rate = 0.01 ** (1.0 / lr_epochs)
    mil = list(range(num_epochs - lr_epochs, num_epochs))

    model = ParticleTransformer(num_classes = 6,
                 num_enc = 8,
                 num_head = 8,
                 embed_dim = 128,
                 cpf_dim = 16,
                 npf_dim = 8,
                 vtx_dim = 14,
                 for_inference = False,)
    model.to(device)

    optimizer = Ranger(model.parameters(), lr=float(arguments.learning_rate))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=mil, gamma=lr_rate)

    print("Training model...")
    trainer = PartTrainer(
        optimizer=optimizer,
        scheduler=scheduler,
        model=model,
        device=device,
    )
    trainer.train(num_epochs, train_loader, val_loader, path=arguments.outputdir)

    print("Done!")