import glob
import sys
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from alt_taggers.modules.models.part import ParticleTransformer, PartTrainer
from alt_taggers.utils.data_loader import DataLoader