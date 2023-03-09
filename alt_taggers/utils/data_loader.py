"""
Loads the data from the data directory into a pytorch dataset.
The dataset is stored in pickle files with the following structure:
X, y =
{
    "global_branches": {...},
    "cpf_branches": {...},
    "npf_branches": {...},
    "vtx_branches": {...},
    "cpf_pts_branches": {...},
    "npf_pts_branches": {...},
    "vtx_pts_branches": {...},
},
{
    "isB": [...],
    "isBB": [...],
    ...
}
"""
import os
import uproot
import torch
import glob
import numpy as np
from torch.utils.data import Dataset
import pickle

class CustomDS(Dataset):
    """
    Custom dataset loader for the data.
    """
    def __init__(self, input_folder) -> None:
        super().__init__()

        self.file_list = glob.glob(os.path.join(input_folder, "*.pkl"))
        assert len(self.file_list) > 0, "No .pkl files found".format(input_folder)
        self.file_list.sort()

        self.all_files_len = self._all_files_len()


    def _all_files_len(self):
        """
        Returns the total number of events in the dataset.
        """
        self.data_list = [] # Len of all files
        for file in self.file_list:
            with open(file, "rb") as f:
                data = pickle.load(f)
                print(data[1])
                ldata = len(data[1]["isB"])
                self.data_list.append(ldata)
        self.total_len = np.sum(self.data_list)


    def map_to_location(self, idx):
        """
        Maps the index to the file and the index in the file.
        """
        file_idx = 0
        while idx >= self.data_list[file_idx]:
            idx -= self.data_list[file_idx]
            file_idx += 1

        return file_idx, idx

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        """
        Returns the data for the given index.
        """
        file_idx, idx = self.map_to_location(idx)

        with open(self.file_list[file_idx], "rb") as f:
            data = pickle.load(f)

        return data[0], data[1]




