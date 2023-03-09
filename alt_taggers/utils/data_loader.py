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
    def __init__(self, input_folder, device="cuda") -> None:
        super().__init__()
        if not input_folder.endswith(".pkl"):
            path = (input_folder + "*.pkl").replace("**", "*")
            self.file_list = glob.glob(path)
        else:
            self.file_list = glob.glob(os.path.join(input_folder) + "/*.pkl")
        assert len(self.file_list) > 0, "No .pkl files found".format(input_folder)
        self.file_list.sort()
        self.device = device
        self.all_files_len = self._all_files_len()


    def _all_files_len(self):
        """
        Returns the total number of events in the dataset.
        """
        self.data_list = [] # Len of all files
        for file in self.file_list:
            with open(file, "rb") as f:
                data = pickle.load(f)
                ldata = data[1].shape[0]
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
        # Read the data from the file
        with open(self.file_list[file_idx], "rb") as f:
            data = pickle.load(f)
        X = data
        X = ([torch.from_numpy(x).permute(1, 2, 0)[idx].to(self.device) for x in data[0]])
        y = torch.from_numpy(data[1][idx]).to(self.device)
        return X, y


class DataLoader:
    """
    Data loader for the dataset.
    """
    def __init__(self, input_folder, batch_size, num_workers=0, shuffle=True, drop_last=True) -> None:
        super().__init__()
        self.input_folder = input_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last

    def get_loader(self):
        """
        Returns the dataloader for the dataset.
        """
        dataset = CustomDS(self.input_folder)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, drop_last=self.drop_last)