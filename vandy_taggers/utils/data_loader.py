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
        return len(self.file_list)

    def transform_features(self, X):
        """
        Args:
            X (List): List of all the features
        """
        all_data = [] # N_enevts x N_branches x N_features
        n_branches = len(X)
        n_enents = X[0].shape[0]
        for i in range(n_enents):
            event_data = []
            for j in range(n_branches):
                event_data.append(torch.from_numpy(X[j][i]).to(self.device))
            all_data.append(event_data)
        return all_data # N_events x N_branches x N_features

    def transform_labels(self, y):
        return [torch.from_numpy(y[i]).to(self.device) for i in range(len(y))]

    def __getitem__(self, file_idx):
        """
        Returns the data for the given index.
        """
        # file_idx, idx = self.map_to_location(idx)
        # Read the data from the file
        with open(self.file_list[file_idx], "rb") as f:
            data = pickle.load(f)
        X = self.transform_features(data[0])
        y = self.transform_labels(data[1])
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
        self.class_weights = []

    def get_loader(self):
        """
        Returns the dataloader for the dataset.
        """
        dataset = CustomDS(self.input_folder)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, drop_last=self.drop_last)