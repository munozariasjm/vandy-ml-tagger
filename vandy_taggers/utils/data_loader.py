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
    def __init__(self, input_folder, device="cuda", get_pt=True) -> None:
        super().__init__()
        if not input_folder.endswith(".pkl"):
            path = (input_folder + "*.pkl").replace("**", "*")
            self.file_list = glob.glob(path)
        else:
            self.file_list = glob.glob(os.path.join(input_folder) + "/*")
        assert len(self.file_list) > 0, "No .pkl files found".format(input_folder)
        self.device = device
        self.len = len(self.file_list)
        self.get_pt = get_pt

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
        return self.len

    def transform_features(self, data):
        """
        Args:
            X (List): List of all the features
        """
        X = data[0]
        y = data[1]
        pts = data[2]
        all_data = [] # N_enevts x N_branches x N_features
        n_branches = len(X)
        n_events = X[0].shape[0]
        all_labels = []
        for i in range(n_events):
            event_data = []
            for j in range(n_branches):
                X_i = torch.from_numpy(X[j][i]).to(self.device).T
                event_data.append(X_i)
            y_i = torch.from_numpy(y[i]).to(self.device)
            pt_i = torch.from_numpy(np.asarray(pts[i])).to(self.device)
            all_data.append((event_data, y_i, pt_i))
        return all_data # N_events x N_branches x N_features

    def __getitem__(self, file_idx):
        """
        Returns the data for the given index.
        """
        # file_idx, idx = self.map_to_location(idx)
        # Read the data from the file
        with open(self.file_list[file_idx], "rb") as f:
            data = pickle.load(f)
        X = self.transform_features(data)
        return X


class DataLoader:
    """
    Data loader for the dataset.
    """
    def __init__(self, input_folder, num_workers=0, shuffle=True, drop_last=True, get_pt=True) -> None:
        super().__init__()
        self.input_folder = input_folder
        self.batch_size = 1
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.class_weights = []
        self.get_pt = get_pt

    def get_loader(self):
        """
        Returns the dataloader for the dataset.
        """
        dataset = CustomDS(self.input_folder, get_pt = self.get_pt)

        return dataset

class DataFileDS(Dataset):
    def __init__(self, jets_data):
        self.len_in = len(jets_data)
        self.X = [jet[0] for jet in jets_data]
        self.y = [jet[1] for jet in jets_data]
        self.pt = [jet[2] for jet in jets_data]

    def __len__(self):
        return self.len_in

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.pt[idx]
