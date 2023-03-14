"""
Utility functions for merging data from different .pkl files The final .pkl file will have the following structure: X, y = { "global_branches": {...}, "cpf_branches": {...}, "npf_branches": {...}, "vtx_branches": {...}, "cpf_pts_branches": {...}, "npf_pts_branches": {...}, "vtx_pts_branches": {...}, }, { "isB": [...], "isBB": [...], ... }
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import pickle

class DataMerger:
    def __init__(self,
                 files_to_merge: List[str],
                 out_path: str,
                 concat_rate: int
                 ) -> None:
        """Tool for merging all files all into one

        Args:
            files_to_merge (List[str]): List with the absolute path of the files to convert.
            out_path (str): Path to store the converted files.
            concat_rate (int): Number of files to keep into one final.
        """
        self.files_to_merge = files_to_merge
        self.out_path = out_path
        self.concat_rate = concat_rate

    def merge_n_files(files_in_list: str, out_path_name:str):
        dict_data = {
                "global_branches": [],
                "cpf_branches": [],
                "npf_branches": [],
                "vtx_branches": [],
                "cpf_pts_branches": [],
                "npf_pts_branches": [],
                "vtx_pts_branches": [],
            }
        ys = []
        pTs = []
        for file in files_in_list:
            with open(file, "rb") as f:
                data = pickle.load(f)

                for k, l in dict_data.items():
                    l.append(
                        data[0][k]
                    )
                ys.append(data[1])
                pTs.append(data[2])

        for k, l in dict_data.items():
            dict_data[k] = np.stack(l)

        ys = np.stack(ys)
        pTs = np.stack(pTs)

        v = (dict_data, ys, pTs)

        with open(out_path_name, "rb") as f:
            pickle.dump(v, f)

