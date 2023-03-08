import os
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
import awkward0
# from uproot3_methods import TLorentzVectorArray
import awkward as ak
import argparse
import gc
from uproot3_methods import TLorentzVectorArray
import numpy as np
import uproot3 as u3
import uproot as u
import awkward as ak
import pandas as pd
import pickle
from joblib import Parallel, delayed

"""
Modification of:
https://github.com/jet-universe/particle_transformer/blob/main/utils/convert_qg_datasets.py
https://github.com/AlexDeMoor/DeepJet/blob/master/modules/datastructures/TrainData_deepFlavour.py#L1
"""
class PartData:
    def __init__(self) -> None:
        self.truth_branches = [
            "isB",
            "isBB",
            "isGBB",
            "isLeptonicB",
            "isLeptonicB_C",
            "isC",
            "isGCC",
            "isCC",
            "isUD",
            "isS",
            "isG",
        ]
        self.undefTruth = ["isUndefined", "isPU"]
        self.weightbranchX = "jet_pt"
        self.weightbranchY = "jet_eta"
        self.remove = True
        self.referenceclass = "isB"  #'flatten'  #Choose 'flatten' for flat or one of the truth branch for ref
        self.red_classes = [
            "cat_B",
            "cat_C",
            "cat_UDS",
            "cat_G",
        ]  # reduced classes (flat only)
        self.truth_red_fusion = [
            ("isB", "isBB", "isGBB", "isLeptonicB", "isLeptonicB_C"),
            ("isC", "isGCC", "isCC"),
            ("isUD", "isS"),
            ("isG"),
        ]  # Indicates how you are making the fusion of your truth branches to the reduced classes for the flat reweighting
        self.class_weights = np.array(
            [1.00, 1.00, 2.5, 5.0], dtype=float
        )  # Ratio between our reduced classes (flat only)
        self.weight_binX = np.array(
            [15, 20, 26, 35, 46, 61, 80, 106, 141, 186, 247, 326, 432, 571, 756, 1000],
            dtype=float,
        )  # Flat reweighting
        self.weight_binY = np.array(
            [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=float
        )  # Flat reweighting

        self.global_branches = [
            "jet_pt",
            "jet_eta",
            "nCpfcand",
            "nNpfcand",
            "nsv",
            "npv",
            "TagVarCSV_trackSumJetEtRatio",
            "TagVarCSV_trackSumJetDeltaR",
            "TagVarCSV_vertexCategory",
            "TagVarCSV_trackSip2dValAboveCharm",
            "TagVarCSV_trackSip2dSigAboveCharm",
            "TagVarCSV_trackSip3dValAboveCharm",
            "TagVarCSV_trackSip3dSigAboveCharm",
            "TagVarCSV_jetNSelectedTracks",
            "TagVarCSV_jetNTracksEtaRel",
        ]

        self.cpf_branches = [
            "Cpfcan_BtagPf_trackEtaRel",
            "Cpfcan_BtagPf_trackPtRel",
            "Cpfcan_BtagPf_trackPPar",
            "Cpfcan_BtagPf_trackDeltaR",
            "Cpfcan_BtagPf_trackPParRatio",
            "Cpfcan_BtagPf_trackSip2dVal",
            "Cpfcan_BtagPf_trackSip2dSig",
            "Cpfcan_BtagPf_trackSip3dVal",
            "Cpfcan_BtagPf_trackSip3dSig",
            "Cpfcan_BtagPf_trackJetDistVal",
            "Cpfcan_ptrel",
            "Cpfcan_drminsv",
            "Cpfcan_VTX_ass",
            "Cpfcan_puppiw",
            "Cpfcan_chi2",
            "Cpfcan_quality",
        ]

        self.n_cpf = 25

        self.npf_branches = [
            "Npfcan_ptrel",
            "Npfcan_etarel",
            "Npfcan_phirel",
            "Npfcan_deltaR",
            "Npfcan_isGamma",
            "Npfcan_HadFrac",
            "Npfcan_drminsv",
            "Npfcan_puppiw",
        ]
        self.n_npf = 25

        self.vtx_branches = [
            "sv_pt",
            "sv_deltaR",
            "sv_mass",
            "sv_etarel",
            "sv_phirel",
            "sv_ntracks",
            "sv_chi2",
            "sv_normchi2",
            "sv_dxy",
            "sv_dxysig",
            "sv_d3d",
            "sv_d3dsig",
            "sv_costhetasvpv",
            "sv_enratio",
        ]

        self.n_vtx = 5

        self.cpf_pts_branches = ["Cpfcan_pt", "Cpfcan_eta", "Cpfcan_phi", "Cpfcan_e"]

        self.npf_pts_branches = ["Npfcan_pt", "Npfcan_eta", "Npfcan_phi", "Npfcan_e"]

        self.vtx_pts_branches = ["sv_pt", "sv_eta", "sv_phi", "sv_e"]

        self.reduced_truth = ["isB", "isBB", "isLeptonicB", "isC", "isUDS", "isG"]

    @staticmethod
    def _pad_data(X: np.array, max_len: int) -> np.array:
        """Pad the data to the maximum length of the batch.
        X: 1D numpy array containing the data

        """
        try:
            if X.shape[0] < max_len:
                X = np.pad(
                    X,
                    ((0, max_len - X.shape[0]), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
            elif X.shape[0] > max_len:
                X = X[:max_len]
        except:
            print(X)

    def _transform(self, X, y, start=0, stop=-1) -> Tuple[awkward0.JaggedArray, np.ndarray]:
        """Transform the data into a format suitable for training.
        X: dict of numpy arrays containing the input features
        y: array containing the target labels
        """
        # Transform global branches
        # TODO
        # Transform cpf branches
        for k in X["cpf_branches"].keys():
            X["cpf_branches"][k] = self._pad_data(X["cpf_branches"][k], self.n_cpf)
        # Transform npf branches
        for k in X["npf_branches"].keys():
            X["npf_branches"][k] = self._pad_data(X["npf_branches"][k], self.n_npf)
        # Transform vtx branches
        for k in X["vtx_branches"].keys():
            X["vtx_branches"][k] = self._pad_data(X["vtx_branches"][k], self.n_vtx)
        # Transform cpf pts branches
        for k in X["cpf_pts_branches"].keys():
            X["cpf_pts_branches"][k] = self._pad_data(X["cpf_pts_branches"][k], self.n_cpf)
        # Transform npf pts branches
        for k in X["npf_pts_branches"].keys():
            X["npf_pts_branches"][k] = self._pad_data(X["npf_pts_branches"][k], self.n_npf)
        # Transform vtx pts branches
        for k in X["vtx_pts_branches"].keys():
            X["vtx_pts_branches"][k] = self._pad_data(X["vtx_pts_branches"][k], self.n_vtx)

        # Transform truth branches
        # TODO
        return X, y

    def convert_single_file(self, sourcefile, destdir, basename, idx):
        # Open a root file in "read" mode
        print("Converting ", sourcefile.split("/")[-1])
        with u.open(sourcefile) as f:
            file_data = f["deepntuplizer;1"]["tree;1"]

            keys = list(file_data.keys())

            # Truth branches
            reduced_truth = {
                k: file_data[k].array(library="np")
                for k in self.truth_branches
            }

            # Vertex branches
            global_branches = {
                k: file_data[k].array(library="np")
                for k in self.global_branches
            }

            # cpf_branches
            cpf_branches = {
                k: file_data[k].array(library="np")
                for k in self.cpf_branches
            }

            # npf_branches
            npf_branches = {
                k: file_data[k].array(library="np")
                for k in self.npf_branches
            }

            # vtx_branches
            vtx_branches = {
                k: file_data[k].array(library="np")
                for k in self.vtx_branches
            }

            # cpf_pts_branches
            cpf_pts_branches = {
                k: file_data[k].array(library="np")
                for k in self.cpf_pts_branches
            }

            # npf_pts_branches
            npf_pts_branches = {
                k: file_data[k].array(library="np")
                for k in self.npf_pts_branches
            }

            # vtx_pts_branches
            vtx_pts_branches = {
                k: file_data[k].array(library="np")
                for k in self.vtx_pts_branches
            }

            X = {
                "global_branches": global_branches,
                "cpf_branches": cpf_branches,
                "npf_branches": npf_branches,
                "vtx_branches": vtx_branches,
                "cpf_pts_branches": cpf_pts_branches,
                "npf_pts_branches": npf_pts_branches,
                "vtx_pts_branches": vtx_pts_branches,
            }
            y = reduced_truth

        output = os.path.join(destdir, "%s_%d.pkl" % (basename, idx))
        if os.path.exists(output):
            os.remove(output)
        v = self._transform(X, y)
        with open(output, "wb") as f:
            pickle.dump(v, f)

    def convert(self, sourcelist, destdir, basename):
        """
        Parralelized conversion of a list of files into pickle files.
        source: path to a single file or a directory of files
        dest: path to the output directory

        """
        files = self.natural_sort(sourcelist)
        if not os.path.exists(destdir):
            os.makedirs(destdir)
        Parallel(n_jobs=20)(
            delayed(self.convert_single_file)(sourcefile, destdir, basename, idx)
            for idx, sourcefile in enumerate(files)
        )

    @staticmethod
    def natural_sort(l: list):
        import re
        def convert(text):
            return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key):
            return [convert(c) for c in re.split("([0-9]+)", key)]
        return sorted(l, key=alphanum_key)