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
# from vandy_taggers.utils.weighter import Weighter
import sys
sys.path.append("/home/jose/Documents/WORKS/CERN/InProgress/OwnArch/vandy-ml-tagger/vandy_taggers/utils")
from weighter import Weighter

"""
Inpired by:
https://github.com/jet-universe/particle__mer/blob/main/utils/convert_qg_datasets.py
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
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(X, str):
            print("X is a string", X)
        if len(X.shape) == 1:
            if X.shape[0] < max_len:
                if len(X) > 0:

                        X = np.pad(
                            X,
                            (0, max_len - X.shape[0]),
                            mode="mean",
                        )
                else:
                    X = np.zeros(max_len)
            elif X.shape[0] > max_len:
                X = X[:max_len]
        elif len(X.shape) == 2:
            if X.shape[0] < max_len and X.shape[1] > 0:
                X = np.pad(
                    X,
                    ((0, max_len - X.shape[0]), (0, 0)),
                    mode="mean",
                )
            elif X.shape[0] < max_len:
                X = np.zeros((max_len, X.shape[1]))
            else:
                X = X[:max_len, :]
        return X

    @staticmethod
    def reduce_truth(uproot_arrays):

        b = uproot_arrays['isB']

        bb = uproot_arrays['isBB']
        gbb = uproot_arrays['isGBB']

        bl = uproot_arrays['isLeptonicB']
        blc = uproot_arrays['isLeptonicB_C']
        lepb = bl+blc

        c = uproot_arrays['isC']
        cc = uproot_arrays['isCC']
        gcc = uproot_arrays['isGCC']

        ud = uproot_arrays['isUD']
        s = uproot_arrays['isS']
        uds = ud+s

        g = uproot_arrays['isG']


        return np.vstack((b,bb+gbb,lepb,c+cc+gcc,uds,g)).transpose()

    def parse_labels(self, y: np.array) -> np.array:
        """Parse the labels into a format suitable for training.
        y: dict coltaining classes as one hot encoded vectors
        """
        return np.stack(y).T

    def data_concat(self, X_branch: List[np.array], n_pad: int=None) -> np.ndarray:
        # X_branch is a list of of arrays [BATCH, N_FEATS, N_DEPCTIONS]
        branches = []
        for sub_branch in X_branch:
            if n_pad:
                branches.append(
                    np.stack([self._pad_data(x, n_pad) for x in sub_branch if isinstance(x, np.ndarray)])
                )
            else:
                branches.append(
                    np.stack(
                        x for x in sub_branch if isinstance(x, np.ndarray)
                    )
                )
        return np.stack(branches).astype(dtype='float32', order='C')

    def detect_nan(self, X):
        if np.isnan(X).any():
            print("NAN DETECTED")
            return True
        return False

    def _transform(self, X, y, pT, start=0, stop=-1) -> Tuple[awkward0.JaggedArray, np.ndarray]:
        """Transform the data into a format suitable for training.
        X: dict of numpy arrays containing the input features
        y: array containing the target labels
        """
        # Transform truth branches
        ground_truth = self.reduce_truth(y)
        ground_truth = np.where(np.isnan(ground_truth), 0, ground_truth)
        sum_prbs = np.sum(ground_truth, axis=1)
        mask = sum_prbs > 0
        pT = pT[mask]
        ground_truth = ground_truth[mask]

        # Transform cpf branches
        cpf_np  = self.data_concat(X["cpf_branches"], self.n_cpf).transpose(1, 0, 2)
        cpf_np = cpf_np[mask]
        # Transform npf branches
        npf_np  = self.data_concat(X["npf_branches"], self.n_npf).transpose(1, 0, 2)
        npf_np = npf_np[mask]
        # Transform vtx branches
        vtx_np = self.data_concat(X["vtx_branches"], self.n_vtx).transpose(1, 0, 2)
        vtx_np = vtx_np[mask]
        # Transform cpf pts branches
        cpf_pts_np = self.data_concat(X["cpf_pts_branches"], self.n_cpf).transpose(1, 0, 2)
        cpf_pts_np = cpf_pts_np[mask]
        # Transform npf pts branches
        npf_pts_np = self.data_concat(X["npf_pts_branches"], self.n_npf).transpose(1, 0, 2)
        npf_pts_np = npf_pts_np[mask]
        # Transform vtx pts branches
        vtx_pts_np = self.data_concat(X["vtx_pts_branches"], self.n_vtx).transpose(1, 0, 2)
        vtx_pts_np = vtx_pts_np[mask]

        cpf_np = np.where(np.isnan(cpf_np), 0, cpf_np)
        npf_np = np.where(np.isnan(npf_np), 0, npf_np)
        vtx_np = np.where( np.isnan(vtx_np), 0, vtx_np)
        cpf_pts_np = np.where(np.isnan(cpf_pts_np), 0, cpf_pts_np)
        npf_pts_np = np.where(np.isnan(npf_pts_np), 0, npf_pts_np)
        vtx_pts_np = np.where(np.isnan(vtx_pts_np), 0, vtx_pts_np)


        X = (
            cpf_np,
            npf_np,
            vtx_np,
            cpf_pts_np,
            npf_pts_np,
            vtx_pts_np,
        )

        return X, ground_truth, pT

    def convert_single_file(self, sourcefile, destdir, basename, idx):

        # Open a root file in "read" mode
        print("Converting ", sourcefile.split("/")[-1])
        with u.open(sourcefile) as f:
            file_data = f["deepntuplizer;1"]["tree;1"]

            keys = list(file_data.keys())

            pT = file_data["jet_pt"].array(library="np")
            if self.remove:
                b = [self.weightbranchX,self.weightbranchY]
                b.extend(self.truth_branches)
                b.extend(self.undefTruth)
                for_remove = file_data.arrays(b, library = 'pd')
            if self.weighter_o is not None:
                notremoves = self.weighter_o['weigther'].createNotRemoveIndices(for_remove, use_uproot = True)
                undef = for_remove['isUndefined']
                notremoves -= undef
                pu = for_remove['isPU']
                notremoves -= pu
            else:
                notremoves = np.ones(len(pT), dtype = bool)

            # Truth branches
            reduced_truth = {
                k: file_data[k].array(library="np")
                for k in self.truth_branches
            }

            # Vertex branches
            global_branches = [
                file_data[k].array(library="np")
                for k in self.global_branches
            ]

            # cpf_branches
            cpf_branches = [
                file_data[k].array(library="np")
                for k in self.cpf_branches
            ]

            # npf_branches
            npf_branches = [
                file_data[k].array(library="np")
                for k in self.npf_branches
            ]

            # vtx_branches
            vtx_branches = [
                file_data[k].array(library="np")
                for k in self.vtx_branches
            ]

            # cpf_pts_branches
            cpf_pts_branches = [
                file_data[k].array(library="np")
                for k in self.cpf_pts_branches
            ]

            # npf_pts_branches
            npf_pts_branches = [
                file_data[k].array(library="np")
                for k in self.npf_pts_branches
            ]

            # vtx_pts_branches
            vtx_pts_branches = [
                file_data[k].array(library="np")
                for k in self.vtx_pts_branches
            ]

            if isinstance(notremoves, pd.DataFrame):
                notremoves = list(notremoves.astype(int).values.flatten())
            notremoves = np.array(notremoves)
            mask = notremoves > 0
            if self.remove:
                print('remove')
                global_branches = [np.ma.masked_where(mask, global_branches[i]) for i in range(len(global_branches))]
                cpf_branches = [np.ma.masked_where(mask, cpf_branches[i]) for i in range(len(cpf_branches))]
                npf_branches = [np.ma.masked_where(mask, npf_branches[i]) for i in range(len(npf_branches))]
                vtx_branches = [np.ma.masked_where(mask, vtx_branches[i]) for i in range(len(vtx_branches))]
                cpf_pts_branches = [np.ma.masked_where(mask, cpf_pts_branches[i]) for i in range(len(cpf_pts_branches))]
                npf_pts_branches = [np.ma.masked_where(mask, npf_pts_branches[i]) for i in range(len(npf_pts_branches))]
                vtx_pts_branches = [np.ma.masked_where(mask, vtx_pts_branches[i]) for i in range(len(vtx_pts_branches))]
                reduced_truth = {k: np.ma.masked_where(mask, reduced_truth[k]) for k in reduced_truth}
                #reduced_truth={k: reduced_truth[k][notremoves > 0] for k in reduced_truth}
                # pT = [pT[i] for i in range(len(pT)) if mask[i] == False]

            #newnsamp = global_branches

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
        v = self._transform(X, y, pT)
        with open(output, "wb") as f:
            pickle.dump(v, f)

    def convert(self, sourcelist, destdir, basename, is_train=True):
        """
        Parralelized conversion of a list of files into pickle files.
        source: path to a single file or a directory of files
        dest: path to the output directory
        """
        if "test" in basename or not is_train:
            self.weighter_o = None
            self.remove = False
        elif is_train or "train" in basename:
            self.remove = True
            self.weighter_o = self.weighter(sourcelist)
        else:
            self.remove = False
            self.weighter_o = None
        files = self.natural_sort(sourcelist)
        if not os.path.exists (destdir):
            os.makedirs(destdir)
        Parallel(n_jobs = 20)(
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

    def weighter(self, allsourcefiles: list):
        weighter = Weighter()
        weighter.undefTruth = self.undefTruth
        weighter.class_weights = self.class_weights
        branches = [self.weightbranchX,self.weightbranchY]
        branches.extend(self.truth_branches)

        if self.remove:
            weighter.setBinningAndClasses(
                [self.weight_binX,self.weight_binY],
                self.weightbranchX,self.weightbranchY,
                self.truth_branches, self.red_classes,
                self.truth_red_fusion, method = self.referenceclass
            )

        counter=0
        if self.remove:
            for fname in allsourcefiles:
                events = u.open(fname)["deepntuplizer/tree"]
                nparray = events.arrays(branches, library = 'np')
                keys = list(nparray.keys())
                for k in range(len(branches)):
                    nparray[branches[k]] = nparray.pop(keys[k])
                nparray = pd.Series(nparray)
                norm_hist = True
                if self.referenceclass == 'flatten':
                    norm_hist = False
                weighter.addDistributions(nparray, norm_h = norm_hist)
                #del nparray
                counter=counter+1
            weighter.createRemoveProbabilitiesAndWeights(self.referenceclass)
            return {'weigther':weighter}