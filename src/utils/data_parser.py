import gc
import numpy as np
import uproot3 as u3
import uproot as u
import awkward as ak
import pandas as pd


def uproot_root2array(tree, stop=None, branches=None):
    dtypes = np.dtype([(b, np.dtype("O")) for b in branches])
    #    if isinstance(fname, list):
    #       fname = fname[0]
    # tree = u.open(fname)[treename]

    print("0", branches[0], fname)

    new_arr = np.empty(len(tree[branches[0]].array()), dtype=dtypes)

    for branch in branches:
        print(branch)
        new_arr[branch] = np.array(ak.to_list(tree[branch].array()), dtype="O")

    return new_arr


def uproot_tree_to_numpy(
    tree, inbranches_listlist, nMaxlist, nevents, stop=None, branches=None, flat=True
):

    # open the root file, tree and get the branches list wanted for building the array
    # tree  = u.open(fname)[treename]
    branches = tree.arrays(inbranches_listlist, library="numpy")
    # branches = [tree[branch_name].array() for branch_name in inbranches_listlist]

    # Initialize the output_array with the correct dimension and 0s everywhere. We will fill the correct
    if nMaxlist == 1:
        output_array = np.zeros(shape=(nevents, len(inbranches_listlist)))

        # Loop and fill our output_array
        for i in range(nevents):
            for j, branch in enumerate(inbranches_listlist):
                output_array[i, j] = branches[branch][i]

    if nMaxlist > 1:
        output_array = np.zeros(shape=(nevents, len(inbranches_listlist), nMaxlist))

        # Loop and fill w.r.t. the zero padding method our output_array
        for i in range(nevents):
            lenght = len(branches[inbranches_listlist[0]][i])
            for j, branch in enumerate(inbranches_listlist):
                if lenght >= nMaxlist:
                    output_array[i, j, :] = branches[branch][i][:nMaxlist]
                if lenght < nMaxlist:
                    output_array[i, j, :lenght] = branches[branch][i]

        output_array = np.transpose(output_array, (0, 2, 1))

    ### Debugging lines ###
    print(output_array.shape)
    # print(output_array[:3,0])

    return


def uproot_MeanNormZeroPad(
    Filename_in, MeanNormTuple, inbranches_listlist, nMaxslist, nevents
):
    # savely copy lists (pass by value)
    import copy

    inbranches_listlist = copy.deepcopy(inbranches_listlist)
    nMaxslist = copy.deepcopy(nMaxslist)

    # Read in total number of events
    totallengthperjet = 0
    for i in range(len(nMaxslist)):
        if nMaxslist[i] >= 0:
            totallengthperjet += len(inbranches_listlist[i]) * nMaxslist[i]
        else:
            totallengthperjet += len(inbranches_listlist[i])  # flat branch

    print("Total event-length per jet: {}".format(totallengthperjet))

    # shape could be more generic here... but must be passed to c module then
    array = numpy.zeros((nevents, totallengthperjet), dtype="float32")

    # filling mean and normlist
    normslist = []
    meanslist = []
    for inbranches in inbranches_listlist:
        means = []
        norms = []
        for b in inbranches:
            if MeanNormTuple is None:
                means.append(0)
                norms.append(1)
            else:
                means.append(MeanNormTuple[b][0])
                norms.append(MeanNormTuple[b][1])
        meanslist.append(means)
        normslist.append(norms)

    # now start filling the array


