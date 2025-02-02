############ Import packages

import numpy as np
import numpy.random as npr
import copy
import time

import torch
from torch.utils.data import Dataset
from sklearn.metrics import r2_score



############ Utilities

def get_sample_weights(Y,eps=.1):

    """
    Function for getting weights for each sample (So different time points can be upweighted/downweighted in the cost function)
    The weights are inversely related to the norm of the activity across all neurons

    Parameters
    ----------
    Y: neural data
        numpy 2d array of shape [n_time,n_output_neurons]
    eps: a small offset that limits the maximal sample weight of a time point (in the scenario there is zero activity)
        scalar

    Returns
    -------
    The sample weights - an array of shape [n_time,1]

    """


    tmp=1/(np.sqrt(np.sum(Y**2,axis=1))+eps)
    tmp2=tmp/np.mean(tmp)
    return tmp2[:,None]


def torchify(array_list):

    """
    Function that turns a list of arrays into a list of torch tensors.

    Parameters
    ----------
    array_list: a list of numpy arrays

    Returns
    -------
    a list of torch tensors (corresponding to the original arrays)

    """
    # use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return [torch.tensor(array_entry, dtype=torch.float).to(device) for array_entry in array_list]

def dict_torchify(dict_array_list):
    for i, elem in enumerate(dict_array_list):
        dict_array_list[i] = {k: torchify(v) for k, v in elem.items()}
    return dict_array_list


def get_accuracy(self,X,sample_weight=None):

    Xhat = self.reconstruct(X)
    if sample_weight is None:
        r2=r2_score(X,Xhat,multioutput='variance_weighted')
        reconstruction_loss=np.sum(((Xhat - X))**2)
    else:
        r2=r2_score(X,Xhat,sample_weight=sample_weight,multioutput='variance_weighted')
        reconstruction_loss=np.sum((sample_weight*(Xhat - X))**2)

    return [r2,reconstruction_loss]


def concatenate_region_dict(X):
    return np.concatenate([np.concatenate(v, axis=0) for _, v in X.items()], axis=1)


def list_of_dicts(X):
    """TODO: probably a simpler way to write this"""
    trial_data = list(zip(*[v for _, v in X.items()]))
    return [{k: trial[i] for i, k in enumerate(X.keys())} for trial in trial_data]

def combine_dict_list(X, filter_length=0):
    combined = {k: [] for k, _ in X[0].items()}
    if filter_length:
        [combined[k].append(trim(v, filter_length)) for list_item in X  for k, v in list_item.items()]
    else:
        [combined[k].append(v) for list_item in X  for k, v in list_item.items()]

    return {k: torch.cat(v) for k, v in combined.items()}

def combine_region_dict(X):
    return torch.cat([v for _, v in X.items()], axis=1).detach().numpy()

def trim(X, filter_length):
    return X[filter_length-1:-filter_length+1]