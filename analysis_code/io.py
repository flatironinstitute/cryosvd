import torch
import numpy as np
import os
import mrcfile
from natsort import natsorted
from typing import Tuple

def load_volumes(path_to_volumes: str) -> torch.Tensor:
    """
    Load volumes from a directory of .mrc files.

    Parameters
    ----------
    path_to_volumes : str
        Path to the directory containing the volumes.

    Returns
    -------
    torch.Tensor
        A tensor of shape (num_volumes, box_size^3) containing the volumes.
    """

    
    vol_files = []
    for file in os.listdir(path_to_volumes):
        if file.endswith(".mrc"):
            vol_files.append(file)
    vol_files = natsorted(vol_files)

    box_size = mrcfile.open(path_to_volumes + vol_files[0], mode='r', permissive=True).data.shape[-1]
    volumes = torch.zeros((len(vol_files), box_size * box_size * box_size))

    for i, file in enumerate(vol_files):
        with mrcfile.open(path_to_volumes + file, mode='r', permissive=True) as mrc:
            vol = mrc.data.copy()
            vol /= np.linalg.norm(vol)
            
            volumes[i] = torch.tensor(vol).flatten()

    return volumes

def load_methods_results(paths_to_methods: dict, mask: torch.Tensor) -> dict:
    """
    Load the results of the methods from the directories containing the volumes.

    Parameters
    ----------
    paths_to_methods : dict
        A dictionary containing the paths to the directories containing the volumes for each method.
    mask : torch.Tensor
        A tensor of shape (box_size, box_size, box_size) containing a binary mask to apply to the volumes.

    Returns
    -------
    dict:
        A dictionary that contains the SVD of the mean-removed volumes obtained by each method.
    """

    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor.")
    
    if len(mask.shape) == 3:
        mask = mask.flatten()[None, ...]

    elif len(mask.shape) == 4:
        assert mask.shape[0] == 1
        mask = mask.reshape(1, -1)
        raise Warning("mask is a 4D tensor. Assuming the first dimension is the batch dimension.")
    
    elif len(mask.shape) == 1:
        mask = mask[None, ...]

    elif len(mask.shape) == 2:
        assert mask.shape[0] == 1
        raise Warning("mask is a 2D tensor. Assuming the first dimension is the batch dimension.")

    else:
        raise ValueError("mask must be a 3D tensor or a 1D tensor. Got a tensor of shape {}.".format(mask.shape))

    methods = list(paths_to_methods.keys())
    methods_data = {}

    for method in methods:

        volumes = load_volumes(paths_to_methods[method]) * mask
        volumes -= volumes.mean(0, keepdim=True)

        u_matrices, singular_values, eigenvectors = torch.linalg.svd(volumes, full_matrices=False)

        methods_data[method] = {
            "u_matrices": u_matrices,
            "singular_values": singular_values,
            "eigenvectors": eigenvectors.T
        }

    return methods_data