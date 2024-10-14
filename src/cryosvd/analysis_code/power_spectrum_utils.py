import torch
import tqdm
import numpy as np 
from typing import Tuple

"""
This module contains utility functions to compute the power spectrum of a volume. This was adapted from the code provided in ASPIRE (https://github.com/ComputationalCryoEM/ASPIRE-Python/) and modified to work with PyTorch tensors.
"""


def _cart2sph(x, y, z):
    """
    Converts a grid in cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x: torch.tensor
        x-coordinate of the grid.
    y: torch.tensor
        y-coordinate of the grid.
    z: torch.tensor
    """
    hxy = torch.hypot(x, y)
    r = torch.hypot(hxy, z)
    el = torch.atan2(z, hxy)
    az = torch.atan2(y, x)
    return az, el, r


def _grid_3d(n, dtype=torch.float32):
    """
    Generates a centered 3D grid. The grid is given in both cartesian and spherical coordinates.

    Parameters
    ----------
    n: int
        Size of the grid.
    dtype: torch.dtype
        Data type of the grid.

    Returns
    -------
    grid: dict
        Dictionary containing the grid in cartesian and spherical coordinates.
        keys: x, y, z, phi, theta, r
    """
    start = -n // 2 + 1
    end = n // 2

    if n % 2 == 0:
        start -= 1 / 2
        end -= 1 / 2

    grid = torch.linspace(start, end, n, dtype=dtype)
    z, x, y = torch.meshgrid(grid, grid, grid, indexing="ij")

    phi, theta, r = _cart2sph(x, y, z)

    theta = torch.pi / 2 - theta

    return {"x": x, "y": y, "z": z, "phi": phi, "theta": theta, "r": r}


def _centered_fftn(x, dim=None):
    """
    Wrapper around torch.fft.fftn that centers the Fourier transform.
    """
    x = torch.fft.fftn(x, dim=dim)
    x = torch.fft.fftshift(x, dim=dim)
    return x


def _centered_ifftn(x, dim=None):
    """
    Wrapper around torch.fft.ifftn that centers the inverse Fourier transform.
    """
    x = torch.fft.fftshift(x, dim=dim)
    x = torch.fft.ifftn(x, dim=dim)
    return x


def _average_over_single_shell(shell_index, volume, radii, shell_width=0.5):
    """
    Given a volume in Fourier space, compute the average value of the volume over a shell.

    Parameters
    ----------
    shell_index: int
        Index of the shell in Fourier space.
    volume: torch.tensor
        Volume in Fourier space.
    radii: torch.tensor
        Radii of the Fourier space grid.
    shell_width: float
        Width of the shell.

    Returns
    -------
    average: float
        Average value of the volume over the shell.
    """
    inner_diameter = shell_width + shell_index
    outer_diameter = shell_width + (shell_index + 1)
    mask = (radii > inner_diameter) & (radii < outer_diameter)
    return torch.sum(mask * volume) / torch.sum(mask)


def _average_over_shells(volume_in_fourier_space, shell_width=0.5):
    """
    Vmap wrapper over _average_over_single_shell to compute the average value of a volume in Fourier space over all shells. The input should be a volumetric quantity in Fourier space.

    Parameters
    ----------
    volume_in_fourier_space: torch.tensor
        Volume in Fourier space.

    Returns
    -------
    radial_average: torch.tensor
        Average value of the volume over all shells.
    """
    L = volume_in_fourier_space.shape[0]
    dtype = torch.float32
    radii = _grid_3d(L, dtype=dtype)["r"]

    radial_average = torch.vmap(
        _average_over_single_shell, in_dims=(0, None, None, None)
    )(torch.arange(0, L // 2), volume_in_fourier_space, radii, shell_width)

    return radial_average


def compute_power_spectrum_per_volume(volumes, box_size, shell_width=0.5):
    """
    Compute the power spectrum of a volume.

    Parameters
    ----------
    volume: torch.tensor
        Volume for which to compute the power spectrum.
    shell_width: float
        Width of the shell.

    Returns
    -------
    power_spectrum: torch.tensor
        Power spectrum of the volume.

    Examples
    --------
    volume = mrcfile.open("volume.mrc").data.copy()
    volume = torch.tensor(volume, dtype=torch.float32)
    power_spectrum = compute_power_spectrum(volume)
    """

    # Compute centered Fourier transforms.
    
    n_volumes = len(volumes)
    volumes = volumes.reshape(n_volumes,box_size,box_size,box_size)
    vol_ffts = [torch.abs(_centered_fftn(volume)) ** 2 for volume in volumes]
    power_spectrums = [_average_over_shells(vol_fft, shell_width=shell_width) for vol_fft in vol_ffts]

    return power_spectrums

def compute_power_spectrums (methods_data: dict, pixel_size) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Compute the distance matrix between the methods using the relative capture variance.

    Parameters
    ----------
    methods_data : dict
        A dictionary containing the SVD of the mean-removed volumes obtained by each method.

    Returns
    -------
    Tuple[torch.Tensor, np.ndarray]
        A tuple containing the power spectrums for each volume series and the named labels.
    """
    n_subs = len(list(methods_data.keys()))
    labels = list(methods_data.keys())
    n_volumes = methods_data[labels[0]]["volumes"].shape[0]
    box_size = int(np.round(methods_data[labels[0]]["volumes"][0].shape[0]**(1/3)))
    print(box_size)
    frequencies = np.fft.fftshift(np.fft.fftfreq(box_size,pixel_size)) 
    frequencies = frequencies[frequencies >= 0]

    power_spectrums = torch.zeros(n_subs,n_volumes,box_size)
    

    power_spectrums = [compute_power_spectrum_per_volume(methods_data[method]['volumes'] +  methods_data[method]['mean_volume'],box_size) for method in tqdm.tqdm(labels) ]

    results = {}
    for i,method in enumerate (labels):
        results[method] = {
            "power_spectrums" : torch.stack(power_spectrums[i]),
            "frequencies" : frequencies
        }

    return results

