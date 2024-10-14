<h1 align=`center`>CryoSVD</h1>

This repository contains a simple linear pipeline for the analysis of volume series output by heterogeneity software designed for the anlayis of single-particle Cryo-EM samples. It is based on Singular Value Decomposition (Singular Value Decomposition) of a series volumes, as well as an inspection of power power spectrum across the series. 

Details about CryoSVD construction can be found in the associated [publication](https://www.biorxiv.org/content/10.1101/2024.10.07.617120v1)

The data necessary to run the `compare_volume_series.ipynb` notebook can be found on [zenodo](https://zenodo.org/records/13900836).

# Installation 
  1. `git clone git@github.com/flatironinstitute/cryosvd`
  2. `conda create -n cryosvd python=3.10`
  3. `conda activate cryosvd`
  4. `cd cryosvd`
  5. `pip install .`
  6. Run the `compare_volume_series.ipynb` jupyter notebook.


