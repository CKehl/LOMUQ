# LOMUQ
Uncertainty Quantification of Lagrangian Ocean Models

The goal of this project is to use methods of pattern analysis- and replication, as well as Machine Learning, to quantify uncertainties in particle distributions and densities from trajectories emerging from Lagrangian ocean modelling (in our case: OceanParcels).

This reppository holds the required scripts to generate and plot the training data. For the plotting scripts, you need the following dependency packages installed:

- h5py
- xarray
- matplotlib
- Numpy

In order to run the data generation scripts, you'll additionally need:

- parcels (>= 2.2)
- scipy
- netcdf
