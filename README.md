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

## Run the plot scripts

memory-consuming script (*): python3 nc_h5_compare_vis_temporal.py -d <folder-containing-training-data-files> -A

simple-but-inefficient script: python3 nc_h5_compare_vis.py -d <folder-containing-training-data-files> -A

## Run the generation scripts

python3 CMEMS_scenario.py -f \<folder-containing-training-data-files\>/file.txt -t \<int-simtime-days\> -dt \<int-simdt-minutes\> -ot \<int-writtendt-minutes\> -im \<'rk4'|'rk45'|'em'|'m1'\> -N \<equation-num-samples\> -sres \<arcdegree-ratio-uniform-sampledensity\> -gres \<arcdegree-ration-quadgrit-sample\> -sm \<sampling-distribution-scheme='regular_jitter'|'uniform'|'gaussian'|'triangular'|'vonmises'\>

python3 doublegyre_scenario.py -f <\folder-containing-training-data-files\>/file.txt -t \<int-simtime-days\> -dt \<int-simdt-minutes\> -ot \<int-writtendt-minutes\> -im \<'rk4'|'rk45'|'em'|'m1'\> -N \<equation-num-samples\> -sres \<arcdegree-ratio-uniform-sampledensity\> -gres \<arcdegree-ration-quadgrit-sample\> -sm \<sampling-distribution-scheme='regular_jitter'|'uniform'|'gaussian'|'triangular'|'vonmises'\>
