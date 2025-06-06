"""
Author: Dr. Christian Kehl
Date: 11-02-2020
"""

from parcels import AdvectionEE, AdvectionRK45, AdvectionRK4, AdvectionDiffusionEM, AdvectionDiffusionM1, DiffusionUniformKh
from parcels import FieldSet, ScipyParticle, JITParticle, Variable
try:
    from parcels import StateCode, OperationCode, ErrorCode
except:
    from parcels import StatusCode
from parcels.particleset import ParticleSet as DryParticleSet
from parcels.particleset import BenchmarkParticleSet
from parcels.field import Field, VectorField, NestedField
from parcels.grid import RectilinearZGrid
from parcels import ParcelsRandom, logger
from datetime import timedelta, datetime
import math
from argparse import ArgumentParser
import numpy as np
from numpy.random import default_rng
from dask.diagnostics import ProgressBar as DaskProgressBar
import fnmatch
import sys
import gc
import os
import time as ostime
# from scipy.interpolate import interpn
from glob import glob

import xarray as xr
import dask.array as da
from netCDF4 import Dataset
import h5py

try:
    from mpi4py import MPI
except:
    MPI = None
with_GC = False
DBG_MSG = False

pset = None
ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
method = {'RK4': AdvectionRK4, 'EE': AdvectionEE, 'RK45': AdvectionRK45}
global_t_0 = 0
Nparticle = int(math.pow(2,10)) # equals to Nparticle = 1024

# ---- total extents ---- #
# xs = -16.0  # arc degree
# xe = 13.0  # arc degree
# ys = 46.0  # arc degree
# ye = 62.74  # arc degree

# ------ entire ENWS ------ #
# xs = -6.25  # arc degree
# xe = 12.75  # arc degree
# ys = 50.0  # arc degree
# ye = 60.0  # arc degree
# ------ DutchCoast ------ #
xs = 2.5394  # arc degree
xe = 7.2084  # arc degree
ys = 51.262  # arc degree
ye = 55.765  # arc degree

tsteps = 0
tstepsize = 0
ttotal = 1.0

# we need to modify the kernel.execute / pset.execute so that it returns from the JIT
# in a given time WITHOUT writing to disk via outfie => introduce a pyloop_dt


# -------------------------------------------------------------------------------------------------------------------- #

def time_index_value(tx, _ft, _ft_dt=None):
    # expect ft to be forward-linear
    ft = _ft
    if isinstance(_ft, xr.DataArray):
        ft = ft.data
    f_dt = _ft_dt
    if f_dt is None:
        f_dt = ft[1] - ft[0]
        if type(f_dt) not in [np.float64, np.float32]:
            f_dt = timedelta(f_dt).total_seconds()
    # f_interp = (f_min + tx) / f_dt if f_dt >= 0 else (f_max + tx) / f_dt
    f_interp = tx / f_dt
    ti = int(math.floor(f_interp))
    ti = max(0, min(ft.shape[0]-1, ti))
    return ti


def time_partion_value(tx, _ft, _ft_dt=None):
    # expect ft to be forward-linear
    ft = _ft
    if isinstance(_ft, xr.DataArray):
        ft = ft.data
    f_dt = _ft_dt
    if f_dt is None:
        f_dt = ft[1] - ft[0]
        if type(f_dt) not in [np.float64, np.float32]:
            f_dt = timedelta(f_dt).total_seconds()
    # f_interp = (f_min + tx) / f_dt if f_dt >= 0 else (f_max + tx) / f_dt
    f_interp = abs(tx / f_dt)
    f_interp = max(0.0, min(float(ft.shape[0]-1), f_interp))
    f_t = f_interp - math.floor(f_interp)
    return f_t

# Helper function for time-conversion from the calendar format
def convert_timearray(t_array, dt_minutes, ns_per_sec, debug=False, array_name="time array"):
    """

    :param t_array: 2D array of time values in either calendar- or float-time format; dim-0 = object entities, dim-1 = time steps (or 1D with just timesteps)
    :param dt_minutes: expected delta_t als float-value (in minutes)
    :param ns_per_sec: conversion value of number of nanoseconds within 1 second
    :param debug: parameter telling to print debug messages or not
    :param array_name: name of the array (for debug prints)
    :return: converted t_array
    """
    ta = t_array
    while len(ta.shape) > 1:
        ta = ta[0]
    if isinstance(ta[0], datetime) or isinstance(ta[0], timedelta) or isinstance(ta[0], np.timedelta64) or isinstance(ta[0], np.datetime64) or np.float64(ta[1]-ta[0]) > (dt_minutes+dt_minutes/2.0):
        if debug:
            print("{}.dtype before conversion: {}".format(array_name, t_array.dtype))
        t_array = (t_array / ns_per_sec).astype(np.float64)
        ta = (ta / ns_per_sec).astype(np.float64)
        if debug:
            print("{0}.range and {0}.dtype after conversion: ({1}, {2}) \t {3}".format(array_name, ta.min(), ta.max(), ta.dtype))
    else:
        if debug:
            print("{0}.range and {0}.dtype: ({1}, {2}) \t {3} \t(no conversion applied)".format(array_name, ta.min(), ta.max(), ta.dtype))
        pass
    return t_array

def DeleteParticle(particle, fieldset, time):
    if particle.valid < 0:
        particle.valid = 0
    particle.delete()

def perIterGC():
    gc.collect()

def create_ENWS_fieldset(datahead, periodic_wrap, period, chunk_level=0, anisotropic_diffusion=False):
    currentshead = os.path.join(datahead, "currents")
    print(currentshead)
    stokeshead  = os.path.join(datahead, "waves")
    ENWS_files = sorted(glob(os.path.join(currentshead, "metoffice_foam1_amm7_NWS_SSC_hi2022*.nc")))
    currents_files = {'U': ENWS_files, 'V': ENWS_files}
    currents_variables = {'U': 'uo', 'V': 'vo'}
    currents_dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    currents_nchs = None
    if chunk_level > 1:
        currents_nchs = {
            'U': {'lon': ('longitude', 16), 'lat': ('latitude', 16), 'time': ('time', 1)},  #
            'V': {'lon': ('longitude', 16), 'lat': ('latitude', 16), 'time': ('time', 1)},  #
        }
    elif chunk_level > 0:
        currents_nchs = {
            'U': 'auto',
            'V': 'auto'
        }
    else:
        currents_nchs = False

    # Kh_zonal = np.ones(U.shape, dtype=np.float32) * 0.5  # ?
    # Kh_meridional = np.ones(U.shape, dtype=np.float32) * 0.5  # ?
    # fieldset.add_constant_field("Kh_zonal", 1, mesh="flat")
    # fieldset.add_constant_field("Kh_meridonal", 1, mesh="flat")
    global ttotal
    ttotal = period  # days
    fieldset = None
    if periodic_wrap:
        fieldset = FieldSet.from_netcdf(currents_files, currents_variables, currents_dimensions, chunksize=currents_nchs, time_periodic=timedelta(days=ttotal))
    else:
        fieldset = FieldSet.from_netcdf(currents_files, currents_variables, currents_dimensions, chunksize=currents_nchs, allow_time_extrapolation=True)
    global tsteps
    tsteps = len(fieldset.U.grid.time_full)
    global tstepsize
    tstepsize = int(math.floor(ttotal/tsteps))

    stokes_files = sorted(glob(os.path.join(stokeshead, "metoffice_wave_amm15_NWS_WAV_3hi2022*.nc")))
    stokes_variables = {'Uw': 'VSDX', 'Vw': 'VSDY'}
    stokes_dimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
    stokes_nchs = None
    if chunk_level > 1:
        stokes_nchs = {
            'Uw': {'lon': ('longitude', 64), 'lat': ('latitude', 64), 'time': ('time', 1)},
            'Vw': {'lon': ('longitude', 64), 'lat': ('latitude', 64), 'time': ('time', 1)}
        }
    elif chunk_level > 0:
        stokes_nchs = {
            'Uw': 'auto',
            'Vw': 'auto'
        }
    else:
        stokes_nchs = False
    fieldset_stokes = None
    if periodic_wrap:
        fieldset_stokes = FieldSet.from_netcdf(stokes_files, stokes_variables, stokes_dimensions, chunksize=stokes_nchs, time_periodic=timedelta(days=ttotal))
    else:
        fieldset_stokes = FieldSet.from_netcdf(stokes_files, stokes_variables, stokes_dimensions, chunksize=stokes_nchs, allow_time_extrapolation=True)
    fieldset.add_field(fieldset_stokes.Uw)
    fieldset.add_field(fieldset_stokes.Vw)

    Kh_zonal = None
    Kh_meridional = None
    xdim = fieldset.U.grid.xdim
    ydim = fieldset.U.grid.ydim
    tdim = fieldset.U.grid.tdim
    if anisotropic_diffusion: # simplest case: 10 m/s^2 -> Lacerda et al. 2019
        print("Generating anisotropic diffusion fields ...")
        Kh_zonal = np.ones((ydim, xdim), dtype=np.float32) * 0.5 * 100.
        Kh_meridional = np.empty((ydim, xdim), dtype=np.float32)
        alpha = 1.  # Profile steepness
        L = 1.  # Basin scale
        # Ny = lat.shape[0]  # Number of grid cells in y_direction (101 +2, one level above and one below, where fields are set to zero)
        # dy = 1.03 / Ny  # Spatial resolution
        # y = np.linspace(-0.01, 1.01, 103)  # y-coordinates for grid
        # y_K = np.linspace(0., 1., 101)  # y-coordinates used for setting diffusivity
        beta = np.zeros(ydim)  # Placeholder for fraction term in K(y) formula

        # for yi in range(len(y_K)):
        for yi in range(ydim):
            yk = (fieldset.U.lat[yi] - fieldset.U.lat[0]) / (fieldset.U.lat[1] - fieldset.U.lat[0])
            if yk < L / 2:
                beta[yi] = yk * np.power(L - 2 * yk, 1 / alpha)
            elif yk >= L / 2:
                beta[yi] = (L - yk) * np.power(2 * yk - L, 1 / alpha)
        Kh_meridional_profile = 0.1 * (2 * (1 + alpha) * (1 + 2 * alpha)) / (alpha ** 2 * np.power(L, 1 + 1 / alpha)) * beta
        for i in range(xdim):
            for j in range(ydim):
                Kh_meridional[j, i] = Kh_meridional_profile[j] * 100.
    else:
        print("Generating isotropic diffusion value ...")
        # Kh_zonal = np.ones((ydim, xdim), dtype=np.float32) * np.random.uniform(0.85, 1.15) * 100.0  # in m^2/s
        # Kh_meridional = np.ones((ydim, xdim), dtype=np.float32) * np.random.uniform(0.7, 1.3) * 100.0  # in m^2/s
        # mesh_conversion = 1.0 / 1852. / 60 if fieldset.U.grid.mesh == 'spherical' else 1.0
        Kh_zonal = np.random.uniform(0.85, 1.15) * 100.0  # in m^2/s
        Kh_meridional = np.random.uniform(0.7, 1.3) * 100.0  # in m^2/s


    if anisotropic_diffusion:
        Kh_grid = RectilinearZGrid(lon=fieldset.U.lon, lat=fieldset.U.lat, mesh=fieldset.U.grid.mesh)
        # fieldset.add_field(Field("Kh_zonal", Kh_zonal, lon=lon, lat=lat, to_write=False, mesh=mesh, transpose=False))
        # fieldset.add_field(Field("Kh_meridional", Kh_meridional, lon=lon, lat=lat, to_write=False, mesh=mesh, transpose=False))
        fieldset.add_field(Field("Kh_zonal", Kh_zonal, grid=Kh_grid, to_write=False, mesh=fieldset.U.grid.mesh, transpose=False))
        fieldset.add_field(Field("Kh_meridional", Kh_meridional, grid=Kh_grid, to_write=False, mesh=fieldset.U.grid.mesh, transpose=False))
        fieldset.add_constant("dres", max(fieldset.U.lat[1]-fieldset.U.lat[0], fieldset.U.lon[1]-fieldset.U.lon[0]))
    else:
        fieldset.add_constant_field("Kh_zonal", Kh_zonal, mesh=fieldset.U.grid.mesh)
        fieldset.add_constant_field("Kh_meridional", Kh_meridional, mesh=fieldset.U.grid.mesh)
        # fieldset.add_constant("Kh_zonal", Kh_zonal)
        # fieldset.add_constant("Kh_meridional", Kh_meridional)

    return fieldset

class SampleParticle(JITParticle):
    valid = Variable('valid', dtype=np.int32, initial=-1, to_write=True)
    sample_u = Variable('sample_u', initial=0., dtype=np.float32, to_write=True)
    sample_v = Variable('sample_v', initial=0., dtype=np.float32, to_write=True)
    sample_uw = Variable('sample_uw', initial=0., dtype=np.float32, to_write=True)
    sample_vw = Variable('sample_vw', initial=0., dtype=np.float32, to_write=True)

def sample_uv(particle, fieldset, time):
    (hu, hv) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    uw = fieldset.Uw[time, particle.depth, particle.lat, particle.lon]
    vw = fieldset.Vw[time, particle.depth, particle.lat, particle.lon]
    particle.sample_u = hu
    particle.sample_v = hv
    particle.sample_uw = uw
    particle.sample_vw = vw
    if particle.valid < 0:
        if (math.isnan(particle.time) == True):
            particle.valid = 0
        else:
            particle.valid = 1

# projectcode = "ENWS"
# projectcode = "DutchCoast"
# projectcode = "DutchCoast25km"
projectcode = "DutchCoastHR"
# ====
# start example: python3 ENWS_scenario_Euler-only.py -f metadata.txt -im 'rk4' -gres 1 -t 365 -dt 600 -ot 3600
#                python3 ENWS_scenario_Euler-only.py -f metadata.txt -im 'rk45' -gres 4 -t 365 -dt 600 -ot 3600
#                python3 ENWS_scenario_Euler-only.py -f metadata.txt -im 'rk45' -gres 32 -t 365 -dt 600 -ot 3600
#                python3 ENWS_scenario_Euler-only.py -f metadata.txt -im 'rk4' -gres 48 -t 365 -dt 600 -ot 3600
# ====
# There are 3600 arcseconds in 1 degree
if __name__=='__main__':
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-f", "--filename", dest="filename", type=str, default="file.txt", help="(relative) (text) file path of the csv hyper parameters")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=365, help="runtime in days (default: 1)")
    parser.add_argument("-dt", "--deltatime", dest="dt", type=int, default=300, help="computational delta_t time stepping in seconds (default: 300sec = 5min = 0.05h)")
    parser.add_argument("-ot", "--outputtime", dest="outdt", type=int, default=1800, help="repeating release rate of added particles in seconds (default: 1800 sec = 30min = 0.5h)")
    parser.add_argument("-im", "--interp_mode", dest="interp_mode", choices=['rk4','rk45', 'ee', 'em', 'm1'], default="jit", help="interpolation mode = [rk4, rk45, ee (Eulerian Estimation), em (Euler-Maruyama), m1 (Milstein-1)]")
    parser.add_argument("-gres", "--grid_resolution", dest="gres", type=str, default="10", help="number of cells per arc-degree or metre (default: 10)")
    parser.add_argument("-G", "--GC", dest="useGC", action='store_true', default=False, help="using a garbage collector (default: false)")
    parser.add_argument("-chs", "--chunksize", dest="chs", type=int, default=0, help="defines the chunksize level: 0=None, 1='auto', 2=fine tuned; default: 0")
    parser.add_argument("--writeNC", dest="writeNC", action='store_true', default=False, help="write output to NetCDF (default: false)")
    parser.add_argument("--writeH5", dest="writeH5", action='store_true', default=False, help="write output to HDF5 (default: false)")
    args = parser.parse_args()

    ParticleSet = DryParticleSet

    filename=args.filename
    deleteBC = True
    animate_result = False
    visualize_results = False
    periodicFlag=True
    backwardSimulation = False
    repeatdtFlag=False
    repeatRateMinutes=720
    time_in_days = args.time_in_days
    agingParticles = False
    writeout = True
    with_GC = args.useGC
    chs = args.chs
    # gres = int(float(eval(args.gres)))
    gres = float(eval(args.gres))
    interp_mode = args.interp_mode
    compute_mode = 'jit'  # args.compute_mode
    netcdf_write = args.writeNC
    hdf5_write = args.writeH5

    dt_seconds = args.dt
    outdt_seconds = args.outdt
    nowtime = datetime.now()
    ParcelsRandom.seed(nowtime.microsecond)

    branch = "ENWS"
    computer_env = "local/unspecified"
    scenario = "resample"
    headdir = ""
    odir = ""
    dirread_pal = ""
    datahead = ""
    dirread_top = ""
    dirread_top_bgc = ""
    if os.uname()[1] in ['science-bs35', 'science-bs36', 'science-bs37']:  # Gemini
        # headdir = "/scratch/{}/experiments/palaeo-parcels".format(os.environ['USER'])
        headdir = "/scratch/{}/experiments".format("ckehl")
        odir = headdir
        datahead = "/data/oceanparcels/input_data"
        dirread_top = os.path.join(datahead, projectcode)
        computer_env = "Gemini"
    elif fnmatch.fnmatchcase(os.uname()[1], "*.bullx*"):  # Cartesius
        CARTESIUS_SCRATCH_USERNAME = 'ckehluu'
        headdir = "/scratch/shared/{}/experiments".format(CARTESIUS_SCRATCH_USERNAME)
        odir = headdir
        datahead = "/projects/0/topios/hydrodynamic_data"
        dirread_top = os.path.join(datahead, projectcode)
        computer_env = "Cartesius"
    elif fnmatch.fnmatchcase(os.uname()[1], "PROMETHEUS"):  # Prometheus computer - use USB drive
        CARTESIUS_SCRATCH_USERNAME = 'christian'
        headdir = "/media/{}/DATA/data/hydrodynamics".format(CARTESIUS_SCRATCH_USERNAME)
        odir = os.path.join(headdir, projectcode)
        datahead = "/media/{}/DATA/data/hydrodynamics".format(CARTESIUS_SCRATCH_USERNAME)
        dirread_top = os.path.join(datahead, "ENWS")
        computer_env = "Prometheus"
    else:
        headdir = "/var/scratch/experiments"
        odir = headdir
        dirread_pal = headdir
        datahead = "/data"
        dirread_top = os.path.join(datahead, projectcode)
    print("running {} on {} (uname: {}) - branch '{}' - argv: {}".format(scenario, computer_env, os.uname()[1], branch, sys.argv[1:]))

    if os.path.sep in filename:
        head_dir = os.path.dirname(filename)
        if head_dir[0] == os.path.sep:
            odir = head_dir
        else:
            odir = os.path.join(odir, head_dir)
        filename = os.path.split(filename)[1]

    func_time = []
    mem_used_GB = []

    np.random.seed(0)

    step = 1.0/gres
    xsteps = int(np.floor((xe-xs) * gres))
    # xsteps = int(np.ceil(a * gres))
    ysteps = int(np.floor((ye-ys) * gres))
    # ysteps = int(np.ceil(b * gres))

    xval = np.arange(start=xs, stop=xe, step=step, dtype=np.float32)
    yval = np.arange(start=ys, stop=ye, step=step, dtype=np.float32)
    centers_x = xval + step/2.0
    centers_y = yval + step/2.0
    us = np.zeros((centers_y.shape[0], centers_x.shape[0]))
    vs = np.zeros((centers_y.shape[0], centers_x.shape[0]))
    sdx = np.zeros((centers_y.shape[0], centers_x.shape[0]))
    sdy = np.zeros((centers_y.shape[0], centers_x.shape[0]))
    seconds_per_day = 24.0*60.0*60.0
    num_t_samples = int(np.floor((time_in_days*seconds_per_day) / outdt_seconds))
    global_fT = np.linspace(start=.0, stop=time_in_days*seconds_per_day, num=num_t_samples, endpoint=True, dtype=np.float64)

    out_fname = "flow"
    sample_outname = out_fname + "_sampleuv"
    zarr_sample_filename = os.path.join(odir, sample_outname + ".zarr")
    nc_sample_filename = os.path.join(odir, sample_outname + ".nc")
    p_center_y, p_center_x = None, None
    period = 365.0
    if(os.path.exists(zarr_sample_filename) == False) or (os.path.exists(nc_sample_filename) == False):
        print("Sampling UV on CMEMS grid ...")
        sample_time = 0
        # sample_func = sample_uv
        fieldset = create_ENWS_fieldset(datahead=dirread_top, periodic_wrap=True, period=period, chunk_level=chs, anisotropic_diffusion=False)
        p_center_y, p_center_x = np.meshgrid(centers_y, centers_x, sparse=False, indexing='ij')
        sample_pset = ParticleSet(fieldset=fieldset, pclass=SampleParticle, lon=np.array(p_center_x).flatten(), lat=np.array(p_center_y).flatten(), time=sample_time)
        sample_kernel = sample_pset.Kernel(sample_uv)
        # sample_output_file = sample_pset.ParticleFile(name=nc_sample_filename, outputdt=timedelta(seconds=outdt_seconds))
        sample_output_file = sample_pset.ParticleFile(name=zarr_sample_filename, outputdt=timedelta(seconds=outdt_seconds))
        postProcessFuncs = []
        if with_GC:
            postProcessFuncs = [perIterGC, ]
        if backwardSimulation:
            sample_pset.execute(sample_kernel, runtime=timedelta(days=time_in_days), dt=timedelta(seconds=-dt_seconds), output_file=sample_output_file, postIterationCallbacks=postProcessFuncs, callbackdt=timedelta(seconds=outdt_seconds))
        else:
            sample_pset.execute(sample_kernel, runtime=timedelta(days=time_in_days), dt=timedelta(seconds=dt_seconds), output_file=sample_output_file, postIterationCallbacks=postProcessFuncs, callbackdt=timedelta(seconds=outdt_seconds))
        # sample_output_file.close()
        del sample_output_file
        del sample_pset
        del sample_kernel
        del fieldset
        print("UV on CMEMS grid sampled.")
    else:
        print("Using previously-computed zArray files in '%s'." % (zarr_sample_filename, ))
        p_center_y, p_center_x = np.meshgrid(centers_y, centers_x, sparse=False, indexing='ij')

    if(os.path.exists(nc_sample_filename) == False):
        print("Convert zArray to NetCDF.")
        try:
            ds_zarr = xr.open_zarr(zarr_sample_filename, chunks=None)
            # encoding_dict = {key: {"zlib": True, "complevel": 3} for key in ds_zarr.data_vars}
            encoding_dict = {key: {"zlib": False} for key in ds_zarr.data_vars}
            with DaskProgressBar():
                ds_zarr.to_netcdf(nc_sample_filename, compute=True, encoding=encoding_dict)
        except Exception as e:
            ds_zarr = xr.open_zarr(zarr_sample_filename, chunks='auto')
            # encoding_dict = {key: {"zlib": True, "complevel": 3} for key in ds_zarr.data_vars}
            encoding_dict = {key: {"zlib": False} for key in ds_zarr.data_vars}
            with DaskProgressBar():
                ds_zarr.to_netcdf(nc_sample_filename, compute=True, encoding=encoding_dict)
        print("NetCDF conversion done.")
    else:
        print("Using previously-computed NetCDF files in '%s'." % (nc_sample_filename,))
        # sample_xarray = xr.open_dataset(nc_sample_filename)
        # p_center_x = np.squeeze(sample_xarray['lon'].data[:, 0]).reshape((centers_y.shape[0], centers_x.shape[0]))
        # p_center_y = np.squeeze(sample_xarray['lat'].data[:, 0]).reshape((centers_y.shape[0], centers_x.shape[0]))
    print("p_center_x shape: {}; p_center_y shape: {}".format(p_center_x.shape, p_center_y.shape))

    print("Load sampled data ...")
    sample_xarray = xr.open_dataset(nc_sample_filename)
    N_s = sample_xarray['lon'].shape[0]
    tN_s = sample_xarray['lon'].shape[1]
    print("N: {}, t_N: {}".format(N_s, tN_s))
    valid_array = np.maximum(np.max(np.array(sample_xarray['valid'][:, 0:2]), axis=1), 0).astype(np.bool_)
    if DBG_MSG:
        print("Valid array: any true ? {}; all true ? {}".format(valid_array.any(), valid_array.all()))
    # this can break the procedure for very large files ...
    ctime_array_s = sample_xarray['time'].data
    time_in_min_s = np.nanmin(ctime_array_s, axis=0)
    time_in_max_s = np.nanmax(ctime_array_s, axis=0)
    print("ctimes array shape: {}".format(ctime_array_s.shape))
    print("|V|: {}, |t|: {}".format(N_s, tN_s))
    assert ctime_array_s.shape[0] == N_s
    assert ctime_array_s.shape[1] == tN_s
    # mask_array_s = np.array([True,] * ctime_array_s.shape[0])  # this array is used to separate valid ocean particles (True) from invalid ones (e.g. land; False)
    mask_array_s = valid_array
    for ti in range(ctime_array_s.shape[1]):
        replace_indices = np.isnan(ctime_array_s[:, ti])
        ctime_array_s[replace_indices, ti] = time_in_max_s[ti]  # in this application, it should always work cause there's no delauyed release
    if DBG_MSG:
        print("time info from file before baselining: shape = {} type = {} range = ({}, {})".format(ctime_array_s.shape, type(ctime_array_s[0 ,0]), np.min(ctime_array_s[0]), np.max(ctime_array_s[0])))
    # timebase_s = ctime_array_s[:, 0]
    # dtime_array_s = np.transpose(ctime_array_s.transpose() - timebase_s)
    timebase_s = time_in_max_s[0]
    dtime_array_s = ctime_array_s - timebase_s
    if DBG_MSG:
        print("time info from file after baselining: shape = {} type = {} range = {}".format( dtime_array_s.shape, type(dtime_array_s[0 ,0]), (np.min(dtime_array_s), np.max(dtime_array_s)) ))
        # print(dtime_array.dtype)
        # print(ns_per_sec.dtype)

    psX = sample_xarray['lon']  # to be loaded from pfile
    psY = sample_xarray['lat']  # to be loaded from pfile
    psZ = None
    if 'depth' in sample_xarray.keys():
        psZ = sample_xarray['depth']  # to be loaded from pfile
    elif 'z' in sample_xarray.keys():
        psZ = sample_xarray['z']  # to be loaded from pfile
    psT = dtime_array_s
    np.set_printoptions(linewidth=160)
    ns_per_sec = np.timedelta64(1, 's')  # nanoseconds in an sec
    global_psT = time_in_max_s -time_in_max_s[0]
    psT = convert_timearray(psT, outdt_seconds, ns_per_sec, debug=DBG_MSG, array_name="psT")
    global_psT = convert_timearray(global_psT, outdt_seconds, ns_per_sec, debug=DBG_MSG, array_name="global_psT")
    psT_dt = global_psT[1] - global_psT[0]
    reverse_time = (np.all(global_psT <= np.finfo(global_psT.dtype).eps) or (np.max(psT[0]) - np.min(psT[0])) < 0) and (psT_dt < 0)
    psT_ext = (global_psT.min(), global_psT.max())
    print("|t_sample|: {}; dt = {}; [T] = {}".format(global_psT.shape, psT_dt, psT_ext))
    psU = sample_xarray['sample_u']  # to be loaded from pfile
    psV = sample_xarray['sample_v']  # to be loaded from pfile
    psUw = sample_xarray['sample_uw']  # to be loaded from pfile
    psVw = sample_xarray['sample_vw']  # to be loaded from pfile
    print("Sampled data loaded.")

    # ==== time interpolation ==== #
    ti_min = 0
    ti_max = global_psT.shape[0]-1
    pT_max = max(global_psT[ti_max], global_psT[ti_min])
    pT_min = min(global_psT[ti_max], global_psT[ti_min])
    interpolate_particles = True
    idt = math.copysign(1.0 * 86400.0, psT_dt)
    iT = global_psT
    cap_min = global_psT[ti_min]
    cap_max = global_psT[ti_max]
    iT_max = np.max(global_psT)
    iT_min = np.min(global_psT)
    tsteps = int(math.floor((psT_ext[1]-psT_ext[0])/psT_dt))
    if interpolate_particles:
        # del us
        # us = None
        # del vs
        # vs = None
        # tsteps = int(math.floor((pT_max-pT_min)/idt)) if not reverse_time else int(math.floor((pT_min-pT_max)/idt))
        tsteps = int(math.ceil((pT_max - pT_min) / idt)) if not reverse_time else int(math.floor((pT_min - pT_max) / idt))
        tsteps = abs(tsteps)
        iT = np.linspace(pT_min, pT_max, tsteps, endpoint=True, dtype=np.float64) if not reverse_time else np.linspace(pT_max, pT_min, tsteps, endpoint=True, dtype=np.float64)
        ti_min = max(np.min(np.nonzero(iT >= cap_min)[0])-1, 0) if not reverse_time else max(np.min(np.nonzero(iT <= cap_min)[0])-1, 0)
        ti_max = min(np.max(np.nonzero(iT <= cap_max)[0])+1, iT.shape[0]-1) if not reverse_time else min(np.max(np.nonzero(iT >= cap_max)[0])+1, iT.shape[0]-1)
        iT_max = np.max(iT)
        iT_min = np.min(iT)
        print("New time field: t_min = {}, t_max = {}, dt = {}, |T| = {}, ti_min_new = {}, ti_max_new = {}".format(iT_min, iT_max, idt, iT.shape[0], ti_min, ti_max))
        # us = np.zeros((centers_y.shape[0], centers_x.shape[0]))
        # vs = np.zeros((centers_y.shape[0], centers_x.shape[0]))
    # ==== end time interpolation ==== #

    if hdf5_write:
        grid_file = h5py.File(os.path.join(odir, "grid.h5"), "w")
        grid_lon_ds = grid_file.create_dataset("longitude", data=centers_x, compression="gzip", compression_opts=4)
        grid_lon_ds.attrs['unit'] = "arc degree"
        grid_lon_ds.attrs['name'] = 'longitude'
        grid_lon_ds.attrs['min'] = centers_x.min()
        grid_lon_ds.attrs['max'] = centers_x.max()
        grid_lat_ds = grid_file.create_dataset("latitude", data=centers_y, compression="gzip", compression_opts=4)
        grid_lat_ds.attrs['unit'] = "arc degree"
        grid_lat_ds.attrs['name'] = 'latitude'
        grid_lat_ds.attrs['min'] = centers_y.min()
        grid_lat_ds.attrs['max'] = centers_y.max()
        grid_time_ds = grid_file.create_dataset("times", data=iT, compression="gzip", compression_opts=4)
        grid_time_ds.attrs['unit'] = "seconds"
        grid_time_ds.attrs['name'] = 'time'
        grid_time_ds.attrs['min'] = np.nanmin(iT)
        grid_time_ds.attrs['max'] = np.nanmax(iT)
        grid_file.close()

    us_minmax = [0., 0.]
    us_statistics = [0., 0.]
    us_file, us_file_ds = None, None
    us_nc_file, us_nc_tdim, us_nc_ydim, us_nc_xdim, us_nc_uvel = None, None, None, None, None
    us_nc_tvar, us_nc_yvar, us_nc_xvar = None, None, None
    if not interpolate_particles:
        if hdf5_write:
            us_file = h5py.File(os.path.join(odir, "hydrodynamic_U.h5"), "w")
            us_file_ds = us_file.create_dataset("uo",
                                                shape=(1, us.shape[0], us.shape[1]),
                                                maxshape=(iT.shape[0], us.shape[0], us.shape[1]),
                                                dtype=us.dtype,
                                                compression="gzip", compression_opts=4)
            us_file_ds.attrs['unit'] = "m/s"
            us_file_ds.attrs['name'] = 'meridional_velocity'
        if netcdf_write:
            us_nc_file = Dataset(os.path.join(odir, "hydrodynamic_U.nc"), mode='w', format='NETCDF4_CLASSIC')
            us_nc_xdim = us_nc_file.createDimension('lon', us.shape[1])
            us_nc_ydim = us_nc_file.createDimension('lat', us.shape[0])
            us_nc_tdim = us_nc_file.createDimension('time', None)
            us_nc_xvar = us_nc_file.createVariable('lon', np.float32, ('lon', ))
            us_nc_yvar = us_nc_file.createVariable('lat', np.float32, ('lat', ))
            us_nc_tvar = us_nc_file.createVariable('time', np.float32, ('time', ))
            us_nc_file.title = "hydrodynamic-2D-U"
            us_nc_file.subtitle = "365d-daily"
            us_nc_xvar.units = "arcdegree_eastwards"
            us_nc_xvar.long_name = "longitude"
            us_nc_yvar.units = "arcdegree_northwards"
            us_nc_yvar.long_name = "latitude"
            us_nc_tvar.units = "seconds"
            us_nc_tvar.long_name = "time"
            us_nc_xvar[:] = centers_x
            us_nc_yvar[:] = centers_y
            us_nc_uvel = us_nc_file.createVariable('u', np.float32, ('time', 'lat', 'lon'))
            us_nc_uvel.units = "m/s"
            us_nc_uvel.standard_name = "eastwards longitudinal zonal velocity"

    vs_minmax = [0., 0.]
    vs_statistics = [0., 0.]
    vs_file, vs_file_ds = None, None
    vs_nc_file, vs_nc_tdim, vs_nc_ydim, vs_nc_xdim, vs_nc_vvel = None, None, None, None, None
    vs_nc_tvar, vs_nc_yvar, vs_nc_xvar = None, None, None
    if not interpolate_particles:
        if hdf5_write:
            vs_file = h5py.File(os.path.join(odir, "hydrodynamic_V.h5"), "w")
            vs_file_ds = vs_file.create_dataset("vo",
                                                shape=(1, vs.shape[0], vs.shape[1]),
                                                maxshape=(iT.shape[0], vs.shape[0], vs.shape[1]),
                                                dtype=vs.dtype,
                                                compression="gzip", compression_opts=4)
            vs_file_ds.attrs['unit'] = "m/s"
            vs_file_ds.attrs['name'] = 'zonal_velocity'
        if netcdf_write:
            vs_nc_file = Dataset(os.path.join(odir, "hydrodynamic_V.nc"), mode='w', format='NETCDF4_CLASSIC')
            vs_nc_xdim = vs_nc_file.createDimension('lon', vs.shape[1])
            vs_nc_ydim = vs_nc_file.createDimension('lat', vs.shape[0])
            vs_nc_tdim = vs_nc_file.createDimension('time', None)
            vs_nc_xvar = vs_nc_file.createVariable('lon', np.float32, ('lon', ))
            vs_nc_yvar = vs_nc_file.createVariable('lat', np.float32, ('lat', ))
            vs_nc_tvar = vs_nc_file.createVariable('time', np.float32, ('time', ))
            vs_nc_file.title = "hydrodynamic-2D-V"
            vs_nc_file.subtitle = "365d-daily"
            vs_nc_xvar.units = "arcdegree_eastwards"
            vs_nc_xvar.long_name = "longitude"
            vs_nc_yvar.units = "arcdegree_northwards"
            vs_nc_yvar.long_name = "latitude"
            vs_nc_tvar.units = "seconds"
            vs_nc_tvar.long_name = "time"
            vs_nc_xvar[:] = centers_x
            vs_nc_yvar[:] = centers_y
            vs_nc_vvel = vs_nc_file.createVariable('v', np.float32, ('time', 'lat', 'lon'))
            vs_nc_vvel.units = "m/s"
            vs_nc_vvel.standard_name = "northwards latitudinal meridional velocity"

    sdx_minmax = [0., 0.]
    sdx_statistics = [0., 0.]
    sdx_file, sdx_file_ds = None, None
    sdx_nc_file, sdx_nc_tdim, sdx_nc_ydim, sdx_nc_xdim, sdx_nc_xvel = None, None, None, None, None
    sdx_nc_tvar, sdx_nc_yvar, sdx_nc_xvar = None, None, None
    if not interpolate_particles:
        if hdf5_write:
            sdx_file = h5py.File(os.path.join(odir, "hydrodynamic_SDX.h5"), "w")
            sdx_file_ds = sdx_file.create_dataset("SDX",
                                                  shape=(1, sdx.shape[0], sdx.shape[1]),
                                                  maxshape=(iT.shape[0], sdx.shape[0], sdx.shape[1]),
                                                  dtype=sdx.dtype,
                                                  compression="gzip", compression_opts=4)
            sdx_file_ds.attrs['unit'] = "m/s"
            sdx_file_ds.attrs['name'] = 'meridional_velocity'
        if netcdf_write:
            sdx_nc_file = Dataset(os.path.join(odir, "hydrodynamic_SDX.nc"), mode='w', format='NETCDF4_CLASSIC')
            sdx_nc_xdim = sdx_nc_file.createDimension('lon', sdx.shape[1])
            sdx_nc_ydim = sdx_nc_file.createDimension('lat', sdx.shape[0])
            sdx_nc_tdim = sdx_nc_file.createDimension('time', None)
            sdx_nc_xvar = sdx_nc_file.createVariable('lon', np.float32, ('lon', ))
            sdx_nc_yvar = sdx_nc_file.createVariable('lat', np.float32, ('lat', ))
            sdx_nc_tvar = sdx_nc_file.createVariable('time', np.float32, ('time', ))
            sdx_nc_file.title = "hydrodynamic-2D-SDX"
            sdx_nc_file.subtitle = "365d-daily"
            sdx_nc_xvar.units = "arcdegree_eastwards"
            sdx_nc_xvar.long_name = "longitude"
            sdx_nc_yvar.units = "arcdegree_northwards"
            sdx_nc_yvar.long_name = "latitude"
            sdx_nc_tvar.units = "seconds"
            sdx_nc_tvar.long_name = "time"
            sdx_nc_xvar[:] = centers_x
            sdx_nc_yvar[:] = centers_y
            sdx_nc_xvel = sdx_nc_file.createVariable('SDX', np.float32, ('time', 'lat', 'lon'))
            sdx_nc_xvel.units = "m/s"
            sdx_nc_xvel.standard_name = "eastwards longitudinal zonal stokes-drift velocity"

    sdy_minmax = [0., 0.]
    sdy_statistics = [0., 0.]
    sdy_file, sdy_file_ds = None, None
    sdy_nc_file, sdy_nc_tdim, sdy_nc_ydim, sdy_nc_xdim, sdy_nc_yvel = None, None, None, None, None
    sdy_nc_tvar, sdy_nc_yvar, sdy_nc_xvar = None, None, None
    if not interpolate_particles:
        if hdf5_write:
            sdy_file = h5py.File(os.path.join(odir, "hydrodynamic_SDY.h5"), "w")
            sdy_file_ds = sdy_file.create_dataset("SDY",
                                                  shape=(1, sdy.shape[0], sdy.shape[1]),
                                                  maxshape=(iT.shape[0], sdy.shape[0], sdy.shape[1]),
                                                  dtype=sdy.dtype,
                                                  compression="gzip", compression_opts=4)
            sdy_file_ds.attrs['unit'] = "m/s"
            sdy_file_ds.attrs['name'] = 'zonal_velocity'
        if netcdf_write:
            sdy_nc_file = Dataset(os.path.join(odir, "hydrodynamic_SDY.nc"), mode='w', format='NETCDF4_CLASSIC')
            sdy_nc_xdim = sdy_nc_file.createDimension('lon', sdy.shape[1])
            sdy_nc_ydim = sdy_nc_file.createDimension('lat', sdy.shape[0])
            sdy_nc_tdim = sdy_nc_file.createDimension('time', None)
            sdy_nc_xvar = sdy_nc_file.createVariable('lon', np.float32, ('lon', ))
            sdy_nc_yvar = sdy_nc_file.createVariable('lat', np.float32, ('lat', ))
            sdy_nc_tvar = sdy_nc_file.createVariable('time', np.float32, ('time', ))
            sdy_nc_file.title = "hydrodynamic-2D-SDY"
            sdy_nc_file.subtitle = "365d-daily"
            sdy_nc_xvar.units = "arcdegree_eastwards"
            sdy_nc_xvar.long_name = "longitude"
            sdy_nc_yvar.units = "arcdegree_northwards"
            sdy_nc_yvar.long_name = "latitude"
            sdy_nc_tvar.units = "seconds"
            sdy_nc_tvar.long_name = "time"
            sdy_nc_xvar[:] = centers_x
            sdy_nc_yvar[:] = centers_y
            sdy_nc_yvel = sdy_nc_file.createVariable('SDY', np.float32, ('time', 'lat', 'lon'))
            sdy_nc_yvel.units = "m/s"
            sdy_nc_yvel.standard_name = "northwards latitudinal meridional stokes-drift velocity"

    print("Interpolating UV on a regular-square grid ...")
    # total_items = psT.shape[1]
    total_items = (ti_max - ti_min)+1
    for ti in range(ti_min, ti_max+1):  # range(psT.shape[1]):
        current_time = iT[ti]
        if interpolate_particles:
            # ==== ==== create files ==== ==== #
            us_minmax = [0., 0.]
            us_statistics = [0., 0.]
            if hdf5_write:
                u_filename = "hydrodynamic_U_d%d.h5" % (ti, )
                us_file = h5py.File(os.path.join(odir, u_filename), "w")
                us_file_ds = us_file.create_dataset("uo",
                                                    shape=(1, us.shape[0], us.shape[1]),
                                                    maxshape=(1, us.shape[0], us.shape[1]),
                                                    dtype=us.dtype,
                                                    compression="gzip", compression_opts=4)
                us_file_ds.attrs['unit'] = "m/s"
                us_file_ds.attrs['time'] = current_time
                us_file_ds.attrs['time_unit'] = "s"
                us_file_ds.attrs['name'] = 'meridional_velocity'
            if netcdf_write:
                u_filename = "hydrodynamic_U_d%d.nc" % (ti, )
                us_nc_file = Dataset(os.path.join(odir, u_filename), mode='w', format='NETCDF4_CLASSIC')
                us_nc_xdim = us_nc_file.createDimension('lon', us.shape[1])
                us_nc_ydim = us_nc_file.createDimension('lat', us.shape[0])
                us_nc_tdim = us_nc_file.createDimension('time', 1)
                us_nc_xvar = us_nc_file.createVariable('lon', np.float32, ('lon', ))
                us_nc_yvar = us_nc_file.createVariable('lat', np.float32, ('lat', ))
                us_nc_tvar = us_nc_file.createVariable('time', np.float32, ('time', ))
                us_nc_file.title = "hydrodynamic-2D-U"
                us_nc_file.subtitle = "365d-daily"
                us_nc_xvar.units = "arcdegree_eastwards"
                us_nc_xvar.long_name = "longitude"
                us_nc_yvar.units = "arcdegree_northwards"
                us_nc_yvar.long_name = "latitude"
                us_nc_tvar.units = "seconds"
                us_nc_tvar.long_name = "time"
                us_nc_xvar[:] = centers_x
                us_nc_yvar[:] = centers_y
                us_nc_uvel = us_nc_file.createVariable('u', np.float32, ('time', 'lat', 'lon'))
                us_nc_uvel.units = "m/s"
                us_nc_uvel.standard_name = "eastwards longitudinal zonal velocity"

            vs_minmax = [0., 0.]
            vs_statistics = [0., 0.]
            if hdf5_write:
                v_filename = "hydrodynamic_V_d%d.h5" %(ti, )
                vs_file = h5py.File(os.path.join(odir, v_filename), "w")
                vs_file_ds = vs_file.create_dataset("vo",
                                                    shape=(1, vs.shape[0], vs.shape[1]),
                                                    maxshape=(1, vs.shape[0], vs.shape[1]),
                                                    dtype=vs.dtype,
                                                    compression="gzip", compression_opts=4)
                vs_file_ds.attrs['unit'] = "m/s"
                vs_file_ds.attrs['time'] = current_time
                vs_file_ds.attrs['time_unit'] = "s"
                vs_file_ds.attrs['name'] = 'zonal_velocity'
            if netcdf_write:
                v_filename = "hydrodynamic_V_d%d.nc" % (ti, )
                vs_nc_file = Dataset(os.path.join(odir, v_filename), mode='w', format='NETCDF4_CLASSIC')
                vs_nc_xdim = vs_nc_file.createDimension('lon', vs.shape[1])
                vs_nc_ydim = vs_nc_file.createDimension('lat', vs.shape[0])
                vs_nc_tdim = vs_nc_file.createDimension('time', 1)
                vs_nc_xvar = vs_nc_file.createVariable('lon', np.float32, ('lon', ))
                vs_nc_yvar = vs_nc_file.createVariable('lat', np.float32, ('lat', ))
                vs_nc_tvar = vs_nc_file.createVariable('time', np.float32, ('time', ))
                vs_nc_file.title = "hydrodynamic-2D-V"
                vs_nc_file.subtitle = "365d-daily"
                vs_nc_xvar.units = "arcdegree_eastwards"
                vs_nc_xvar.long_name = "longitude"
                vs_nc_yvar.units = "arcdegree_northwards"
                vs_nc_yvar.long_name = "latitude"
                vs_nc_tvar.units = "seconds"
                vs_nc_tvar.long_name = "time"
                vs_nc_xvar[:] = centers_x
                vs_nc_yvar[:] = centers_y
                vs_nc_vvel = vs_nc_file.createVariable('v', np.float32, ('time', 'lat', 'lon'))
                vs_nc_vvel.units = "m/s"
                vs_nc_vvel.standard_name = "northwards latitudinal meridional velocity"

            sdx_minmax = [0., 0.]
            sdx_statistics = [0., 0.]
            if hdf5_write:
                sdx_filename = "hydrodynamic_SDX_d%d.h5" % (ti, )
                sdx_file = h5py.File(os.path.join(odir, sdx_filename), "w")
                sdx_file_ds = sdx_file.create_dataset("SDX",
                                                      shape=(1, us.shape[0], us.shape[1]),
                                                      maxshape=(1, us.shape[0], us.shape[1]),
                                                      dtype=us.dtype,
                                                      compression="gzip", compression_opts=4)
                sdx_file_ds.attrs['unit'] = "m/s"
                sdx_file_ds.attrs['time'] = current_time
                sdx_file_ds.attrs['time_unit'] = "s"
                sdx_file_ds.attrs['name'] = 'meridional_stokes-drift_velocity'
            if netcdf_write:
                sdx_filename = "hydrodynamic_SDX_d%d.nc" % (ti, )
                sdx_nc_file = Dataset(os.path.join(odir, sdx_filename), mode='w', format='NETCDF4_CLASSIC')
                sdx_nc_xdim = sdx_nc_file.createDimension('lon', sdx.shape[1])
                sdx_nc_ydim = sdx_nc_file.createDimension('lat', sdx.shape[0])
                sdx_nc_tdim = sdx_nc_file.createDimension('time', 1)
                sdx_nc_xvar = sdx_nc_file.createVariable('lon', np.float32, ('lon', ))
                sdx_nc_yvar = sdx_nc_file.createVariable('lat', np.float32, ('lat', ))
                sdx_nc_tvar = sdx_nc_file.createVariable('time', np.float32, ('time', ))
                sdx_nc_file.title = "hydrodynamic-2D-SDX"
                sdx_nc_file.subtitle = "365d-daily"
                sdx_nc_xvar.units = "arcdegree_eastwards"
                sdx_nc_xvar.long_name = "longitude"
                sdx_nc_yvar.units = "arcdegree_northwards"
                sdx_nc_yvar.long_name = "latitude"
                sdx_nc_tvar.units = "seconds"
                sdx_nc_tvar.long_name = "time"
                sdx_nc_xvar[:] = centers_x
                sdx_nc_yvar[:] = centers_y
                sdx_nc_xvel = sdx_nc_file.createVariable('SDX', np.float32, ('time', 'lat', 'lon'))
                sdx_nc_xvel.units = "m/s"
                sdx_nc_xvel.standard_name = "eastwards longitudinal zonal stokes-drift velocity"

            sdy_minmax = [0., 0.]
            sdy_statistics = [0., 0.]
            if hdf5_write:
                sdy_filename = "hydrodynamic_SDY_d%d.h5" %(ti, )
                sdy_file = h5py.File(os.path.join(odir, sdy_filename), "w")
                sdy_file_ds = sdy_file.create_dataset("SDY",
                                                      shape=(1, vs.shape[0], vs.shape[1]),
                                                      maxshape=(1, vs.shape[0], vs.shape[1]),
                                                      dtype=vs.dtype,
                                                      compression="gzip", compression_opts=4)
                sdy_file_ds.attrs['unit'] = "m/s"
                sdy_file_ds.attrs['time'] = current_time
                sdy_file_ds.attrs['time_unit'] = "s"
                sdy_file_ds.attrs['name'] = 'zonal_stokes-drift_velocity'
            if netcdf_write:
                sdy_filename = "hydrodynamic_SDY_d%d.nc" %(ti, )
                sdy_nc_file = Dataset(os.path.join(odir, sdy_filename), mode='w', format='NETCDF4_CLASSIC')
                sdy_nc_xdim = sdy_nc_file.createDimension('lon', sdy.shape[1])
                sdy_nc_ydim = sdy_nc_file.createDimension('lat', sdy.shape[0])
                sdy_nc_tdim = sdy_nc_file.createDimension('time', 1)
                sdy_nc_xvar = sdy_nc_file.createVariable('lon', np.float32, ('lon', ))
                sdy_nc_yvar = sdy_nc_file.createVariable('lat', np.float32, ('lat', ))
                sdy_nc_tvar = sdy_nc_file.createVariable('time', np.float32, ('time', ))
                sdy_nc_file.title = "hydrodynamic-2D-SDY"
                sdy_nc_file.subtitle = "365d-daily"
                sdy_nc_xvar.units = "arcdegree_eastwards"
                sdy_nc_xvar.long_name = "longitude"
                sdy_nc_yvar.units = "arcdegree_northwards"
                sdy_nc_yvar.long_name = "latitude"
                sdy_nc_tvar.units = "seconds"
                sdy_nc_tvar.long_name = "time"
                sdy_nc_xvar[:] = centers_x
                sdy_nc_yvar[:] = centers_y
                sdy_nc_yvel = sdy_nc_file.createVariable('SDY', np.float32, ('time', 'lat', 'lon'))
                sdy_nc_yvel.units = "m/s"
                sdy_nc_yvel.standard_name = "northwards latitudinal meridional stokes-drift velocity"
            # ==== === files created. === ==== #

        if interpolate_particles:
            tx0 = iT_min + float(ti) * idt if not reverse_time else iT_max + float(ti) * idt
            # tx1 = iT_min + float((ti + 1) % iT.shape[0]) * idt if periodicFlag else iT_min + float(min(ti + 1, iT.shape[0]-1)) * idt
            # tx1 = (iT_max + float((ti + 1) % iT.shape[0]) * idt if periodicFlag else iT_max + float(min(ti + 1, iT.shape[0] - 1)) * idt) if reverse_time else tx1
            tx1 = iT_min + float(min(ti + 1, iT.shape[0]-1)) * idt
            tx1 = (iT_max + float(min(ti + 1, iT.shape[0] - 1)) * idt) if reverse_time else tx1
            if DBG_MSG:  #
                print("tx0: {}, tx1: {}".format(tx0, tx1))
            p_ti0 = time_index_value(tx0, global_psT)
            p_tt = time_partion_value(tx0, global_psT)
            p_ti1 = time_index_value(tx1, global_psT)
            if DBG_MSG:  #
                print("p_ti0: {}, p_ti1: {}, p_tt: {}".format(p_ti0, p_ti1, p_tt))
            # psU_ti0 = np.array(psU[:, p_ti0])
            # psU_ti1 = np.array(psU[:, p_ti1])
            # psUw_ti0 = np.array(psUw[:, p_ti0])
            # psUw_ti1 = np.array(psUw[:, p_ti1])
            # psV_ti0 = np.array(psV[:, p_ti0])
            # psV_ti1 = np.array(psV[:, p_ti1])
            # psVw_ti0 = np.array(psVw[:, p_ti0])
            # psVw_ti1 = np.array(psVw[:, p_ti1])
            #==================================== #
            # us_local = np.squeeze(np.array((1.0-p_tt) * (psU[:, p_ti0] + psUw[:, p_ti0]) + p_tt * (psU[:, p_ti1] + psUw[:, p_ti1])))
            us_local = np.squeeze(np.array((1.0 - p_tt) * (psU[:, p_ti0]) + p_tt * (psU[:, p_ti1])))
            sdx_local = np.squeeze(np.array((1.0 - p_tt) * (psUw[:, p_ti0]) + p_tt * (psUw[:, p_ti1])))
            if DBG_MSG:
                print("us_local (after creation): {}".format(us_local.shape))
            us_local = np.expand_dims(us_local, axis=1)
            sdx_local = np.expand_dims(sdx_local, axis=1)
            if DBG_MSG:
                print("us_local (after dim-extension): {}".format(us_local.shape))
            # us_local = np.reshape(us_local, (N_s, 1))  # us_local[:, np.newaxis]  # np.expand_dims(us_local, axis=1)
            # if DBG_MSG:
            #     print("us_local (after reshape): {}".format(us_local.shape))
            us_local[~mask_array_s, :] = 0
            sdx_local[~mask_array_s, :] = 0
            if DBG_MSG:
                print("us_local (after trimming): {}".format(us_local.shape))
            # vs_local = np.squeeze(np.array((1.0-p_tt) * (psV[:, p_ti0] + psVw[:, p_ti0]) + p_tt * (psV[:, p_ti1] + psVw[:, p_ti1])))
            vs_local = np.squeeze(np.array((1.0 - p_tt) * (psV[:, p_ti0]) + p_tt * (psV[:, p_ti1])))
            sdy_local = np.squeeze(np.array((1.0 - p_tt) * (psVw[:, p_ti0]) + p_tt * (psVw[:, p_ti1])))
            vs_local = np.expand_dims(vs_local, axis=1)
            sdy_local = np.expand_dims(sdy_local, axis=1)
            # vs_local = np.reshape(vs_local, (N_s, 1))  # vs_local[:, np.newaxis]  # np.expand_dims(vs_local, axis=1)
            vs_local[~mask_array_s, :] = 0
            sdy_local[~mask_array_s, :] = 0
        else:
            # us_local = np.expand_dims(psU[:, ti]+psUw[:, ti], axis=1)
            # us_local[~mask_array_s, :] = 0
            # vs_local = np.expand_dims(psV[:, ti]+psVw[:, ti], axis=1)
            # vs_local[~mask_array_s, :] = 0
            us_local = np.expand_dims(psU[:, ti], axis=1)
            us_local[~mask_array_s, :] = 0
            vs_local = np.expand_dims(psV[:, ti], axis=1)
            vs_local[~mask_array_s, :] = 0
            sdx_local = np.expand_dims(psUw[:, ti], axis=1)
            sdx_local[~mask_array_s, :] = 0
            sdy_local = np.expand_dims(psVw[:, ti], axis=1)
            sdy_local[~mask_array_s, :] = 0
        if ti == 0 and DBG_MSG:
            print("us.shape {}; us_local.shape: {}; psU.shape: {}; p_center_y.shape: {}".format(us.shape, us_local.shape, psU.shape, p_center_y.shape))

        us[:, :] = np.reshape(us_local, p_center_y.shape)
        vs[:, :] = np.reshape(vs_local, p_center_y.shape)
        sdx[:, :] = np.reshape(sdx_local, p_center_y.shape)
        sdy[:, :] = np.reshape(sdy_local, p_center_y.shape)

        us_minmax = [min(us_minmax[0], us.min()), max(us_minmax[1], us.max())]
        us_statistics[0] += us.mean()
        us_statistics[1] += us.std()
        vs_minmax = [min(vs_minmax[0], vs.min()), max(vs_minmax[1], vs.max())]
        vs_statistics[0] += vs.mean()
        vs_statistics[1] += vs.std()
        sdx_minmax = [min(sdx_minmax[0], sdx.min()), max(sdx_minmax[1], sdx.max())]
        sdx_statistics[0] += sdx.mean()
        sdx_statistics[1] += sdx.std()
        sdy_minmax = [min(sdy_minmax[0], sdy.min()), max(sdy_minmax[1], sdy.max())]
        sdy_statistics[0] += sdy.mean()
        sdy_statistics[1] += sdy.std()
        if not interpolate_particles:
            if hdf5_write:
                us_file_ds.resize((ti+1), axis=0)
                us_file_ds[ti, :, :] = us
                vs_file_ds.resize((ti+1), axis=0)
                vs_file_ds[ti, :, :] = vs
                sdx_file_ds.resize((ti+1), axis=0)
                sdx_file_ds[ti, :, :] = sdx
                sdy_file_ds.resize((ti+1), axis=0)
                sdy_file_ds[ti, :, :] = sdy
            if netcdf_write:
                us_nc_uvel[ti, :, :] = us
                vs_nc_vvel[ti, :, :] = vs
                sdx_nc_xvel[ti, :, :] = sdx
                sdy_nc_yvel[ti, :, :] = sdy
        else:
            if hdf5_write:
                us_file_ds[0, :, :] = us
                vs_file_ds[0, :, :] = vs
                sdx_file_ds[0, :, :] = sdx
                sdy_file_ds[0, :, :] = sdy
            if netcdf_write:
                us_nc_uvel[0, :, :] = us
                vs_nc_vvel[0, :, :] = vs
                sdx_nc_xvel[0, :, :] = sdx
                sdy_nc_yvel[0, :, :] = sdy

        if interpolate_particles:
            if hdf5_write:
                us_file_ds.attrs['min'] = us_minmax[0]
                us_file_ds.attrs['max'] = us_minmax[1]
                us_file_ds.attrs['mean'] = us_statistics[0]
                us_file_ds.attrs['std'] = us_statistics[1]
                us_file.close()
                vs_file_ds.attrs['min'] = vs_minmax[0]
                vs_file_ds.attrs['max'] = vs_minmax[1]
                vs_file_ds.attrs['mean'] = vs_statistics[0]
                vs_file_ds.attrs['std'] = vs_statistics[1]
                vs_file.close()
                sdx_file_ds.attrs['min'] = sdx_minmax[0]
                sdx_file_ds.attrs['max'] = sdx_minmax[1]
                sdx_file_ds.attrs['mean'] = sdx_statistics[0]
                sdx_file_ds.attrs['std'] = sdx_statistics[1]
                sdx_file.close()
                sdy_file_ds.attrs['min'] = sdy_minmax[0]
                sdy_file_ds.attrs['max'] = sdy_minmax[1]
                sdy_file_ds.attrs['mean'] = sdy_statistics[0]
                sdy_file_ds.attrs['std'] = sdy_statistics[1]
                sdy_file.close()
            if netcdf_write:
                us_nc_tvar[0] = current_time
                us_nc_file.close()
                vs_nc_tvar[0] = current_time
                vs_nc_file.close()
                sdx_nc_tvar[0] = current_time
                sdx_nc_file.close()
                sdy_nc_tvar[0] = current_time
                sdy_nc_file.close()

        # del us_local
        # del vs_local
        # del sdx_local
        # del sdy_local
        us_local = None
        vs_local = None
        sdx_local = None
        sdy_local = None
        current_item = ti
        workdone = current_item / total_items
        print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
    print("\nFinished UV-interpolation.")

    if not interpolate_particles:
        if hdf5_write:
            us_file_ds.attrs['min'] = us_minmax[0]
            us_file_ds.attrs['max'] = us_minmax[1]
            us_file_ds.attrs['mean'] = us_statistics[0] / float(iT.shape[0])
            us_file_ds.attrs['std'] = us_statistics[1] / float(iT.shape[0])
            us_file.close()
            vs_file_ds.attrs['min'] = vs_minmax[0]
            vs_file_ds.attrs['max'] = vs_minmax[1]
            vs_file_ds.attrs['mean'] = vs_statistics[0] / float(iT.shape[0])
            vs_file_ds.attrs['std'] = vs_statistics[1] / float(iT.shape[0])
            vs_file.close()
            sdx_file_ds.attrs['min'] = sdx_minmax[0]
            sdx_file_ds.attrs['max'] = sdx_minmax[1]
            sdx_file_ds.attrs['mean'] = sdx_statistics[0] / float(iT.shape[0])
            sdx_file_ds.attrs['std'] = sdx_statistics[1] / float(iT.shape[0])
            sdx_file.close()
            sdy_file_ds.attrs['min'] = sdy_minmax[0]
            sdy_file_ds.attrs['max'] = sdy_minmax[1]
            sdy_file_ds.attrs['mean'] = sdy_statistics[0] / float(iT.shape[0])
            sdy_file_ds.attrs['std'] = sdy_statistics[1] / float(iT.shape[0])
            sdy_file.close()
        if netcdf_write:
            us_nc_tvar[:] = iT[0]
            vs_nc_tvar[:] = iT[0]
            sdx_nc_tvar[:] = iT[0]
            sdy_nc_tvar[:] = iT[0]
            us_nc_file.close()
            vs_nc_file.close()
            sdx_nc_file.close()
            sdy_nc_file.close()

    del centers_x
    del centers_y
    del xval
    del yval
    del global_fT
