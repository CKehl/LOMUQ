"""
Author: Dr. Christian Kehl
Date: 11-02-2020
"""

from parcels import AdvectionEE, AdvectionRK45, AdvectionRK4, AdvectionDiffusionEM, AdvectionDiffusionM1, AdvectionRK4_3D
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
# from numpy.random import default_rng
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
# ------ DutchCoast ------ #
# xs = 2.5394  # arc degree
# xe = 7.2084  # arc degree
# ys = 51.262  # arc degree
# ye = 55.765  # arc degree
# ------ Southern Ocean ------ #
xs = -15.0  # arc degree
xe = 45.0  # arc degree
ys = -65.0  # arc degree
ye = -20.0  # arc degree
zs = 0.5  # arc degree
ze = 5725.0  # arc degree

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


def PolyTEOS10_bsq(particle, fieldset, time):
    '''
    calculates density based on the polyTEOS10-bsq algorithm from Appendix A.2 of
    https://www.sciencedirect.com/science/article/pii/S1463500315000566
    requires fieldset.abs_salinity and fieldset.cons_temperature Fields in the fieldset
    and a particle.density Variable in the ParticleSet

    References:
    Roquet, F., Madec, G., McDougall, T. J., Barker, P. M., 2014: Accurate
    polynomial expressions for the density and specific volume of
    seawater using the TEOS-10 standard. Ocean Modelling.
    McDougall, T. J., D. R. Jackett, D. G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater.  Journal of Atmospheric and
    Oceanic Technology, 20, 730-741.
    '''

    Z = -particle.depth  # note: use negative depths!
    SA = fieldset.salinity[time, particle.depth, particle.lat, particle.lon]
    CT = fieldset.temperature[time, particle.depth, particle.lat, particle.lon]
    rho_sw = 1023.6  # kg/m^3
    g_const = 9.80665  # m/s^2
    p0 = 1.01325  # bar; hPA = bar * 1000

    SAu = 40 * 35.16504 / 35
    CTu = 40
    Zu = 1e4
    deltaS = 32
    R000 = 8.0189615746e+02
    R100 = 8.6672408165e+02
    R200 = -1.7864682637e+03
    R300 = 2.0375295546e+03
    R400 = -1.2849161071e+03
    R500 = 4.3227585684e+02
    R600 = -6.0579916612e+01
    R010 = 2.6010145068e+01
    R110 = -6.5281885265e+01
    R210 = 8.1770425108e+01
    R310 = -5.6888046321e+01
    R410 = 1.7681814114e+01
    R510 = -1.9193502195e+00
    R020 = -3.7074170417e+01
    R120 = 6.1548258127e+01
    R220 = -6.0362551501e+01
    R320 = 2.9130021253e+01
    R420 = -5.4723692739e+00
    R030 = 2.1661789529e+01
    R130 = -3.3449108469e+01
    R230 = 1.9717078466e+01
    R330 = -3.1742946532e+00
    R040 = -8.3627885467e+00
    R140 = 1.1311538584e+01
    R240 = -5.3563304045e+00
    R050 = 5.4048723791e-01
    R150 = 4.8169980163e-01
    R060 = -1.9083568888e-01
    R001 = 1.9681925209e+01
    R101 = -4.2549998214e+01
    R201 = 5.0774768218e+01
    R301 = -3.0938076334e+01
    R401 = 6.6051753097e+00
    R011 = -1.3336301113e+01
    R111 = -4.4870114575e+00
    R211 = 5.0042598061e+00
    R311 = -6.5399043664e-01
    R021 = 6.7080479603e+00
    R121 = 3.5063081279e+00
    R221 = -1.8795372996e+00
    R031 = -2.4649669534e+00
    R131 = -5.5077101279e-01
    R041 = 5.5927935970e-01
    R002 = 2.0660924175e+00
    R102 = -4.9527603989e+00
    R202 = 2.5019633244e+00
    R012 = 2.0564311499e+00
    R112 = -2.1311365518e-01
    R022 = -1.2419983026e+00
    R003 = -2.3342758797e-02
    R103 = -1.8507636718e-02
    R013 = 3.7969820455e-01
    ss = math.sqrt((SA + deltaS) / SAu)
    tt = CT / CTu
    zz = -Z / Zu
    rz3 = R013 * tt + R103 * ss + R003
    rz2 = (R022 * tt + R112 * ss + R012) * tt + (R202 * ss + R102) * ss + R002
    rz1 = (((R041 * tt + R131 * ss + R031) * tt + (R221 * ss + R121) * ss + R021) * tt + ((R311 * ss + R211) * ss + R111) * ss + R011) * tt + (((R401 * ss + R301) * ss + R201) * ss + R101) * ss + R001
    rz0 = (((((R060 * tt + R150 * ss + R050) * tt + (R240 * ss + R140) * ss + R040) * tt + ((R330 * ss + R230) * ss + R130) * ss + R030) * tt + (((R420 * ss + R320) * ss + R220) * ss + R120) * ss + R020) * tt + ((((R510 * ss + R410) * ss + R310) * ss + R210) * ss + R110) * ss + R010) * tt + (((((R600 * ss + R500) * ss + R400) * ss + R300) * ss + R200) * ss + R100) * ss + R000
    particle.density = ((rz3 * zz + rz2) * zz + rz1) * zz + rz0  # [kg/m^3]
    particle.pressure = rho_sw * g_const * particle.depth + p0  # [bar]

def create_CMEMS_fieldset(datahead, periodic_wrap, chunk_level=0, anisotropic_diffusion=False):
    ddir = os.path.join(datahead, "CMEMS/GLOBAL_REANALYSIS_PHY_001_030/")
    files = sorted(glob(ddir+"mercatorglorys12v1_gl12_mean_2016*.nc"))
    variables = {'U': 'uo', 'V': 'vo', 'salinity': 'so', 'temperature': 'thetao'}
    dimensions = {'lon': 'longitude', 'lat': 'latitude', 'depth': 'depth', 'time': 'time'}
    chs = None
    if chunk_level > 1:
        chs = {
            'U': {'lon': ('longitude', 64), 'lat': ('latitude', 32), 'depth': ('depth', 5), 'time': ('time', 1)},  #
            'V': {'lon': ('longitude', 64), 'lat': ('latitude', 32), 'depth': ('depth', 5), 'time': ('time', 1)},  #
            'salinity': {'lon': ('longitude', 64), 'lat': ('latitude', 32), 'depth': ('depth', 5), 'time': ('time', 1)},  #
            'temperature': {'lon': ('longitude', 64), 'lat': ('latitude', 32), 'depth': ('depth', 5), 'time': ('time', 1)},  #
        }
    elif chunk_level > 0:
        chs = {
            'U': 'auto',
            'V': 'auto',
            'salinity': 'auto',
            'temperature': 'auto'
        }
    else:
        chs = False

    # Kh_zonal = np.ones(U.shape, dtype=np.float32) * 0.5  # ?
    # Kh_meridional = np.ones(U.shape, dtype=np.float32) * 0.5  # ?
    # fieldset.add_constant_field("Kh_zonal", 1, mesh="flat")
    # fieldset.add_constant_field("Kh_meridonal", 1, mesh="flat")
    global ttotal
    ttotal = 31  # days
    fieldset = None
    if periodic_wrap:
        # fieldset = FieldSet.from_netcdf(files, variables, dimensions, chunksize=chs, time_periodic=timedelta(days=31))
        fieldset = FieldSet.from_netcdf(files, variables, dimensions, chunksize=chs, time_periodic=timedelta(days=366))
    else:
        fieldset = FieldSet.from_netcdf(files, variables, dimensions, chunksize=chs, allow_time_extrapolation=True)
    global tsteps
    tsteps = len(fieldset.U.grid.time_full)
    global tstepsize
    tstepsize = int(math.floor(ttotal/tsteps))

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
            elif yk[yi] >= L / 2:
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
    density = Variable('density', initial=0., dtype=np.float32, to_write=True)
    pressure = Variable('pressure', initial=0., dtype=np.float32, to_write=True)

def sample_uv(particle, fieldset, time):
    (u, v) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    particle.sample_u = u
    particle.sample_v = v
    # particle.sample_u = fieldset.U[time, particle.depth, particle.lat, particle.lon]
    # particle.sample_v = fieldset.V[time, particle.depth, particle.lat, particle.lon]
    if particle.valid < 0:
        if (math.isnan(particle.time) == True):
            particle.valid = 0
        else:
            particle.valid = 1

# projectcode = "ENWS"
# projectcode = "DutchCoast"
# projectcode = "DutchCoast25km"
projectcode = "SouthernOcean"

# ====
# standard gres: 1 / 0.08333588 = 11.999633291 = 12
# start example: python3 SouthernOcean_Euler-only.py -f metadata.txt -im 'rk4' -gres 1 -t 366 -dt 600 -ot 3600
#                python3 SouthernOcean_Euler-only.py -f metadata.txt -im 'rk45' -gres 4 -t 366 -dt 600 -ot 3600
#                python3 SouthernOcean_Euler-only.py -f metadata.txt -im 'rk45' -gres 32 -t 366 -dt 600 -ot 3600
#                python3 SouthernOcean_Euler-only.py -f metadata.txt -im 'ee' -gres 6 -t 366 -dt 1800 -ot 3600 -chs 2 --convert_chunk --writeNC --writeH5
#                python3 SouthernOcean_Euler-only.py -f metadata.txt -im 'ee' -gres 12 -t 366 -dt 1800 -ot 3600 -chs 2 --convert_chunk --writeNC --writeH5
#                python3 SouthernOcean_Euler-only.py -f metadata.txt -im 'ee' -gres 48 -t 366 -dt 1350 -ot 3600
# ====
if __name__=='__main__':
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-f", "--filename", dest="filename", type=str, default="file.txt", help="(relative) (text) file path of the csv hyper parameters")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=365, help="runtime in days (default: 1)")
    parser.add_argument("-dt", "--deltatime", dest="dt", type=int, default=300, help="computational delta_t time stepping in seconds (default: 300sec = 5min = 0.05h)")
    parser.add_argument("-ot", "--outputtime", dest="outdt", type=int, default=1800, help="repeating release rate of added particles in seconds (default: 1800 sec = 30min = 0.5h)")
    parser.add_argument("-im", "--interp_mode", dest="interp_mode", choices=['rk4','rk45', 'ee', 'em', 'm1'], default="jit", help="interpolation mode = [rk4, rk45, ee (Eulerian Estimation), em (Euler-Maruyama), m1 (Milstein-1)]")
    parser.add_argument("-gres", "--grid_resolution", dest="gres", type=str, default="10", help="number of cells per arc-degree or metre (default: 10)")
    parser.add_argument("-zsteps", "--z_resolution", dest="zsteps", type=int, default=50, help="number of cells in z-direction (default: 50)")
    parser.add_argument("-G", "--GC", dest="useGC", action='store_true', default=False, help="using a garbage collector (default: false)")
    parser.add_argument("-chs", "--chunksize", dest="chs", type=int, default=0, help="defines the chunksize level: 0=None, 1='auto', 2=fine tuned; default: 0")
    parser.add_argument("--writeNC", dest="writeNC", action='store_true', default=False, help="write output to NetCDF (default: false)")
    parser.add_argument("--writeH5", dest="writeH5", action='store_true', default=False, help="write output to HDF5 (default: false)")
    parser.add_argument("--convert_chunk", dest="convert_chunk", action='store_true', default=False, help="use Dask chunking to convert sample particles (default: false)")
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

    branch = projectcode
    computer_env = "local/unspecified"
    scenario = "sample"
    headdir = ""
    odir = ""
    dirread_pal = ""
    datahead = ""
    dirread_top = ""
    dirread_top_bgc = ""
    if os.uname()[1] in ['science-bs35', 'science-bs36', 'science-bs37']:  # Gemini
        # headdir = "/scratch/{}/experiments/palaeo-parcels".format(os.environ['USER'])
        headdir = "/scratch/{}/experiments".format("ckehl")
        odir = os.path.join(headdir, "CMEMS", branch)
        datahead = "/data/oceanparcels/input_data"
        dirread_top = os.path.join(datahead, 'CMEMS', 'GLOBAL_REANALYSIS_PHY_001_030/')
        computer_env = "Gemini"
    elif fnmatch.fnmatchcase(os.uname()[1], "*.bullx*"):  # Cartesius
        CARTESIUS_SCRATCH_USERNAME = 'ckehluu'
        headdir = "/scratch/shared/{}/experiments".format(CARTESIUS_SCRATCH_USERNAME)
        odir = os.path.join(headdir, "CMEMS", branch)
        datahead = "/projects/0/topios/hydrodynamic_data"
        dirread_top = os.path.join(datahead, 'CMEMS', 'GLOBAL_REANALYSIS_PHY_001_030/')
        computer_env = "Cartesius"
    elif fnmatch.fnmatchcase(os.uname()[1], "PROMETHEUS"):  # Prometheus computer - use USB drive
        CARTESIUS_SCRATCH_USERNAME = 'christian'
        headdir = "/media/{}/DATA/data/hydrodynamics".format(CARTESIUS_SCRATCH_USERNAME)
        odir = os.path.join(headdir, "CMEMS", branch)
        datahead = "/media/{}/MyPassport/data/hydrodynamic".format(CARTESIUS_SCRATCH_USERNAME)
        dirread_top = os.path.join(datahead, 'CMEMS', 'GLOBAL_REANALYSIS_PHY_001_030/')
        computer_env = "Prometheus"
    else:
        headdir = "/var/scratch/experiments"
        odir = os.path.join(headdir, "CMEMS", branch)
        dirread_pal = headdir
        datahead = "/data"
        dirread_top = os.path.join(datahead, 'CMEMS', 'GLOBAL_REANALYSIS_PHY_001_030/')
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
    dx = (xe-xs)
    dy = (ye-ys)
    zsteps = args.zsteps
    dz_step = (ze-zs)/zsteps
    print("dx: {}, dy = {}, dz_step = {}\n".format(dx, dy, dz_step))
    xsteps = int(np.floor(dx * gres))
    # xsteps = int(np.ceil(a * gres))
    ysteps = int(np.floor(dy * gres))
    # ysteps = int(np.ceil(b * gres))
    print("xsteps: {}, ysteps = {}, dz_step = {}\n".format(xsteps, ysteps, zsteps))

    xval = np.arange(start=xs, stop=xe, step=step, dtype=np.float32)
    yval = np.arange(start=ys, stop=ye, step=step, dtype=np.float32)
    zval = np.arange(start=zs, stop=ze, step=dz_step, dtype=np.float32)
    # centers_x = (xval + step/2.0)[0:-1]
    # centers_y = (yval + step/2.0)[0:-1]
    # centers_z = (yval + dz_step/2.0)[0:-1]
    corners_x = xval
    corners_y = yval
    corners_z = zval
    us = np.zeros((corners_z.shape[0], corners_y.shape[0], corners_x.shape[0]))
    vs = np.zeros((corners_z.shape[0], corners_y.shape[0], corners_x.shape[0]))
    # ws = np.zeros((corners_z.shape[0], corners_y.shape[0], corners_x.shape[0]))
    salt = np.zeros((corners_z.shape[0], corners_y.shape[0], corners_x.shape[0]))
    rho = np.zeros((corners_z.shape[0], corners_y.shape[0], corners_x.shape[0]))
    pres = np.zeros((corners_z.shape[0], corners_y.shape[0], corners_x.shape[0]))
    seconds_per_day = 24.0*60.0*60.0
    num_t_samples = int(np.floor((time_in_days*seconds_per_day) / outdt_seconds))
    global_fT = np.linspace(start=.0, stop=time_in_days*seconds_per_day, num=num_t_samples, endpoint=True, dtype=np.float64)

    out_fname = "flow"
    sample_outname = out_fname + "_sampleuv"
    zarr_sample_filename = os.path.join(odir, sample_outname + ".zarr")
    nc_sample_filename = os.path.join(odir, sample_outname + ".nc")
    # p_center_z, p_center_y, p_center_x = None, None, None
    p_corner_z, p_corner_y, p_corner_x = None, None, None
    period = 366.0
    if(os.path.exists(zarr_sample_filename) == False) and (os.path.exists(nc_sample_filename) == False):
        print("Sampling UV on CMEMS grid ...")
        sample_time = 0
        # sample_func = sample_uv
        fieldset = create_CMEMS_fieldset(datahead=datahead, periodic_wrap=True, chunk_level=chs, anisotropic_diffusion=False)
        p_corner_z, p_corner_y, p_corner_x = np.meshgrid(corners_z, corners_y, corners_x, sparse=False, indexing='ij')
        sample_pset = ParticleSet(fieldset=fieldset, pclass=SampleParticle, lon=np.array(p_corner_x).flatten(), lat=np.array(p_corner_y).flatten(), depth=np.array(p_corner_z).flatten(), time=sample_time)
        sample_kernel = sample_pset.Kernel(sample_uv) + sample_pset.Kernel(PolyTEOS10_bsq)
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
        p_corner_z, p_corner_y, p_corner_x = np.meshgrid(corners_z, corners_y, corners_x, sparse=False, indexing='ij')

    if(os.path.exists(nc_sample_filename) == False):
        print("Convert zArray to NetCDF.")
        ncchunks = {'trajectory': 1024, 'obs': 31}
        if args.convert_chunk:
            ds_zarr = xr.open_zarr(zarr_sample_filename, chunks='auto')
            encoding_dict = {key: {"zlib": True, "complevel": 9, 'chunksizes': None} for key in ds_zarr.data_vars}
            # encoding_dict = {key: {"zlib": False} for key in ds_zarr.data_vars}
            with DaskProgressBar():
                ds_zarr.to_netcdf(nc_sample_filename, compute=True, encoding=encoding_dict, format="NETCDF4", engine="netcdf4")
        else:
            try:
                ds_zarr = xr.open_zarr(zarr_sample_filename, chunks={})
                # encoding_dict = {key: {"zlib": True, "complevel": 3} for key in ds_zarr.data_vars}
                encoding_dict = {key: {"zlib": False} for key in ds_zarr.data_vars}
                with DaskProgressBar():
                    ds_zarr.to_netcdf(nc_sample_filename, compute=True, encoding=encoding_dict, format="NETCDF4_CLASSIC", engine="netcdf4")
            except Exception as e:
                ds_zarr = xr.open_zarr(zarr_sample_filename, chunks='auto')
                encoding_dict = {key: {"zlib": True, "complevel": 9, 'chunksizes': None} for key in ds_zarr.data_vars}
                # encoding_dict = {key: {"zlib": False} for key in ds_zarr.data_vars}
                with DaskProgressBar():
                    ds_zarr.to_netcdf(nc_sample_filename, compute=True, encoding=encoding_dict, format="NETCDF4", engine="netcdf4")
        print("NetCDF conversion done.")
    else:
        print("Using previously-computed NetCDF files in '%s'." % (nc_sample_filename,))
        # sample_xarray = xr.open_dataset(nc_sample_filename)
        # p_center_x = np.squeeze(sample_xarray['lon'].data[:, 0]).reshape((centers_y.shape[0], centers_x.shape[0]))
        # p_center_y = np.squeeze(sample_xarray['lat'].data[:, 0]).reshape((centers_y.shape[0], centers_x.shape[0]))
    print("p_corner_x shape: {}; p_corner_y shape: {}; p_corner_z shape: {}".format(p_corner_x.shape, p_corner_y.shape, p_corner_z.shape))

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
    ctime_array_s = None
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
    # psSalt = sample_xarray['salinity']  # to be loaded from pfile
    psRho = sample_xarray['density']  # to be loaded from pfile
    psPres = sample_xarray['pressure']  # to be loaded from pfile
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
    # ==== end time interpolation ==== #

    if hdf5_write:
        grid_file = h5py.File(os.path.join(odir, "grid.h5"), "w")
        grid_lon_ds = grid_file.create_dataset("longitude", data=corners_x, compression="gzip", compression_opts=4)
        grid_lon_ds.attrs['unit'] = "arc degree"
        grid_lon_ds.attrs['name'] = 'longitude'
        grid_lon_ds.attrs['min'] = corners_x.min()
        grid_lon_ds.attrs['max'] = corners_x.max()
        grid_lat_ds = grid_file.create_dataset("latitude", data=corners_y, compression="gzip", compression_opts=4)
        grid_lat_ds.attrs['unit'] = "arc degree"
        grid_lat_ds.attrs['name'] = 'latitude'
        grid_lat_ds.attrs['min'] = corners_y.min()
        grid_lat_ds.attrs['max'] = corners_y.max()
        grid_lat_ds = grid_file.create_dataset("depth", data=corners_z, compression="gzip", compression_opts=4)
        grid_lat_ds.attrs['unit'] = "metres"
        grid_lat_ds.attrs['name'] = 'depth'
        grid_lat_ds.attrs['min'] = corners_z.min()
        grid_lat_ds.attrs['max'] = corners_z.max()
        grid_time_ds = grid_file.create_dataset("times", data=iT, compression="gzip", compression_opts=4)
        grid_time_ds.attrs['unit'] = "seconds"
        grid_time_ds.attrs['name'] = 'time'
        grid_time_ds.attrs['min'] = np.nanmin(iT)
        grid_time_ds.attrs['max'] = np.nanmax(iT)
        grid_file.close()

    us_minmax = [0., 0.]
    us_statistics = [0., 0.]
    us_file, us_file_ds = None, None
    us_nc_file, us_nc_tdim, us_nc_zdim, us_nc_ydim, us_nc_xdim, us_nc_uvel = None, None, None, None, None, None
    us_nc_tvar, us_nc_zvar, us_nc_yvar, us_nc_xvar = None, None, None, None
    if not interpolate_particles:
        if hdf5_write:
            us_file = h5py.File(os.path.join(odir, "hydrodynamic_U.h5"), "w")
            us_file_ds = us_file.create_dataset("uo",
                                                shape=(1, us.shape[0], us.shape[1], us.shape[2]),
                                                maxshape=(iT.shape[0], us.shape[0], us.shape[1], us.shape[2]),
                                                dtype=us.dtype,
                                                compression="gzip", compression_opts=4)
            us_file_ds.attrs['unit'] = "m/s"
            us_file_ds.attrs['name'] = 'meridional_velocity'
        if netcdf_write:
            us_nc_file = Dataset(os.path.join(odir, "hydrodynamic_U.nc"), mode='w', format='NETCDF4_CLASSIC')
            us_nc_xdim = us_nc_file.createDimension('lon', us.shape[2])
            us_nc_ydim = us_nc_file.createDimension('lat', us.shape[1])
            us_nc_zdim = us_nc_file.createDimension('depth', us.shape[0])
            us_nc_tdim = us_nc_file.createDimension('time', None)
            us_nc_xvar = us_nc_file.createVariable('lon', np.float32, ('lon', ))
            us_nc_yvar = us_nc_file.createVariable('lat', np.float32, ('lat', ))
            us_nc_zvar = us_nc_file.createVariable('depth', np.float32, ('depth', ))
            us_nc_tvar = us_nc_file.createVariable('time', np.float32, ('time', ))
            us_nc_file.title = "hydrodynamic-2D-U"
            us_nc_file.subtitle = "365d-daily"
            us_nc_xvar.units = "arcdegree_eastwards"
            us_nc_xvar.long_name = "longitude"
            us_nc_yvar.units = "arcdegree_northwards"
            us_nc_yvar.long_name = "latitude"
            us_nc_zvar.units = "metres_down"
            us_nc_zvar.long_name = "depth"
            us_nc_tvar.units = "seconds"
            us_nc_tvar.long_name = "time"
            us_nc_xvar[:] = corners_x
            us_nc_yvar[:] = corners_y
            us_nc_zvar[:] = corners_z
            us_nc_uvel = us_nc_file.createVariable('u', np.float32, ('time', 'depth', 'lat', 'lon'))
            us_nc_uvel.units = "m/s"
            us_nc_uvel.standard_name = "eastwards longitudinal zonal velocity"

    vs_minmax = [0., 0.]
    vs_statistics = [0., 0.]
    vs_file, vs_file_ds = None, None
    vs_nc_file, vs_nc_tdim, vs_nc_zdim, vs_nc_ydim, vs_nc_xdim, vs_nc_vvel = None, None, None, None, None, None
    vs_nc_tvar, vs_nc_zvar, vs_nc_yvar, vs_nc_xvar = None, None, None, None
    if not interpolate_particles:
        if hdf5_write:
            vs_file = h5py.File(os.path.join(odir, "hydrodynamic_V.h5"), "w")
            vs_file_ds = vs_file.create_dataset("vo",
                                                shape=(1, vs.shape[0], vs.shape[1], vs.shape[2]),
                                                maxshape=(iT.shape[0], vs.shape[0], vs.shape[1], vs.shape[2]),
                                                dtype=vs.dtype,
                                                compression="gzip", compression_opts=4)
            vs_file_ds.attrs['unit'] = "m/s"
            vs_file_ds.attrs['name'] = 'zonal_velocity'
        if netcdf_write:
            vs_nc_file = Dataset(os.path.join(odir, "hydrodynamic_V.nc"), mode='w', format='NETCDF4_CLASSIC')
            vs_nc_xdim = vs_nc_file.createDimension('lon', vs.shape[2])
            vs_nc_ydim = vs_nc_file.createDimension('lat', vs.shape[1])
            vs_nc_zdim = vs_nc_file.createDimension('depth', vs.shape[0])
            vs_nc_tdim = vs_nc_file.createDimension('time', None)
            vs_nc_xvar = vs_nc_file.createVariable('lon', np.float32, ('lon', ))
            vs_nc_yvar = vs_nc_file.createVariable('lat', np.float32, ('lat', ))
            vs_nc_zvar = vs_nc_file.createVariable('depth', np.float32, ('depth', ))
            vs_nc_tvar = vs_nc_file.createVariable('time', np.float32, ('time', ))
            vs_nc_file.title = "hydrodynamic-2D-V"
            vs_nc_file.subtitle = "365d-daily"
            vs_nc_xvar.units = "arcdegree_eastwards"
            vs_nc_xvar.long_name = "longitude"
            vs_nc_yvar.units = "arcdegree_northwards"
            vs_nc_yvar.long_name = "latitude"
            vs_nc_zvar.units = "metres_down"
            vs_nc_zvar.long_name = "depth"
            vs_nc_tvar.units = "seconds"
            vs_nc_tvar.long_name = "time"
            vs_nc_xvar[:] = corners_x
            vs_nc_yvar[:] = corners_y
            vs_nc_zvar[:] = corners_z
            vs_nc_vvel = vs_nc_file.createVariable('v', np.float32, ('time', 'depth', 'lat', 'lon'))
            vs_nc_vvel.units = "m/s"
            vs_nc_vvel.standard_name = "northwards latitudinal meridional velocity"

    # salt_minmax = [0., 0.]
    # salt_statistics = [0., 0.]
    # salt_file, salt_file_ds = None, None
    # salt_nc_file, salt_nc_tdim, salt_nc_zdim, salt_nc_ydim, salt_nc_xdim, salt_nc_value = None, None, None, None, None, None
    # salt_nc_tvar, salt_nc_zvar, salt_nc_yvar, salt_nc_xvar = None, None, None, None
    # if not interpolate_particles:
    #     if hdf5_write:
    #         salt_file = h5py.File(os.path.join(odir, "hydrodynamic_salt.h5"), "w")
    #         salt_file_ds = salt_file.create_dataset("salinity",
    #                                                 shape=(1, salt.shape[0], salt.shape[1], salt.shape[2]),
    #                                                 maxshape=(iT.shape[0], salt.shape[0], salt.shape[1], salt.shape[2]),
    #                                                 dtype=salt.dtype,
    #                                                 compression="gzip", compression_opts=4)
    #         salt_file_ds.attrs['unit'] = "kg/m^3"
    #         salt_file_ds.attrs['name'] = "sea water salinity"
    #     if netcdf_write:
    #         salt_nc_file = Dataset(os.path.join(odir, "hydrodynamic_salt.nc"), mode='w', format='NETCDF4_CLASSIC')
    #         salt_nc_xdim = salt_nc_file.createDimension('lon', salt.shape[2])
    #         salt_nc_ydim = salt_nc_file.createDimension('lat', salt.shape[1])
    #         salt_nc_zdim = salt_file.createDimension('depth', salt.shape[0])
    #         salt_nc_tdim = salt_nc_file.createDimension('time', None)
    #         salt_nc_xvar = salt_nc_file.createVariable('lon', np.float32, ('lon', ))
    #         salt_nc_yvar = salt_nc_file.createVariable('lat', np.float32, ('lat', ))
    #         salt_nc_zvar = salt_nc_file.createVariable('depth', np.float32, ('depth', ))
    #         salt_nc_tvar = salt_nc_file.createVariable('time', np.float32, ('time', ))
    #         salt_nc_file.title = "hydrodynamic-2D-salt"
    #         salt_nc_file.subtitle = "365d-daily"
    #         salt_nc_xvar.units = "arcdegree_eastwards"
    #         salt_nc_xvar.long_name = "longitude"
    #         salt_nc_yvar.units = "arcdegree_northwards"
    #         salt_nc_yvar.long_name = "latitude"
    #         salt_nc_zvar.units = "metres_down"
    #         salt_nc_zvar.long_name = "depth"
    #         salt_nc_tvar.units = "seconds"
    #         salt_nc_tvar.long_name = "time"
    #         salt_nc_xvar[:] = corners_x
    #         salt_nc_yvar[:] = corners_y
    #         salt_nc_zvar[:] = corners_z
    #         salt_nc_value = salt_nc_file.createVariable('salinity', np.float32, ('time', 'depth', 'lat', 'lon'))
    #         salt_nc_value.units = "kg/m^3"
    #         salt_nc_value.standard_name = "sea water salinity"

    rho_minmax = [0., 0.]
    rho_statistics = [0., 0.]
    rho_file, rho_file_ds = None, None
    rho_nc_file, rho_nc_tdim, rho_nc_zdim, rho_nc_ydim, rho_nc_xdim, rho_nc_value = None, None, None, None, None, None
    rho_nc_tvar, rho_nc_zvar, rho_nc_yvar, rho_nc_xvar = None, None, None, None
    if not interpolate_particles:
        if hdf5_write:
            rho_file = h5py.File(os.path.join(odir, "hydrodynamic_rho.h5"), "w")
            rho_file_ds = rho_file.create_dataset("density",
                                                  shape=(1, rho.shape[0], rho.shape[1], rho.shape[2]),
                                                  maxshape=(iT.shape[0], rho.shape[0], rho.shape[1], rho.shape[2]),
                                                  dtype=salt.dtype,
                                                  compression="gzip", compression_opts=4)
            rho_file_ds.attrs['unit'] = "kg/m^3"
            rho_file_ds.attrs['name'] = "sea water density"
        if netcdf_write:
            rho_nc_file = Dataset(os.path.join(odir, "hydrodynamic_rho.nc"), mode='w', format='NETCDF4_CLASSIC')
            rho_nc_xdim = rho_nc_file.createDimension('lon', rho.shape[2])
            rho_nc_ydim = rho_nc_file.createDimension('lat', rho.shape[1])
            rho_nc_zdim = rho_nc_file.createDimension('depth', rho.shape[0])
            rho_nc_tdim = rho_nc_file.createDimension('time', None)
            rho_nc_xvar = rho_nc_file.createVariable('lon', np.float32, ('lon', ))
            rho_nc_yvar = rho_nc_file.createVariable('lat', np.float32, ('lat', ))
            rho_nc_zvar = rho_nc_file.createVariable('depth', np.float32, ('depth', ))
            rho_nc_tvar = rho_nc_file.createVariable('time', np.float32, ('time', ))
            rho_nc_file.title = "hydrodynamic-2D-salt"
            rho_nc_file.subtitle = "365d-daily"
            rho_nc_xvar.units = "arcdegree_eastwards"
            rho_nc_xvar.long_name = "longitude"
            rho_nc_yvar.units = "arcdegree_northwards"
            rho_nc_yvar.long_name = "latitude"
            rho_nc_zvar.units = "metres_down"
            rho_nc_zvar.long_name = "depth"
            rho_nc_tvar.units = "seconds"
            rho_nc_tvar.long_name = "time"
            rho_nc_xvar[:] = corners_x
            rho_nc_yvar[:] = corners_y
            rho_nc_zvar[:] = corners_z
            rho_nc_value = rho_nc_file.createVariable('density', np.float32, ('time', 'depth', 'lat', 'lon'))
            rho_nc_value.units = "kg/m^3"
            rho_nc_value.standard_name = "sea water density"

    pres_minmax = [0., 0.]
    pres_statistics = [0., 0.]
    pres_file, pres_file_ds = None, None
    pres_nc_file, pres_nc_tdim, pres_nc_zdim, pres_nc_ydim, pres_nc_xdim, pres_nc_value = None, None, None, None, None, None
    pres_nc_tvar, pres_nc_zvar, pres_nc_yvar, pres_nc_xvar = None, None, None, None
    if not interpolate_particles:
        if hdf5_write:
            pres_file = h5py.File(os.path.join(odir, "hydrodynamic_pressure.h5"), "w")
            pres_file_ds = pres_file.create_dataset("pressure",
                                                    shape=(1, pres.shape[0], pres.shape[1], pres.shape[2]),
                                                    maxshape=(iT.shape[0], pres.shape[0], pres.shape[1], pres.shape[2]),
                                                    dtype=pres.dtype,
                                                    compression="gzip", compression_opts=4)
            pres_file_ds.attrs['unit'] = "bar"
            pres_file_ds.attrs['name'] = "pressure"
        if netcdf_write:
            pres_nc_file = Dataset(os.path.join(odir, "hydrodynamic_pressure.nc"), mode='w', format='NETCDF4_CLASSIC')
            pres_nc_xdim = pres_nc_file.createDimension('lon', pres.shape[2])
            pres_nc_ydim = pres_nc_file.createDimension('lat', pres.shape[1])
            pres_nc_zdim = pres_nc_file.createDimension('depth', pres.shape[0])
            pres_nc_tdim = pres_nc_file.createDimension('time', None)
            pres_nc_xvar = pres_nc_file.createVariable('lon', np.float32, ('lon', ))
            pres_nc_yvar = pres_nc_file.createVariable('lat', np.float32, ('lat', ))
            pres_nc_zvar = pres_nc_file.createVariable('depth', np.float32, ('depth', ))
            pres_nc_tvar = pres_nc_file.createVariable('time', np.float32, ('time', ))
            pres_nc_file.title = "hydrodynamic-2D-pressure"
            pres_nc_file.subtitle = "365d-daily"
            pres_nc_xvar.units = "arcdegree_eastwards"
            pres_nc_xvar.long_name = "longitude"
            pres_nc_yvar.units = "arcdegree_northwards"
            pres_nc_yvar.long_name = "latitude"
            pres_nc_zvar.units = "metres_down"
            pres_nc_zvar.long_name = "depth"
            pres_nc_tvar.units = "seconds"
            pres_nc_tvar.long_name = "time"
            pres_nc_xvar[:] = corners_x
            pres_nc_yvar[:] = corners_y
            pres_nc_zvar[:] = corners_z
            pres_nc_value = pres_nc_file.createVariable('pressure', ('time', 'depth', 'lat', 'lon'))
            pres_nc_value.units = "bar"
            pres_nc_value.standard_name = "pressure"

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
                                                    shape=(1, us.shape[0], us.shape[1], us.shape[2]),
                                                    maxshape=(iT.shape[0], us.shape[0], us.shape[1], us.shape[2]),
                                                    dtype=us.dtype,
                                                    compression="gzip", compression_opts=4)
                us_file_ds.attrs['unit'] = "m/s"
                us_file_ds.attrs['time'] = current_time
                us_file_ds.attrs['time_unit'] = "s"
                us_file_ds.attrs['name'] = 'meridional_velocity'
            if netcdf_write:
                u_filename = "hydrodynamic_U_d%d.nc" % (ti, )
                us_nc_file = Dataset(os.path.join(odir, u_filename), mode='w', format='NETCDF4_CLASSIC')
                us_nc_xdim = us_nc_file.createDimension('lon', us.shape[2])
                us_nc_ydim = us_nc_file.createDimension('lat', us.shape[1])
                us_nc_zdim = us_nc_file.createDimension('depth', us.shape[0])
                us_nc_tdim = us_nc_file.createDimension('time', 1)
                us_nc_xvar = us_nc_file.createVariable('lon', np.float32, ('lon',))
                us_nc_yvar = us_nc_file.createVariable('lat', np.float32, ('lat',))
                us_nc_zvar = us_nc_file.createVariable('depth', np.float32, ('depth',))
                us_nc_tvar = us_nc_file.createVariable('time', np.float32, ('time',))
                us_nc_file.title = "hydrodynamic-2D-U"
                us_nc_file.subtitle = "365d-daily"
                us_nc_xvar.units = "arcdegree_eastwards"
                us_nc_xvar.long_name = "longitude"
                us_nc_yvar.units = "arcdegree_northwards"
                us_nc_yvar.long_name = "latitude"
                us_nc_zvar.units = "metres_down"
                us_nc_zvar.long_name = "depth"
                us_nc_tvar.units = "seconds"
                us_nc_tvar.long_name = "time"
                us_nc_xvar[:] = corners_x
                us_nc_yvar[:] = corners_y
                us_nc_zvar[:] = corners_z
                us_nc_uvel = us_nc_file.createVariable('u', np.float32, ('time', 'depth', 'lat', 'lon'))
                us_nc_uvel.units = "m/s"
                us_nc_uvel.standard_name = "eastwards longitudinal zonal velocity"

            vs_minmax = [0., 0.]
            vs_statistics = [0., 0.]
            if hdf5_write:
                v_filename = "hydrodynamic_V_d%d.h5" %(ti, )
                vs_file = h5py.File(os.path.join(odir, v_filename), "w")
                vs_file_ds = vs_file.create_dataset("vo",
                                                    shape=(1, vs.shape[0], vs.shape[1], vs.shape[2]),
                                                    maxshape=(iT.shape[0], vs.shape[0], vs.shape[1], vs.shape[2]),
                                                    dtype=vs.dtype,
                                                    compression="gzip", compression_opts=4)
                vs_file_ds.attrs['unit'] = "m/s"
                vs_file_ds.attrs['time'] = current_time
                vs_file_ds.attrs['time_unit'] = "s"
                vs_file_ds.attrs['name'] = 'zonal_velocity'
            if netcdf_write:
                v_filename = "hydrodynamic_V_d%d.nc" % (ti, )
                vs_nc_file = Dataset(os.path.join(odir, v_filename), mode='w', format='NETCDF4_CLASSIC')
                vs_nc_xdim = vs_nc_file.createDimension('lon', vs.shape[2])
                vs_nc_ydim = vs_nc_file.createDimension('lat', vs.shape[1])
                vs_nc_zdim = vs_nc_file.createDimension('depth', vs.shape[0])
                vs_nc_tdim = vs_nc_file.createDimension('time', 1)
                vs_nc_xvar = vs_nc_file.createVariable('lon', np.float32, ('lon', ))
                vs_nc_yvar = vs_nc_file.createVariable('lat', np.float32, ('lat', ))
                vs_nc_zvar = vs_nc_file.createVariable('depth', np.float32, ('depth', ))
                vs_nc_tvar = vs_nc_file.createVariable('time', np.float32, ('time', ))
                vs_nc_file.title = "hydrodynamic-2D-V"
                vs_nc_file.subtitle = "365d-daily"
                vs_nc_xvar.units = "arcdegree_eastwards"
                vs_nc_xvar.long_name = "longitude"
                vs_nc_yvar.units = "arcdegree_northwards"
                vs_nc_yvar.long_name = "latitude"
                vs_nc_zvar.units = "metres_down"
                vs_nc_zvar.long_name = "depth"
                vs_nc_tvar.units = "seconds"
                vs_nc_tvar.long_name = "time"
                vs_nc_xvar[:] = corners_x
                vs_nc_yvar[:] = corners_y
                vs_nc_zvar[:] = corners_z
                vs_nc_vvel = vs_nc_file.createVariable('v', np.float32, ('time', 'depth', 'lat', 'lon'))
                vs_nc_vvel.units = "m/s"
                vs_nc_vvel.standard_name = "northwards latitudinal meridional velocity"

            # salt_minmax = [0., 0.]
            # salt_statistics = [0., 0.]
            # if hdf5_write:
            #     salt_filename = "hydrodynamic_salt_d%d.h5" % (ti, )
            #     salt_file = h5py.File(os.path.join(odir, salt_filename), "w")
            #     salt_file_ds = salt_file.create_dataset("salinity",
            #                                             shape=(1, salt.shape[0], salt.shape[1], salt.shape[2]),
            #                                             maxshape=(iT.shape[0], salt.shape[0], salt.shape[1], salt.shape[2]),
            #                                             dtype=salt.dtype,
            #                                             compression="gzip", compression_opts=4)
            #     salt_file_ds.attrs['unit'] = "kg/m^3"
            #     salt_file_ds.attrs['time'] = current_time
            #     salt_file_ds.attrs['time_unit'] = "s"
            #     salt_file_ds.attrs['name'] = "sea water salinity"
            # if netcdf_write:
            #     salt_filename = "hydrodynamic_salt_d%d.nc" % (ti, )
            #     salt_nc_file = Dataset(os.path.join(odir, salt_filename), mode='w', format='NETCDF4_CLASSIC')
            #     salt_nc_xdim = salt_nc_file.createDimension('lon', salt.shape[2])
            #     salt_nc_ydim = salt_nc_file.createDimension('lat', salt.shape[1])
            #     salt_nc_zdim = salt_nc_file.createDimension('depth', salt.shape[0])
            #     salt_nc_tdim = salt_nc_file.createDimension('time', None)
            #     salt_nc_xvar = salt_nc_file.createVariable('lon', np.float32, ('lon', ))
            #     salt_nc_yvar = salt_nc_file.createVariable('lat', np.float32, ('lat', ))
            #     salt_nc_zvar = salt_nc_file.createVariable('depth', np.float32, ('depth', ))
            #     salt_nc_tvar = salt_nc_file.createVariable('time', np.float32, ('time', ))
            #     salt_nc_file.title = "hydrodynamic-2D-salt"
            #     salt_nc_file.subtitle = "365d-daily"
            #     salt_nc_xvar.units = "arcdegree_eastwards"
            #     salt_nc_xvar.long_name = "longitude"
            #     salt_nc_yvar.units = "arcdegree_northwards"
            #     salt_nc_yvar.long_name = "latitude"
            #     salt_nc_zvar.units = "metres_down"
            #     salt_nc_zvar.long_name = "depth"
            #     salt_nc_tvar.units = "seconds"
            #     salt_nc_tvar.long_name = "time"
            #     salt_nc_xvar[:] = corners_x
            #     salt_nc_yvar[:] = corners_y
            #     salt_nc_zvar[:] = corners_z
            #     salt_nc_value = salt_nc_file.createVariable('salinity', np.float32, ('time', 'depth', 'lat', 'lon'))
            #     salt_nc_value.units = "kg/m^3"
            #     salt_nc_value.standard_name = "sea water salinity"

            rho_minmax = [0., 0.]
            rho_statistics = [0., 0.]
            if hdf5_write:
                rho_filename = "hydrodynamic_rho_d%d.h5" % (ti, )
                rho_file = h5py.File(os.path.join(odir, rho_filename), "w")
                rho_file_ds = rho_file.create_dataset("density",
                                                      shape=(1, rho.shape[0], rho.shape[1], rho.shape[2]),
                                                      maxshape=(iT.shape[0], rho.shape[0], rho.shape[1], rho.shape[2]),
                                                      dtype=rho.dtype,
                                                      compression="gzip", compression_opts=4)
                rho_file_ds.attrs['unit'] = "kg/m^3"
                rho_file_ds.attrs['time'] = current_time
                rho_file_ds.attrs['time_unit'] = "s"
                rho_file_ds.attrs['name'] = "sea water density"
            if netcdf_write:
                rho_filename = "hydrodynamic_rho_d%d.nc" % (ti, )
                rho_nc_file = Dataset(os.path.join(odir, rho_filename), mode='w', format='NETCDF4_CLASSIC')
                rho_nc_xdim = rho_nc_file.createDimension('lon', rho.shape[2])
                rho_nc_ydim = rho_nc_file.createDimension('lat', rho.shape[1])
                rho_nc_zdim = rho_nc_file.createDimension('depth', rho.shape[0])
                rho_nc_tdim = rho_nc_file.createDimension('time', None)
                rho_nc_xvar = rho_nc_file.createVariable('lon', np.float32, ('lon', ))
                rho_nc_yvar = rho_nc_file.createVariable('lat', np.float32, ('lat', ))
                rho_nc_zvar = rho_nc_file.createVariable('depth', np.float32, ('depth', ))
                rho_nc_tvar = rho_nc_file.createVariable('time', np.float32, ('time', ))
                rho_nc_file.title = "hydrodynamic-2D-salt"
                rho_nc_file.subtitle = "365d-daily"
                rho_nc_xvar.units = "arcdegree_eastwards"
                rho_nc_xvar.long_name = "longitude"
                rho_nc_yvar.units = "arcdegree_northwards"
                rho_nc_yvar.long_name = "latitude"
                rho_nc_zvar.units = "metres_down"
                rho_nc_zvar.long_name = "depth"
                rho_nc_tvar.units = "seconds"
                rho_nc_tvar.long_name = "time"
                rho_nc_xvar[:] = corners_x
                rho_nc_yvar[:] = corners_y
                rho_nc_zvar[:] = corners_z
                rho_nc_value = rho_nc_file.createVariable('density', np.float32, ('time', 'depth', 'lat', 'lon'))
                rho_nc_value.units = "kg/m^3"
                rho_nc_value.standard_name = "sea water density"

            pres_minmax = [0., 0.]
            pres_statistics = [0., 0.]
            if hdf5_write:
                pres_filename = "hydrodynamic_pressure_d%d.h5" %(ti, )
                pres_file = h5py.File(os.path.join(odir, pres_filename), "w")
                pres_file_ds = pres_file.create_dataset("pressure",
                                                        shape=(1, pres.shape[0], pres.shape[1], pres.shape[2]),
                                                        maxshape=(iT.shape[0], pres.shape[0], pres.shape[1], pres.shape[2]),
                                                        dtype=pres.dtype,
                                                        compression="gzip", compression_opts=4)
                pres_file_ds.attrs['unit'] = "bar"
                pres_file_ds.attrs['time'] = current_time
                pres_file_ds.attrs['time_unit'] = "s"
                pres_file_ds.attrs['name'] = "pressure"
            if netcdf_write:
                pres_filename = "hydrodynamic_pressure_d%d.nc" %(ti, )
                pres_nc_file = Dataset(os.path.join(odir, pres_filename), mode='w', format='NETCDF4_CLASSIC')
                pres_nc_xdim = pres_nc_file.createDimension('lon', pres.shape[2])
                pres_nc_ydim = pres_nc_file.createDimension('lat', pres.shape[1])
                pres_nc_zdim = pres_nc_file.createDimension('depth', pres.shape[0])
                pres_nc_tdim = pres_nc_file.createDimension('time', None)
                pres_nc_xvar = pres_nc_file.createVariable('lon', np.float32, ('lon', ))
                pres_nc_yvar = pres_nc_file.createVariable('lat', np.float32, ('lat', ))
                pres_nc_zvar = pres_nc_file.createVariable('depth', np.float32, ('depth', ))
                pres_nc_tvar = pres_nc_file.createVariable('time', np.float32, ('time', ))
                pres_nc_file.title = "hydrodynamic-2D-pressure"
                pres_nc_file.subtitle = "365d-daily"
                pres_nc_xvar.units = "arcdegree_eastwards"
                pres_nc_xvar.long_name = "longitude"
                pres_nc_yvar.units = "arcdegree_northwards"
                pres_nc_yvar.long_name = "latitude"
                pres_nc_zvar.units = "metres_down"
                pres_nc_zvar.long_name = "depth"
                pres_nc_tvar.units = "seconds"
                pres_nc_tvar.long_name = "time"
                pres_nc_xvar[:] = corners_x
                pres_nc_yvar[:] = corners_y
                pres_nc_zvar[:] = corners_z
                pres_nc_value = pres_nc_file.createVariable('pressure', ('time', 'depth', 'lat', 'lon'))
                pres_nc_value.units = "bar"
                pres_nc_value.standard_name = "pressure"
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
            #==================================== #
            us_local = np.squeeze(np.array((1.0 - p_tt) * (psU[:, p_ti0]) + p_tt * (psU[:, p_ti1])))
            vs_local = np.squeeze(np.array((1.0 - p_tt) * (psV[:, p_ti0]) + p_tt * (psV[:, p_ti1])))
            # salt_local = np.squeeze(np.array((1.0 - p_tt) * (psSalt[:, p_ti0]) + p_tt * (psSalt[:, p_ti1])))
            rho_local = np.squeeze(np.array((1.0 - p_tt) * (psRho[:, p_ti0]) + p_tt * (psRho[:, p_ti1])))
            pres_local = np.squeeze(np.array((1.0 - p_tt) * (psPres[:, p_ti0]) + p_tt * (psPres[:, p_ti1])))
            us_local = np.expand_dims(us_local, axis=1)
            vs_local = np.expand_dims(vs_local, axis=1)
            # salt_local = np.expand_dims(salt_local, axis=1)
            rho_local = np.expand_dims(rho_local, axis=1)
            pres_local = np.expand_dims(pres_local, axis=1)
            us_local[~mask_array_s, :] = 0
            vs_local[~mask_array_s, :] = 0
            # salt_local[~mask_array_s, :] = 0
            rho_local[~mask_array_s, :] = 0
            pres_local[~mask_array_s, :] = 0
        else:
            us_local = np.expand_dims(psU[:, ti], axis=1)
            us_local[~mask_array_s, :] = 0
            vs_local = np.expand_dims(psV[:, ti], axis=1)
            vs_local[~mask_array_s, :] = 0
            # salt_local = np.expand_dims(psSalt[:, ti], axis=1)
            # salt_local[~mask_array_s, :] = 0
            rho_local = np.expand_dims(psRho[:, ti], axis=1)
            rho_local[~mask_array_s, :] = 0
            pres_local = np.expand_dims(psPres[:, ti], axis=1)
            pres_local[~mask_array_s, :] = 0
        if ti == 0 and DBG_MSG:
            print("us.shape {}; us_local.shape: {}; psU.shape: {}; p_corner_y.shape: {}".format(us.shape, us_local.shape, psU.shape, p_corner_y.shape))

        us[:, :, :] = np.reshape(us_local, p_corner_x.shape)
        vs[:, :, :] = np.reshape(vs_local, p_corner_x.shape)
        # salt[:, :, :] = np.reshape(salt_local, p_corner_x.shape)
        rho[:, :, :] = np.reshape(rho_local, p_corner_x.shape)
        pres[:, :, :] = np.reshape(pres_local, p_corner_x.shape)

        us_minmax = [min(us_minmax[0], us.min()), max(us_minmax[1], us.max())]
        us_statistics[0] += us.mean()
        us_statistics[1] += us.std()
        vs_minmax = [min(vs_minmax[0], vs.min()), max(vs_minmax[1], vs.max())]
        vs_statistics[0] += vs.mean()
        vs_statistics[1] += vs.std()
        # salt_minmax = [min(salt_minmax[0], salt.min()), max(salt_minmax[1], salt.max())]
        # salt_statistics[0] += salt.mean()
        # salt_statistics[1] += salt.std()
        rho_minmax = [min(rho_minmax[0], rho.min()), max(rho_minmax[1], rho.max())]
        rho_statistics[0] += rho.mean()
        rho_statistics[1] += rho.std()
        pres_minmax = [min(pres_minmax[0], pres.min()), max(pres_minmax[1], pres.max())]
        pres_statistics[0] += pres.mean()
        pres_statistics[1] += pres.std()
        if not interpolate_particles:
            if hdf5_write:
                us_file_ds.resize((ti+1), axis=0)
                us_file_ds[ti, :, :, :] = us
                vs_file_ds.resize((ti+1), axis=0)
                vs_file_ds[ti, :, :, :] = vs
                # salt_file_ds.resize((ti+1), axis=0)
                # salt_file_ds[ti, :, :, :] = salt
                rho_file_ds.resize((ti+1), axis=0)
                rho_file_ds[ti, :, :, :] = rho
                pres_file_ds.resize((ti+1), axis=0)
                pres_file_ds[ti, :, :, :] = pres
            if netcdf_write:
                us_nc_uvel[ti, :, :, :] = us
                vs_nc_vvel[ti, :, :, :] = vs
                # salt_nc_value[ti, :, :, :] = salt
                rho_nc_value[ti, :, :, :] = rho
                pres_nc_value[ti, :, :, :] = pres
        else:
            if hdf5_write:
                us_file_ds[0, :, :, :] = us
                vs_file_ds[0, :, :, :] = vs
                # salt_file_ds[0, :, :, :] = salt
                rho_file_ds[0, :, :, :] = rho
                pres_file_ds[0, :, :, :] = pres
            if netcdf_write:
                us_nc_uvel[0, :, :, :] = us
                vs_nc_vvel[0, :, :, :] = vs
                # salt_nc_value[0, :, :, :] = salt
                rho_nc_value[0, :, :, :] = rho
                pres_nc_value[0, :, :, :] = pres

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
                # salt_file_ds.attrs['min'] = salt_minmax[0]
                # salt_file_ds.attrs['max'] = salt_minmax[1]
                # salt_file_ds.attrs['mean'] = salt_statistics[0]
                # salt_file_ds.attrs['std'] = salt_statistics[1]
                # salt_file.close()
                rho_file_ds.attrs['min'] = rho_minmax[0]
                rho_file_ds.attrs['max'] = rho_minmax[1]
                rho_file_ds.attrs['mean'] = rho_statistics[0]
                rho_file_ds.attrs['std'] = rho_statistics[1]
                rho_file.close()
                pres_file_ds.attrs['min'] = pres_minmax[0]
                pres_file_ds.attrs['max'] = pres_minmax[1]
                pres_file_ds.attrs['mean'] = pres_statistics[0]
                pres_file_ds.attrs['std'] = pres_statistics[1]
                pres_file.close()
            if netcdf_write:
                us_nc_tvar[0] = current_time
                us_nc_file.close()
                vs_nc_tvar[0] = current_time
                vs_nc_file.close()
                # salt_nc_tvar[0] = current_time
                # salt_nc_file.close()
                rho_nc_tvar[0] = current_time
                rho_nc_file.close()
                pres_nc_tvar[0] = current_time
                pres_nc_file.close()

        us_local = None
        vs_local = None
        # salt_local = None
        rho_local = None
        pres_local = None
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
            # salt_file_ds.attrs['min'] = salt_minmax[0]
            # salt_file_ds.attrs['max'] = salt_minmax[1]
            # salt_file_ds.attrs['mean'] = salt_statistics[0] / float(iT.shape[0])
            # salt_file_ds.attrs['std'] = salt_statistics[1] / float(iT.shape[0])
            # salt_file_ds.close()
            rho_file_ds.attrs['min'] = rho_minmax[0]
            rho_file_ds.attrs['max'] = rho_minmax[1]
            rho_file_ds.attrs['mean'] = rho_statistics[0] / float(iT.shape[0])
            rho_file_ds.attrs['std'] = rho_statistics[1] / float(iT.shape[0])
            rho_file_ds.close()
            pres_file_ds.attrs['min'] = pres_minmax[0]
            pres_file_ds.attrs['max'] = pres_minmax[1]
            pres_file_ds.attrs['mean'] = pres_statistics[0] / float(iT.shape[0])
            pres_file_ds.attrs['std'] = pres_statistics[1] / float(iT.shape[0])
            pres_file_ds.close()
        if netcdf_write:
            us_nc_tvar[:] = iT[0]
            vs_nc_tvar[:] = iT[0]
            # salt_nc_tvar[:] = iT[0]
            rho_nc_tvar[:] = iT[0]
            pres_nc_tvar[:] = iT[0]
            us_nc_file.close()
            vs_nc_file.close()
            # salt_nc_file.close()
            rho_nc_file.close()
            pres_nc_file.close()

    del corners_x
    del corners_y
    del corners_z
    del xval
    del yval
    del zval
    del global_fT