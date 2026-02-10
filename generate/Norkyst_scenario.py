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

import warnings
import xarray as xr
warnings.simplefilter("ignore", category=xr.SerializationWarning)
import dask

import datetime
from argparse import ArgumentParser
import numpy as np
from scipy.interpolate import interpn
from glob import glob
# from numpy.random import default_rng
import fnmatch
import sys
import gc
import os
import time as ostime
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
#Nparticle = int(math.pow(2,11)) # equals to Nparticle = 2048
#Nparticle = int(math.pow(2,12)) # equals to Nparticle = 4096
#Nparticle = int(math.pow(2,13)) # equals to Nparticle = 8192
#Nparticle = int(math.pow(2,14)) # equals to Nparticle = 16384
#Nparticle = int(math.pow(2,15)) # equals to Nparticle = 32768
#Nparticle = int(math.pow(2,16)) # equals to Nparticle = 65536
#Nparticle = int(math.pow(2,17)) # equals to Nparticle = 131072
#Nparticle = int(math.pow(2,18)) # equals to Nparticle = 262144
#Nparticle = int(math.pow(2,19)) # equals to Nparticle = 524288

xs = -4.46
xe = 36.53
ys = 54.96
ye = 75.75
zs = 10.0
ze = 3225.66  # by definiton: meters
tsteps = 122 # in steps
tstepsize = 3.0 # unitary
tscale = 24.0*60.0*60.0 # in seconds
# we need to modify the kernel.execute / pset.execute so that it returns from the JIT
# in a given time WITHOUT writing to disk via outfie => introduce a pyloop_dt

# this array is lon-lat [prev. lat-lon]
coords = [(6.868885, 63.145534),
          (6.840849, 63.690171),
          (7.287284, 64.348307),
          (11.756546, 67.097822),
          (8.998916, 67.969822),
          (16.396970, 71.108340),
          (25.734875, 73.004930)]

def generate_lonlats():
    lons = []
    lats = []
    prev = list(coords[0])
    curr = None
    for i in range(1,len(coords),1):
        if curr is not None:
            prev = curr
        curr = list(coords[i])
        # x = np.array([prev[1], curr[1]])
        # y = np.array([prev[0], curr[0]])
        # xval = np.arange(prev[1], curr[1], 0.05)
        x = np.array([prev[0], curr[0]])
        y = np.array([prev[1], curr[1]])
        xval = np.arange(prev[0], curr[0], 0.05)
        yval = np.interp(xval, x, y)
        for row in range(xval.shape[0]):
            lons.append(xval[row])
            lats.append(yval[row])
    return np.array(lons), np.array(lats)


# Helper function for time-conversion from teh calendar format
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
    if isinstance(ta[0], datetime.datetime) or isinstance(ta[0], datetime.timedelta) or isinstance(ta[0], np.timedelta64) or isinstance(ta[0], np.datetime64) or np.float64(ta[1]-ta[0]) > (dt_minutes+dt_minutes/2.0):
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


def RenewParticle(particle, fieldSet, time):
    EA = fieldset.east_lim
    WE = fieldset.west_lim
    dlon = EA - WE
    NO = fieldset.north_lim
    SO = fieldset.south_lim
    dlat = NO - SO
    if particle.lon < WE:
    # if particle.lon < -(dlon/2.0):
        # particle.lon += dlon
        particle.lon = WE + (math.fabs(particle.lon) - math.fabs(WE))
    if particle.lon > EA:
    # if particle.lon > (dlon/2,0):
        # particle.lon -= dlon
        particle.lon = EA - (math.fabs(particle.lon) - math.fabs(EA))
    if particle.lat < SO:
        # particle.lat += dlat
        particle.lat = SO + (math.fabs(particle.lat) - math.fabs(SO))
    if particle.lat > NO:
        # particle.lat -= dlat
        particle.lat = NO - (math.fabs(particle.lat) - math.fabs(NO))
    if fieldset.isThreeD > 0.0:
        TO = fieldset.top
        BO = fieldset.bottom
        if particle.depth < TO:
            particle.depth = TO + 1.0
        if particle.depth > BO:
            particle.depth = BO - 1.0


def DeleteParticle(particle, fieldset, time):
    if particle.valid < 0:
        particle.valid = 0
    particle.delete()


def WrapClip_BC(particle, fieldSet, time):
    EA = fieldset.east_lim
    WE = fieldset.west_lim
    dlon = EA - WE
    NO = fieldset.north_lim
    SO = fieldset.south_lim
    dlat = NO - SO
    if particle.lon < WE:
    # if particle.lon < -(dlon/2.0):
        # particle.lon += dlon
        particle.lon = WE + (math.fabs(particle.lon) - math.fabs(WE))
    if particle.lon > EA:
    # if particle.lon > (dlon/2,0):
        # particle.lon -= dlon
        particle.lon = EA - (math.fabs(particle.lon) - math.fabs(EA))
    if particle.lat < SO:
        # particle.lat += dlat
        particle.lat = SO + (math.fabs(particle.lat) - math.fabs(SO))
    if particle.lat > NO:
        # particle.lat -= dlat
        particle.lat = NO - (math.fabs(particle.lat) - math.fabs(NO))
    if fieldset.isThreeD > 0.0:
        TO = fieldset.top
        BO = fieldset.bottom
        if particle.depth < TO:
            particle.depth = TO + 1.0
        if particle.depth > BO:
            particle.depth = BO - 1.0


periodicBC = WrapClip_BC


def perIterGC():
    gc.collect()

#
def create_NEMO_fieldset(datahead, periodic_wrap, chunk):
    dirread_top = os.path.join(datahead, 'NEMO-MEDUSA', 'ORCA0083-N006', 'means')
    dirread_mesh = os.path.join(datahead, 'NEMO-MEDUSA', 'ORCA0083-N006', 'domain')
    basefile_str = {
        'U': 'ORCA0083-N06_2000????d05U.nc',
        'V': 'ORCA0083-N06_2000????d05V.nc',
        'W': 'ORCA0083-N06_2000????d05W.nc'
    }
    ufiles = sorted(glob(os.path.join(dirread_top, basefile_str['U'])))
    vfiles = sorted(glob(os.path.join(dirread_top, basefile_str['V'])))
    wfiles = sorted(glob(os.path.join(dirread_top, basefile_str['W'])))
    mesh_mask = glob(os.path.join(dirread_mesh, "coordinates.nc"))

    filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ufiles},
                 'V': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': vfiles},
                 'W': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': wfiles}}

    variables = {'U': 'uo',
                 'V': 'vo',
                 'W': 'wo'}

    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                  'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'}}

    chs = None
    nchs = None
    if args.chs > 1:
        chs = {'time_counter': 1, 'depthu': 75, 'depthv': 75, 'depthw': 75, 'deptht': 75, 'y': 200, 'x': 200}
        nchs = {
            'U': {'lon': ('x', 96), 'lat': ('y', 48), 'depth': ('depthw', 25), 'time': ('time_counter', 1)},
            'V': {'lon': ('x', 96), 'lat': ('y', 48), 'depth': ('depthw', 25), 'time': ('time_counter', 1)},
            'W': {'lon': ('x', 96), 'lat': ('y', 48), 'depth': ('depthw', 25), 'time': ('time_counter', 1)}
        }
    elif args.chs > 0:
        chs = 'auto'
        nchs = {
            'U': 'auto',
            'V': 'auto',
            'W': 'auto'
        }
    else:
        chs = False
        nchs = False

    # Kh_zonal = np.ones(U.shape, dtype=np.float32) * 0.5  # ?
    # Kh_meridional = np.ones(U.shape, dtype=np.float32) * 0.5  # ?
    # fieldset.add_constant_field("Kh_zonal", 1, mesh="flat")
    # fieldset.add_constant_field("Kh_meridonal", 1, mesh="flat")
    global ttotal
    ttotal = 31  # days
    fieldset = None
    dask.config.set({'array.chunk-size': '16MiB'})
    if periodic_wrap:
        try:
            fieldset = FieldSet.from_nemo(filenames, variables, dimensions, field_chunksize=chs, time_periodic=delta(days=366))
        except (SyntaxError, ):
            fieldset = FieldSet.from_nemo(filenames, variables, dimensions, chunksize=nchs, time_periodic=delta(days=366))
    else:
        try:
            fieldset = FieldSet.from_nemo(filenames, variables, dimensions, field_chunksize=chs, allow_time_extrapolation=True)
        except (SyntaxError,):
            fieldset = FieldSet.from_nemo(filenames, variables, dimensions, chunksize=nchs, allow_time_extrapolation=True)
    global tsteps
    tsteps = len(fieldset.U.grid.time_full)
    global tstepsize
    tstepsize = int(math.floor(ttotal/tsteps))
    return fieldset


class SampleParticle(JITParticle):
    valid = Variable('valid', dtype=np.int32, initial=-1, to_write=True)
    sample_u = Variable('sample_u', initial=0., dtype=np.float32, to_write=True)
    sample_v = Variable('sample_v', initial=0., dtype=np.float32, to_write=True)
    sample_w = Variable('sample_w', initial=0., dtype=np.float32, to_write=True)


def sample_uvw(particle, fieldset, time):
    particle.sample_u = fieldset.U[time, particle.depth, particle.lat, particle.lon]
    particle.sample_v = fieldset.V[time, particle.depth, particle.lat, particle.lon]
    particle.sample_w = fieldset.W[time, particle.depth, particle.lat, particle.lon]
    if particle.valid < 0:
        if (math.isnan(particle.time) == True):
            particle.valid = 0
        else:
            particle.valid = 1


class AgeParticle_JIT(JITParticle):
    age = Variable('age', dtype=np.float64, initial=0.0, to_write=False)
    age_d = Variable('age_d', dtype=np.float32, initial=0.0, to_write=True)
    # beached = Variable('beached', dtype=np.int32, initial=0, to_write=True)
    valid = Variable('valid', dtype=np.int32, initial=-1, to_write=True)
    pre_lon = Variable('pre_lon', dtype=np.float32, initial=0., to_write=False)
    pre_lat = Variable('pre_lat', dtype=np.float32, initial=0., to_write=False)
    life_expectancy = Variable('life_expectancy', dtype=np.float64, initial=np.finfo(np.float64).max, to_write=False)


class AgeParticle_SciPy(ScipyParticle):
    age = Variable('age', dtype=np.float64, initial=0.0, to_write=False)
    age_d = Variable('age_d', dtype=np.float32, initial=0.0, to_write=True)
    # beached = Variable('beached', dtype=np.int32, initial=0, to_write=True)
    valid = Variable('valid', dtype=np.int32, initial=-1, to_write=True)
    pre_lon = Variable('pre_lon', dtype=np.float32, initial=0., to_write=False)
    pre_lat = Variable('pre_lat', dtype=np.float32, initial=0., to_write=False)
    life_expectancy = Variable('life_expectancy', dtype=np.float64, initial=np.finfo(np.float64).max, to_write=False)


def age_func(particle, fieldset, time):
    # if particle.state == StateCode.Evaluate:
    if particle.state == StatusCode.Evaluate:
        particle.age = particle.age + math.fabs(particle.dt)
        particle.age_d = particle.age/86400.0
    if particle.age > particle.life_expectancy:
        particle.delete()


def validate(particle, fieldset, time):
    if particle.valid < 0:
        if (math.isnan(particle.time) == True):
            particle.valid = 0
        else:
            particle.valid = 1
    else:
        if ((particle.lon == particle.pre_lon) and (particle.lat == particle.pre_lat)):
            particle.valid = 0
        else:
            if(math.isnan(particle.time) == True):
                particle.valid = 0
            else:
                particle.valid = 1
        # particle.pre_lon = min(max(particle.lon, particle.pre_lon), particle.pre_lon)
        # particle.pre_lat = min(max(particle.lat, particle.pre_lat), particle.pre_lat)
        particle.pre_lon = particle.lon
        particle.pre_lat = particle.lat
    if particle.valid < 1:
        particle.delete()


def UniformDiffusion(particle, fieldset, time):
    dWx = 0.
    dWy = 0.
    bx = 0.
    by = 0.

    if particle.state == StateCode.Evaluate:
        # dWt = 0.
        dWt = math.sqrt(math.fabs(particle.dt))
        dWx = ParcelsRandom.normalvariate(0, dWt)
        dWy = ParcelsRandom.normalvariate(0, dWt)
        bx = math.sqrt(2 * fieldset.Kh_zonal)
        by = math.sqrt(2 * fieldset.Kh_meridional)

    particle.lon += bx * dWx
    particle.lat += by * dWy

age_ptype = {'scipy': AgeParticle_SciPy, 'jit': AgeParticle_JIT}
# age_ptype = {'scipy': TemporalParticles_SciPy, 'jit': TemporalParticles_JIT}

# ====
# start example: python3 NEMO_scenario.py -f NNvsGeostatistics/data/file.txt -t 30 -dt 720 -ot 1440 -im 'rk4' -N 2**12 -sres 2 -sm 'regular_jitter'
#                python3 NEMO_scenario.py -f NNvsGeostatistics/data/file.txt -t 366 -dt 720 -ot 2880 -im 'rk4' -N 2**12 -sres 2.5 -gres 5 -sm 'regular_jitter' -fsx 360 -fsy 180
#                python3 NEMO_scenario.py -f vis_example/metadata.txt -t 366 -dt 60 -ot 720 -im 'rk4' -N 2**12 -sres 2.5 -gres 5 -sm 'regular_jitter' -fsx 360 -fsy 180
# ====
if __name__=='__main__':
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-f", "--filename", dest="filename", type=str, default="file.txt", help="(relative) (text) file path of the csv hyper parameters")
    # parser.add_argument("-b", "--backwards", dest="backwards", action='store_true', default=False, help="enable/disable running the simulation backwards")
    parser.add_argument("-p", "--periodic", dest="periodic", action='store_true', default=False, help="enable/disable periodic wrapping (else: extrapolation)")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=1, help="runtime in days (default: 1)")
    parser.add_argument("-dt", "--deltatime", dest="dt", type=int, default=720, help="computational delta_t time stepping in minutes (default: 720min = 12h)")
    parser.add_argument("-ot", "--outputtime", dest="outdt", type=int, default=1440, help="repeating release rate of added particles in minutes (default: 1440min = 24h)")
    parser.add_argument("-im", "--interp_mode", dest="interp_mode", choices=['rk4','rk45', 'ee', 'em', 'm1'], default="jit", help="interpolation mode = [rk4, rk45, ee (Eulerian Estimation), em (Euler-Maruyama), m1 (Milstein-1)]")
    parser.add_argument("-N", "--n_particles", dest="nparticles", type=str, default="2**6", help="number of particles to generate and advect (default: 2e6)")
    parser.add_argument("-sres", "--sample_resolution", dest="sres", type=str, default="2", help="number of particle samples per arc-dgree (default: 2)")
    parser.add_argument("-gres", "--grid_resolution", dest="gres", type=str, default="10", help="number of cells per arc-degree or metre (default: 10)")
    parser.add_argument("-sm", "--samplemode", dest="sample_mode", choices=['regular_jitter','uniform','gaussian','triangular','vonmises'], default='regular_jitter', help="sampling distribution mode = [regular_jitter, (irregular) uniform, (irregular) gaussian, (irregular) triangular, (irregular) vonmises]")
    parser.add_argument("-chs", "--chunksize", dest="chs", type=int, default=1, help="defines the chunksize level: 0=None, 1='auto', 2=fine tuned; default: 0")
    parser.add_argument("-fsx", "--field_sx", dest="field_sx", type=int, default="480", help="number of original field cells in x-direction")
    parser.add_argument("-fsy", "--field_sy", dest="field_sy", type=int, default="240", help="number of original field cells in y-direction")
    parser.add_argument("-fsz", "--field_sz", dest="field_sz", type=int, default="25", help="number of original field cells in z-direction")
    parser.add_argument("-d", "--delParticle", dest="delete_particle", action='store_true', default=False, help="switch to delete a particle (True) or reset a particle (default: False).")
    parser.add_argument("--dry", dest="dryrun", action="store_true", default=False, help="Start dry run (no benchmarking and its classes")
    args = parser.parse_args()

    ParticleSet = BenchmarkParticleSet
    if args.dryrun:
        ParticleSet = DryParticleSet

    filename=args.filename
    field_sx = args.field_sx
    field_sy = args.field_sy
    field_sz = args.field_sz
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
    with_GC = True
    Nparticle = int(float(eval(args.nparticles)))
    target_N = Nparticle
    addParticleN = 1
    np_scaler = 3.0 / 2.0
    cycle_scaler = 7.0 / 4.0
    # start_N_particles = int(float(eval(args.start_nparticles)))
    start_N_particles = Nparticle
    sres = int(float(eval(args.sres)))
    gres = int(float(eval(args.gres)))
    sample_mode = args.sample_mode
    interp_mode = args.interp_mode
    compute_mode = 'jit'  # args.compute_mode
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        if mpi_comm.Get_rank() == 0:
            if agingParticles and not repeatdtFlag:
                sys.stdout.write("N: {} ( {} )\n".format(Nparticle, int(Nparticle * np_scaler)))
            else:
                sys.stdout.write("N: {}\n".format(Nparticle))
    else:
        if agingParticles and not repeatdtFlag:
            sys.stdout.write("N: {} ( {} )\n".format(Nparticle, int(Nparticle * np_scaler)))
        else:
            sys.stdout.write("N: {}\n".format(Nparticle))

    dt_minutes = args.dt
    outdt_minutes = args.outdt
    nowtime = datetime.datetime.now()
    ParcelsRandom.seed(nowtime.microsecond)

    branch = "oilspill"
    computer_env = "local/unspecified"
    scenario = "Norkyst"
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
        dirread_top = os.path.join(datahead, "NEMO-MEDUSA", "ORCA0083-N006")
        computer_env = "Gemini"
    elif fnmatch.fnmatchcase(os.uname()[1], "*.bullx*"):  # Cartesius
        CARTESIUS_SCRATCH_USERNAME = 'ckehluu'
        headdir = "/scratch/shared/{}/experiments".format(CARTESIUS_SCRATCH_USERNAME)
        odir = headdir
        datahead = "/projects/0/topios/hydrodynamic_data"
        dirread_top = os.path.join(datahead, "NEMO-MEDUSA", "ORCA0083-N006")
        computer_env = "Cartesius"
    elif fnmatch.fnmatchcase(os.uname()[1], "PROMETHEUS"):  # Prometheus computer - use USB drive
        CARTESIUS_SCRATCH_USERNAME = 'christian'
        headdir = "/media/{}/DATA/data/hydrodynamics".format(CARTESIUS_SCRATCH_USERNAME)
        odir = headdir
        datahead = "/media/{}/OneTouch/storage/data/hydrodynamics".format(CARTESIUS_SCRATCH_USERNAME)
        dirread_top = os.path.join(datahead, "NEMO-MEDUSA", "ORCA0083-N006")
        computer_env = "Prometheus"
    else:
        headdir = "/var/scratch/experiments"
        odir = headdir
        dirread_pal = headdir
        datahead = "/data"
        dirread_top = os.path.join(datahead, "NEMO-MEDUSA", "ORCA0083-N006")
    print("running {} on {} (uname: {}) - branch '{}' - (target) N: {} - argv: {}".format(scenario, computer_env, os.uname()[1], branch, target_N, sys.argv[1:]))

    if os.path.sep in filename:
        head_dir = os.path.dirname(filename)
        if head_dir[0] == os.path.sep:
            odir = head_dir
        else:
            odir = os.path.join(odir, head_dir)
        filename = os.path.split(filename)[1]

    func_time = []
    mem_used_GB = []

    # fieldset = None
    # flons = np.array([])
    # flats = np.array([])
    # ftimes = np.array([])
    # U = np.array([])
    # V = np.array([])
    # field_fpath = False
    # if writeout:
    #     field_fpath = os.path.join(odir,"CMEMS")
    #     fieldset = create_CMEMS_fieldset(datahead=datahead, periodic_wrap=True)
    #     flons = fieldset.U.grid.lon
    #     flats = fieldset.U.grid.lat
    #     ftimes = fieldset.U.grid.time_full
    #     U = fieldset.U
    #     V = fieldset.V
    fieldset = create_NEMO_fieldset(datahead=datahead, periodic_wrap=True, chunk=args.chs)

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            global_t_0 = ostime.process_time()
    else:
        global_t_0 = ostime.process_time()

    simStart = None
    for f in fieldset.get_fields():
        if type(f) in [VectorField, NestedField, SummedField]:  # or not f.grid.defer_load
            continue
        else:
            if backwardSimulation:
                simStart=f.grid.time_full[-1]
            else:
                simStart = f.grid.time_full[0]
            break

    start_scaler = 1.0
    add_scaler = 1.0

    if agingParticles:
        start_scaler *= np_scaler
        if not repeatdtFlag:
            Nparticle = int(Nparticle * np_scaler)
        fieldset.add_constant('life_expectancy', datetime.timedelta(days=time_in_days).total_seconds())
    else:
        # making sure we do track age, but life expectancy is a hard full simulation time #
        fieldset.add_constant('life_expectancy', datetime.timedelta(days=time_in_days).total_seconds())
        # age_ptype[(compute_mode).lower()].life_expectancy.initial = delta(days=time_in_days).total_seconds()
        # age_ptype[(compute_mode).lower()].initialized_dynamic.initial = 1
    fieldset.add_constant("east_lim", +a * 0.5)
    fieldset.add_constant("west_lim", -a * 0.5)
    fieldset.add_constant("north_lim", +b * 0.5)
    fieldset.add_constant("south_lim", -b * 0.5)
    fieldset.add_constant("isThreeD", 1.0)

    if repeatdtFlag:
        add_scaler = start_scaler/2.0
        addParticleN = Nparticle/2.0
        refresh_cycle = (datetime.timedelta(days=time_in_days).total_seconds() / (addParticleN/start_N_particles)) / cycle_scaler
        if agingParticles:
            refresh_cycle /= cycle_scaler
        repeatRateMinutes = int(refresh_cycle/60.0) if repeatRateMinutes == 720 else repeatRateMinutes

    print("Sampling the grid and creating the particle set now ...")
    if backwardSimulation:
        # ==== backward simulation ==== #
        if agingParticles:
            if repeatdtFlag:
                startlon, startlat = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), int(np.floor(sres*start_scaler)), sample_mode, None)
                if not isinstance(startlon, np.ndarray):
                    startlon = np.array(startlon).flatten()
                if not isinstance(startlat, np.ndarray):
                    startlat = np.array(startlat).flatten()
                startdep = np.ones(startlon.shape[0], dtype=np.float32)
                repeatlon, repeatlat = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), int(np.floor(sres*add_scaler)), sample_mode, None)
                if not isinstance(repeatlon, np.ndarray):
                    repeatlon = np.array(repeatlon).flatten()
                if not isinstance(repeatlat, np.ndarray):
                    repeatlat = np.array(repeatlat).flatten()
                releasedep = np.ones(repeatlon.shape[0], dtype=np.float32)
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=startlon, lat=startlat, depth=startdep, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                psetA = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=repeatlon, lat=repeatlat, depth=releasedep, time=simStart)
                pset.add(psetA)
            else:
                lons, lats = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), sres, sample_mode, None)
                if not isinstance(lons, np.ndarray):
                    lons = np.array(lons).flatten()
                if not isinstance(lats, np.ndarray):
                    lats = np.array(lats).flatten()
                startdep = np.ones(lons.shape[0], dtype=np.float32)
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=lons, lat=lats, depth=startdep, time=simStart)
        else:
            if repeatdtFlag:
                startlon, startlat = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), int(np.floor(sres*start_scaler)), sample_mode, None)
                if not isinstance(startlon, np.ndarray):
                    startlon = np.array(startlon).flatten()
                if not isinstance(startlat, np.ndarray):
                    startlat = np.array(startlat).flatten()
                startdep = np.ones(startlon.shape[0], dtype=np.float32)
                repeatlon, repeatlat = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), int(np.floor(sres*add_scaler)), sample_mode, None)
                if not isinstance(repeatlon, np.ndarray):
                    repeatlon = np.array(repeatlon).flatten()
                if not isinstance(repeatlat, np.ndarray):
                    repeatlat = np.array(repeatlat).flatten()
                releasedep = np.ones(repeatlon.shape[0], dtype=np.float32)
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=startlon, lat=startlat, depth=startdep, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                psetA = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=repeatlon, lat=repeatlat, depth=releasedep, time=simStart)
                pset.add(psetA)
            else:
                lons, lats = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), sres, sample_mode, None)
                if not isinstance(lons, np.ndarray):
                    lons = np.array(lons).flatten()
                if not isinstance(lats, np.ndarray):
                    lats = np.array(lats).flatten()
                startdep = np.ones(lons.shape[0], dtype=np.float32)
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=lons, lat=lats, depth=startdep, time=simStart)
    else:
        # ==== forward simulation ==== #
        if agingParticles:
            if repeatdtFlag:
                startlon, startlat = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), int(np.floor(sres*start_scaler)), sample_mode, None)
                if not isinstance(startlon, np.ndarray):
                    startlon = np.array(startlon).flatten()
                if not isinstance(startlat, np.ndarray):
                    startlat = np.array(startlat).flatten()
                startdep = np.ones(startlon.shape[0], dtype=np.float32)
                repeatlon, repeatlat = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), int(np.floor(sres*add_scaler)), sample_mode, None)
                if not isinstance(repeatlon, np.ndarray):
                    repeatlon = np.array(repeatlon).flatten()
                if not isinstance(repeatlat, np.ndarray):
                    repeatlat = np.array(repeatlat).flatten()
                releasedep = np.ones(repeatlon.shape[0], dtype=np.float32)
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=startlon, lat=startlat, depth=startdep, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                psetA = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=repeatlon, lat=repeatlat, depth=releasedep, time=simStart)
                pset.add(psetA)
            else:
                lons, lats = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), sres, sample_mode, None)
                if not isinstance(lons, np.ndarray):
                    lons = np.array(lons).flatten()
                if not isinstance(lats, np.ndarray):
                    lats = np.array(lats).flatten()
                startdep = np.ones(lons.shape[0], dtype=np.float32)
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=lons, lat=lats, depth=startdep, time=simStart)
        else:
            if repeatdtFlag:
                startlon, startlat = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), int(np.floor(sres*start_scaler)), sample_mode, None)
                if not isinstance(startlon, np.ndarray):
                    startlon = np.array(startlon).flatten()
                if not isinstance(startlat, np.ndarray):
                    startlat = np.array(startlat).flatten()
                startdep = np.ones(startlon.shape[0], dtype=np.float32)
                repeatlon, repeatlat = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), int(np.floor(sres*add_scaler)), sample_mode, None)
                if not isinstance(repeatlon, np.ndarray):
                    repeatlon = np.array(repeatlon).flatten()
                if not isinstance(repeatlat, np.ndarray):
                    repeatlat = np.array(repeatlat).flatten()
                releasedep = np.ones(repeatlon.shape[0], dtype=np.float32)
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=startlon, lat=startlat, depth=startdep, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                psetA = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=repeatlon, lat=repeatlat, depth=releasedep, time=simStart)
                pset.add(psetA)
            else:
                lons, lats = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), sres, sample_mode, None)
                if not isinstance(lons, np.ndarray):
                    lons = np.array(lons).flatten()
                if not isinstance(lats, np.ndarray):
                    lats = np.array(lats).flatten()
                startdep = np.ones(lons.shape[0], dtype=np.float32)
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=lons, lat=lats, depth=startdep, time=simStart)

    total_particles = lons.shape[0] if not repeatdtFlag else lons[idx1].shape[0] + int(
        math.ceil(time_in_days * 24 * 60 / repeatRateMinutes)) * lons[idx2].shape[0]
    print("Sampling concluded.")

    # =================================================== #
    # ==== Writing simulation parameters to CSV file ==== #
    # =================================================== #
    csv_file = os.path.splitext(filename)[0]+".csv"
    with open(os.path.join(odir, csv_file), 'w') as f:
        header_string = ""
        value_string = ""
        header_string += "(t) sim time [d], (dt) time integral [min], (out_dt) output time integral [min],"
        header_string += "(N) number particles, (sres) sample resolution, (gres) (projected) grid resolution,"
        header_string += "(interp) interpolation function, (smode) sample mode"
        header_string += "\n"
        value_string += "{:5.5f}, {:7.7f}, {:7.7f},".format(time_in_days, dt_minutes, outdt_minutes)
        value_string += "{}, {}, {},".format(len(pset), sres, gres)
        value_string += "{}, {}".format(interp_mode, sample_mode)
        f.write(header_string)
        f.write(value_string)


    output_file = None
    out_fname = "NEMO-MEDUSA"
    if writeout:
        if MPI and (MPI.COMM_WORLD.Get_size()>1):
            out_fname += "_MPI"
        else:
            out_fname += "_noMPI"
        if periodicFlag:
            out_fname += "_p"
        out_fname += "_n"+str(Nparticle)
        if backwardSimulation:
            out_fname += "_bwd"
        else:
            out_fname += "_fwd"
        if repeatdtFlag:
            out_fname += "_add"
        if agingParticles:
            out_fname += "_age"
        if use_3D:
            out_fname += "_3D"
    if writeout:
        try:
            output_file = pset.ParticleFile(name=os.path.join(odir, out_fname + ".nc"), outputdt=datetime.timedelta(minutes=outdt_minutes))
        except:
            output_file = pset.ParticleFile(name=os.path.join(odir, out_fname + ".zarr"), outputdt=datetime.timedelta(minutes=outdt_minutes), chunks=(total_particles, 1))


    delete_func = RenewParticle
    if deleteBC:
        delete_func=DeleteParticle
    postProcessFuncs = []

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            starttime = ostime.process_time()
    else:
        starttime = ostime.process_time()

    kernelfunc = AdvectionEE
    if interp_mode == 'rk4':
        kernelfunc = AdvectionRK4
    elif interp_mode == 'rk45':
        kernelfunc = AdvectionRK45
    elif interp_mode == 'em':
        kernelfunc = AdvectionDiffusionEM
    elif interp_mode == 'm1':
        kernelfunc = AdvectionDiffusionM1

    kernels = pset.Kernel(kernelfunc,delete_cfiles=True)
    if agingParticles:
        kernels += pset.Kernel(initialize, delete_cfiles=True)
        kernels += pset.Kernel(Age, delete_cfiles=True)
    kernels += pset.Kernel(TemporalAging, delete_cfiles=True)
    kernels += pset.Kernel(periodicBC, delete_cfiles=True)  # insert here to correct for boundary conditions right after advection)
    kernels += pset.Kernel(validate, delete_cfiles=True)

    postProcessFuncs.append(perIterGC)
    if backwardSimulation:
        # ==== backward simulation ==== #
        if animate_result:
            pset.execute(kernels, runtime=datetime.timedelta(days=time_in_days), dt=datetime.timedelta(minutes=-dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=outdt_minutes), moviedt=datetime.timedelta(hours=6), movie_background_field=fieldset.U)
        else:
            pset.execute(kernels, runtime=datetime.timedelta(days=time_in_days), dt=datetime.timedelta(minutes=-dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=outdt_minutes))
    else:
        # ==== forward simulation ==== #
        if animate_result:
            pset.execute(kernels, runtime=datetime.timedelta(days=time_in_days), dt=datetime.timedelta(minutes=dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=outdt_minutes), moviedt=datetime.timedelta(hours=6), movie_background_field=fieldset.U)
        else:
            pset.execute(kernels, runtime=datetime.timedelta(days=time_in_days), dt=datetime.timedelta(minutes=dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=outdt_minutes))

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            endtime = ostime.process_time()
    else:
        endtime = ostime.process_time()

    if writeout:
        output_file.close()

    del kernels
    del output_file
    del pset
    del fieldset

    # ================================================================================================================ #
    #          P O S T - P R O C E S S I N G
    # ================================================================================================================ #
    step = 1.0/gres
    zstep = gres*10.0
    xsteps = int(np.floor(a * gres))
    # xsteps = int(np.ceil(a * gres))
    ysteps = int(np.floor(b * gres))
    # ysteps = int(np.ceil(b * gres))
    zsteps = int(np.floor(c * (1.0/(gres*10.0))))
    # zsteps = int(np.ceil(c * gres))

    data_xarray = xr.open_dataset(os.path.join(odir, out_fname + ".nc"))
    N = data_xarray['lon'].shape[0]
    tN = data_xarray['lon'].shape[1]
    if DBG_MSG:
        print("N: {}, t_N: {}".format(N, tN))
    # valid_array = np.max(np.array(data_xarray['valid'][:, 0:2]), axis=1)
    # print("Valid array: any true ? {}; all true ? {} (pre-correct)".format(np.any(valid_array > 0), np.all(valid_array > 0)))
    # print("Valid array: any true ? {}; all true ? {} (pre-correct)".format(np.any(valid_array < 0), np.all(valid_array < 0)))
    # valid_array = np.maximum(valid_array, 0)
    # print("Valid array: any true ? {}; all true ? {} (post-correct)".format(np.any(valid_array > 0), np.all(valid_array > 0)))
    # print("Valid array: any true ? {}; all true ? {} (post-correct)".format(np.any(valid_array < 0), np.all(valid_array < 0)))
    # valid_array = valid_array.astype(np.bool_)
    valid_array = np.maximum(np.max(np.array(data_xarray['valid'][:, 0:2]), axis=1), 0).astype(np.bool_)
    if DBG_MSG:
        print("Valid array: any true ? {}; all true ? {}".format(np.any(valid_array), np.all(valid_array)))
    np.set_printoptions(linewidth=160)
    ns_per_sec = np.timedelta64(1, 's')  # nanoseconds in an sec
    if DBG_MSG:
        print("ns_per_sec = {}".format((ns_per_sec/np.timedelta64(1, 'ns')).astype(np.float64)))
    sec_per_day = 86400.0
    ctime_array = data_xarray['time'].data
    time_in_min = np.nanmin(ctime_array, axis=0)
    time_in_max = np.nanmax(ctime_array, axis=0)
    if DBG_MSG:
        print("Times:\n\tmin = {}\n\tmax = {}".format(time_in_min, time_in_max))
    assert ctime_array.shape[1] == time_in_min.shape[0]
    # mask_array = np.array([True,] * ctime_array.shape[0])  # this array is used to separate valid ocean particles (True) from invalid ones (e.g. land; False)
    mask_array = valid_array
    for ti in range(ctime_array.shape[1]):
        replace_indices = np.isnan(ctime_array[:, ti])
        # if (ti < 3):
        #     reverse_replace = ~replace_indices
        #     mask_array &= reverse_replace
        ctime_array[replace_indices, ti] = time_in_max[ti]  # this ONLY works if there is no delayed start
    if DBG_MSG:
        print("time info from file before baselining: shape = {} type = {} range = ({}, {})".format(ctime_array.shape, type(ctime_array[0, 0]), np.min(ctime_array[0]), np.max(ctime_array[0])))
    # timebase = ctime_array[:, 0]
    # dtime_array = np.transpose(ctime_array.transpose() - timebase)
    timebase = time_in_max[0]
    dtime_array = ctime_array - timebase
    if DBG_MSG:
        print("time info from file after baselining: \n\tshape = {}".format(dtime_array.shape))
        print("\ttype = {}".format(type(dtime_array[0, 0])))
        print("\trange = {}".format( (np.nanmin(dtime_array), np.nanmax(dtime_array)) ))

    fX = data_xarray['lon']  # .data
    fY = data_xarray['lat']  # .data
    fZ = None
    if 'depth' in data_xarray.keys():
        fZ = data_xarray['depth']  # .data
    elif 'depthu' in data_xarray.keys():
        fZ = data_xarray['depthu']  # .data
    elif 'depthv' in data_xarray.keys():
        fZ = data_xarray['depthv']  # .data
    elif 'depthw' in data_xarray.keys():
        fZ = data_xarray['depthw']  # .data
    elif 'z' in data_xarray.keys():
        fZ = data_xarray['z']  # .data
    fT = dtime_array
    global_fT = time_in_max - time_in_max[0]  # also: relative global clock
    fT = convert_timearray(fT, outdt_minutes*60, ns_per_sec, debug=DBG_MSG, array_name="fT")
    global_fT = convert_timearray(global_fT, outdt_minutes*60, ns_per_sec, debug=DBG_MSG, array_name="global_fT")
    fA = data_xarray['age'].data  # to be loaded from pfile | age
    fAd = data_xarray['age_d'].data  # to be loaded from pfile | age_d
    for ti in range(fA.shape[1]):
        replace_indices = np.isnan(fA[:, ti])
        replace_values_age = None
        replace_values_age_d = None
        if ti > 0:
            replace_values_age = np.fmax(fA[:, ti-1], 0)
            replace_values_age_d = np.fmax(fAd[:, ti-1], 0)
        else:
            replace_values_age = np.zeros((fA.shape[0],), dtype=fA.dtype)
            replace_values_age_d = np.zeros((fA.shape[0],), dtype=fAd.dtype)
        fA[replace_indices, ti] = replace_values_age[replace_indices]
        fAd[replace_indices, ti] = replace_values_age_d[replace_indices]
    if DBG_MSG:
        print("original fA.range and fA.dtype (before conversion): ({}, {}) \t {}".format(fA[0].min(), fA[0].max(), fA.dtype))
    if not isinstance(fA[0,0], np.float32) and not isinstance(fA[0,0], np.float64):
        # fA = np.transpose(fA.transpose()-timebase)
        fA = fA - timebase
    fA = convert_timearray(fA, outdt_minutes*60, ns_per_sec, debug=DBG_MSG, array_name="fA")

    pcounts = np.zeros((ysteps, xsteps, zsteps), dtype=np.int32)
    pcounts_minmax = [0., 0.]
    pcounts_statistics = [0., 0.]
    pcounts_file = h5py.File(os.path.join(odir, "particlecount.h5"), "w")
    pcounts_file_ds = pcounts_file.create_dataset("pcount", shape=(1, pcounts.shape[0], pcounts.shape[1]), dtype=pcounts.dtype, maxshape=(fT.shape[1], pcounts.shape[0], pcounts.shape[1]), compression="gzip", compression_opts=4)
    pcounts_file_ds.attrs['unit'] = "count (scalar)"
    pcounts_file_ds.attrs['name'] = 'particle_count'

    density = np.zeros((ysteps, xsteps, zsteps), dtype=np.float32)
    density_minmax = [0., 0.]
    density_statistics = [0., 0.]
    density_file = h5py.File(os.path.join(odir, "density.h5"), "w")
    density_file_ds = density_file.create_dataset("density", shape=(1, density.shape[0], density.shape[1]), dtype=density.dtype, maxshape=(fT.shape[1], density.shape[0], density.shape[1]), compression="gzip", compression_opts=4)
    density_file_ds.attrs['unit'] = "pts / arc_deg^2"
    density_file_ds.attrs['name'] = 'density'

    rel_density = np.zeros((ysteps, xsteps, zsteps), dtype=np.float32)
    rel_density_minmax = [0., 0.]
    rel_density_statistics = [0., 0.]
    rel_density_file = h5py.File(os.path.join(odir, "rel_density.h5"), "w")
    rel_density_file_ds = rel_density_file.create_dataset("rel_density", shape=(1, rel_density.shape[0], rel_density.shape[1]), dtype=rel_density.dtype, maxshape=(fT.shape[1], rel_density.shape[0], rel_density.shape[1]), compression="gzip", compression_opts=4)
    rel_density_file_ds.attrs['unit'] = "pts_percentage / arc_deg^2"
    rel_density_file_ds.attrs['name'] = 'relative_density'

    lifetime = np.zeros((ysteps, xsteps, zsteps), dtype=np.float32)
    lifetime_minmax = [0., 0.]
    lifetime_statistics = [0., 0.]
    lifetime_file = h5py.File(os.path.join(odir, "lifetime.h5"), "w")
    lifetime_file_ds = lifetime_file.create_dataset("lifetime", shape=(1, lifetime.shape[0], lifetime.shape[1]), dtype=lifetime.dtype, maxshape=(fT.shape[1], lifetime.shape[0], lifetime.shape[1]), compression="gzip", compression_opts=4)
    lifetime_file_ds.attrs['unit'] = "avg. lifetime"
    lifetime_file_ds.attrs['name'] = 'lifetime'

    print("\nInterpolating particle properties on regular-grid field ...")
    A = float(gres**2)
    total_items = fT.shape[0] * fT.shape[1]
    local_to_global = np.arange(0, fX.shape[0], 1)[mask_array]
    for ti in range(fT.shape[1]):
        pcounts[:, :] = 0
        density[:, :] = 0
        rel_density[:, :] = 0
        lifetime[:, :] = 0
        tlifetime = np.zeros((ysteps, xsteps), dtype=np.float32)
        x_in = np.array(fX[:, ti])[mask_array]
        nonnan_x = ~np.isnan(x_in)
        y_in = np.array(fY[:, ti])[mask_array]
        nonnan_y = ~np.isnan(y_in)
        x_in = x_in[np.logical_and(nonnan_x, nonnan_y)]
        y_in = y_in[np.logical_and(nonnan_x, nonnan_y)]
        # print("/n")
        # print("x_in: {} - min: {}, max: {}, len: {}".format(x_in, np.min(x_in), np.max(x_in), x_in.shape[0]))
        # print("y_in: {} - min: {}, max: {}, len: {}".format(y_in, np.min(y_in), np.max(y_in), y_in.shape[0]))
        xpts = np.floor((x_in+(a/2.0))*gres).astype(np.int32).flatten()
        ypts = np.floor((y_in+(b/2.0))*gres).astype(np.int32).flatten()
        assert xpts.shape[0] == ypts.shape[0], "Dimensions of xpts (={}) does not match ypts(={}).".format(xpts.shape[0], ypts.shape[0])
        xcondition = np.logical_and((xpts >= 0), (xpts < (xsteps - 1)))
        ycondition = np.logical_and((ypts >= 0), (ypts < (ysteps - 1)))
        xpts = xpts[np.logical_and(xcondition, ycondition)]
        ypts = ypts[np.logical_and(xcondition, ycondition)]
        if ti == 0 and DBG_MSG:
            print("xpts: {} - min: {}, max: {}".format(xpts, np.min(xpts), np.max(xpts)))
            print("ypts: {} - min: {}, max: {}".format(ypts, np.min(ypts), np.max(ypts)))
        for pi in range(xpts.shape[0]):
            try:
                pcounts[ypts[pi], xpts[pi]] += 1  # error Jan-14 2024 at ypts
                tlifetime[ypts[pi], xpts[pi]] += fA[local_to_global[pi], ti]
            except (IndexError, ) as error_msg:
                # we don't abort here cause the brownian-motion wiggle of AvectionRK4EulerMarujama always edges on machine precision, which can np.floor(..) make go over-size
                print("\nError trying to index point ({}, {}) with indices ({}, {}), point index {}".format(fX[pi, ti], fY[pi, ti], xpts[pi], ypts[pi], pi))
            if (pi % 100) == 0:
                current_item = (ti*fT.shape[0]) + local_to_global[pi]
                workdone = current_item / total_items
                print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
        density = pcounts.astype(np.float32) / A
        lifetime[:, :] = np.where(pcounts > 0, tlifetime / pcounts, 0)
        # lifetime[:, :] = np.divide(tlifetime, pcounts.astype(np.float32), out=np.zeros_like(tlifetime), where=pcounts>0)
        lifetime = np.round(lifetime / sec_per_day, decimals=3)
        rel_density = density / float(N)

        pcounts_minmax = [min(pcounts_minmax[0], pcounts.min()), max(pcounts_minmax[1], pcounts.max())]
        pcounts_statistics[0] += pcounts.mean()
        pcounts_statistics[1] += pcounts.std()
        pcounts_file_ds.resize((ti+1), axis=0)
        pcounts_file_ds[ti, :, :] = pcounts

        density_minmax = [min(density_minmax[0], density.min()), max(density_minmax[1], density.max())]
        density_statistics[0] += density.mean()
        density_statistics[1] += density.std()
        density_file_ds.resize((ti+1), axis=0)
        density_file_ds[ti, :, :] = density

        rel_density_minmax = [min(rel_density_minmax[0], rel_density.min()), max(rel_density_minmax[1], rel_density.max())]
        rel_density_statistics[0] += rel_density.mean()
        rel_density_statistics[1] += rel_density.std()
        rel_density_file_ds.resize((ti+1), axis=0)
        rel_density_file_ds[ti, :, :] = rel_density

        lifetime_minmax = [min(lifetime_minmax[0], lifetime.min()), max(lifetime_minmax[1], lifetime.max())]
        lifetime_statistics[0] += lifetime.mean()
        lifetime_statistics[1] += lifetime.std()
        lifetime_file_ds.resize((ti+1), axis=0)
        lifetime_file_ds[ti, :, :] = lifetime

        del tlifetime
        del xpts
        del ypts
        del x_in
        del y_in
    print("\nField Interpolation done.")
    # rel_density = density / float(N)
    data_xarray.close()
    if DBG_MSG:
        print("Particle Count info: shape = {} type = {} range = {}".format(pcounts_file_ds.shape, pcounts_file_ds.dtype, pcounts_minmax))
        print("Density info: shape = {} type = {} range = ({}, {})".format(density_file_ds.shape, density_file_ds.dtype, density.min(), density.max()))
        print("Lifetime info: shape = {} type = {} range = ({}, {})".format(lifetime_file_ds.shape, lifetime_file_ds.dtype, lifetime.min(), lifetime.max()))

    # pcounts_file = h5py.File(os.path.join(odir, "particlecount.h5"), "w")
    # pcounts_file_ds = pcounts_file.create_dataset("pcount", data=pcounts, compression="gzip", compression_opts=4)
    pcounts_file_ds.attrs['min'] = pcounts_minmax[0]
    pcounts_file_ds.attrs['max'] = pcounts_minmax[1]
    pcounts_file_ds.attrs['mean'] = pcounts_statistics[0] / float(fT.shape[1])
    pcounts_file_ds.attrs['std'] = pcounts_statistics[1] / float(fT.shape[1])
    pcounts_file.close()

    # density_file = h5py.File(os.path.join(odir, "density.h5"), "w")
    # density_file_ds = density_file.create_dataset("density", data=density, compression="gzip", compression_opts=4)
    density_file_ds.attrs['min'] = density_minmax[0]
    density_file_ds.attrs['max'] = density_minmax[1]
    density_file_ds.attrs['mean'] = density_statistics[0] / float(fT.shape[1])
    density_file_ds.attrs['std'] = density_statistics[1] / float(fT.shape[1])
    density_file.close()

    # rel_density_file = h5py.File(os.path.join(odir, "rel_density.h5"), "w")
    # rel_density_file_ds = rel_density_file.create_dataset("rel_density", data=rel_density, compression="gzip", compression_opts=4)
    rel_density_file_ds.attrs['min'] = rel_density_minmax[0]
    rel_density_file_ds.attrs['max'] = rel_density_minmax[1]
    rel_density_file_ds.attrs['mean'] = rel_density_statistics[0] / float(fT.shape[1])
    rel_density_file_ds.attrs['std'] = rel_density_statistics[1] / float(fT.shape[1])
    rel_density_file.close()

    # lifetime_file = h5py.File(os.path.join(odir, "lifetime.h5"), "w")
    # lifetime_file_ds = lifetime_file.create_dataset("lifetime", data=lifetime, compression="gzip", compression_opts=4)
    lifetime_file_ds.attrs['min'] = lifetime_minmax[0]
    lifetime_file_ds.attrs['max'] = lifetime_minmax[1]
    lifetime_file_ds.attrs['mean'] = lifetime_statistics[0] / float(fT.shape[1])
    lifetime_file_ds.attrs['std'] = lifetime_statistics[1] / float(fT.shape[1])
    lifetime_file.close()

    n_valid_pts = np.nonzero(mask_array)[0].shape[0]
    print("Number valid particles: {} (of total {})".format(n_valid_pts, fX.shape[0]))
    particle_file = h5py.File(os.path.join(odir, "particles.h5"), "w")
    px_ds = particle_file.create_dataset("p_x", shape=(n_valid_pts, 1), dtype=fX.dtype, maxshape=(n_valid_pts, fT.shape[1]), compression="gzip", compression_opts=4)
    # px_ds = particle_file.create_dataset("p_x", data=np.array(fX[mask_array, :]), compression="gzip", compression_opts=4)
    px_ds.attrs['unit'] = "arc degree"
    px_ds.attrs['name'] = 'longitude'
    px_ds.attrs['min'] = fX.min()
    px_ds.attrs['max'] = fX.max()
    py_ds = particle_file.create_dataset("p_y", shape=(n_valid_pts, 1), dtype=fY.dtype, maxshape=(n_valid_pts, fT.shape[1]), compression="gzip", compression_opts=4)
    # py_ds = particle_file.create_dataset("p_y", data=np.array(fY[mask_array, :]), compression="gzip", compression_opts=4)
    py_ds.attrs['unit'] = "arc degree"
    py_ds.attrs['name'] = 'latitude'
    py_ds.attrs['min'] = fY.min()
    py_ds.attrs['max'] = fY.max()
    pz_ds = None
    if fZ is not None:
        pz_ds = particle_file.create_dataset("p_z", shape=(n_valid_pts, 1), dtype=fZ.dtype, maxshape=(n_valid_pts, fT.shape[1]), compression="gzip", compression_opts=4)
        # pz_ds = particle_file.create_dataset("p_z", data=np.array(fZ[mask_array, :]), compression="gzip", compression_opts=4)
        pz_ds.attrs['unit'] = "metres"
        pz_ds.attrs['name'] = 'depth'
        pz_ds.attrs['min'] = fZ.min()
        pz_ds.attrs['max'] = fZ.max()
    pt_ds = particle_file.create_dataset("p_t", shape=(n_valid_pts, 1), dtype=fT.dtype, maxshape=(n_valid_pts, fT.shape[1]), compression="gzip", compression_opts=4)
    # pt_ds = particle_file.create_dataset("p_t", data=np.array(fT[mask_array, :]), compression="gzip", compression_opts=4)
    pt_ds.attrs['unit'] = "seconds"
    pt_ds.attrs['name'] = 'time'
    pt_ds.attrs['min'] = fT.min()
    pt_ds.attrs['max'] = fT.max()
    pt_ds.attrs['time_in_min'] = np.nanmin(global_fT, axis=0)
    pt_ds.attrs['time_in_max'] = np.nanmax(global_fT, axis=0)
    # page_ds = particle_file.create_dataset("p_age", shape=(n_valid_pts, 1), dtype=fA.dtype, maxshape=(n_valid_pts, fT.shape[1]), compression="gzip", compression_opts=4)
    # # page_ds = particle_file.create_dataset("p_age", data=np.array(fA[mask_array, :]), compression="gzip", compression_opts=4)
    # page_ds.attrs['unit'] = "seconds"
    # page_ds.attrs['name'] = 'age'
    # page_ds.attrs['min'] = fA.min()
    # page_ds.attrs['max'] = fA.max()
    page_ds = particle_file.create_dataset("p_age", shape=(n_valid_pts, 1), dtype=fAd.dtype, maxshape=(n_valid_pts, fT.shape[1]), compression="gzip", compression_opts=4)
    # page_ds = particle_file.create_dataset("p_age", data=np.array(fA[mask_array, :]), compression="gzip", compression_opts=4)
    page_ds.attrs['unit'] = "days"
    page_ds.attrs['name'] = 'age'
    page_ds.attrs['min'] = fAd.min()
    page_ds.attrs['max'] = fAd.max()

    total_items = fT.shape[0] * fT.shape[1]
    for ti in range(fT.shape[1]):
        x_in = np.array(fX[:, ti])[mask_array]
        y_in = np.array(fY[:, ti])[mask_array]
        z_in = None
        if fZ is not None:
            z_in = np.array(fZ[:, ti])[mask_array]
        t_in = np.array(fT[:, ti])[mask_array]
        # a_in = np.array(fA[:, ti])[mask_array]
        a_in = np.array(fAd[:, ti])[mask_array]

        px_ds.resize((n_valid_pts, ti+1))
        px_ds[:, ti] = x_in
        py_ds.resize(ti+1, axis=1)
        py_ds[:, ti] = y_in
        if fZ is not None:
            pz_ds.resize(ti+1, axis=1)
            pz_ds[:, ti] = z_in
        pt_ds.resize(ti+1, axis=1)
        pt_ds[:, ti] = t_in
        page_ds.resize(ti+1, axis=1)
        page_ds[:, ti] = a_in

        del x_in
        del y_in
        if fZ is not None:
            del z_in
        del t_in
        del a_in

    particle_file.close()
    del rel_density
    del lifetime
    del density
    del pcounts
    del fX
    del fY
    del fZ
    del fA
    del fAd

    xval = np.arange(start=-a*0.5, stop=a*0.5, step=step, dtype=np.float32)
    yval = np.arange(start=-b*0.5, stop=b*0.5, step=step, dtype=np.float32)
    zval = np.arange(start=0.0, stop=c, step=zstep, dtype=np.float32)
    centers_x = xval + step/2.0
    centers_y = yval + step/2.0
    centers_z = zval + zstep/2.0
    us = np.zeros((centers_y.shape[0], centers_x.shape[0], centers_z.shape[0]))
    vs = np.zeros((centers_y.shape[0], centers_x.shape[0], centers_z.shape[0]))
    ws = np.zeros((centers_y.shape[0], centers_x.shape[0], centers_z.shape[0]))

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
    grid_dep_ds = grid_file.create_dataset("depth", data=centers_z, compression="gzip", compression_opts=4)
    grid_dep_ds.attrs['unit'] = "metres"
    grid_dep_ds.attrs['name'] = 'depth'
    grid_dep_ds.attrs['min'] = centers_z.min()
    grid_dep_ds.attrs['max'] = centers_z.max()
    grid_time_ds = grid_file.create_dataset("times", data=global_fT, compression="gzip", compression_opts=4)
    grid_time_ds.attrs['unit'] = "seconds"
    grid_time_ds.attrs['name'] = 'time'
    grid_time_ds.attrs['min'] = np.nanmin(global_fT)
    grid_time_ds.attrs['max'] = np.nanmax(global_fT)
    grid_file.close()

    us_minmax = [0., 0.]
    us_statistics = [0., 0.]
    us_file = h5py.File(os.path.join(odir, "hydrodynamic_U.h5"), "w")
    us_file_ds = us_file.create_dataset("uo", shape=(1, us.shape[0], us.shape[1], us.shape[2]), dtype=us.dtype, maxshape=(fT.shape[1], us.shape[0], us.shape[1], us.shape[2]), compression="gzip", compression_opts=4)
    us_file_ds.attrs['unit'] = "m/s"
    us_file_ds.attrs['name'] = 'meridional_velocity'

    vs_minmax = [0., 0.]
    vs_statistics = [0., 0.]
    vs_file = h5py.File(os.path.join(odir, "hydrodynamic_V.h5"), "w")
    vs_file_ds = vs_file.create_dataset("vo", shape=(1, vs.shape[0], vs.shape[1], vs.shape[2]), dtype=vs.dtype, maxshape=(fT.shape[1], vs.shape[0], vs.shape[1], vs.shape[2]), compression="gzip", compression_opts=4)
    vs_file_ds.attrs['unit'] = "m/s"
    vs_file_ds.attrs['name'] = 'zonal_velocity'

    ws_minmax = [0., 0.]
    ws_statistics = [0., 0.]
    ws_file = h5py.File(os.path.join(odir, "hydrodynamic_W.h5"), "w")
    ws_file_ds = ws_file.create_dataset("wo", shape=(1, ws.shape[0], ws.shape[1], ws.shape[2]), dtype=ws.dtype, maxshape=(fT.shape[1], ws.shape[0], ws.shape[1], ws.shape[2]), compression="gzip", compression_opts=4)
    ws_file_ds.attrs['unit'] = "m/s"
    ws_file_ds.attrs['name'] = 'vertical_velocity'

    print("Sampling UVW on NEMO grid ...")
    # sample_lats = centers_y
    # sample_lons = centers_x
    sample_time = 0
    # sample_func = sample_uv
    fieldset = create_NEMO_fieldset(datahead=datahead, periodic_wrap=True, chunk=args.chs)
    p_center_z, p_center_y, p_center_x = np.meshgrid(centers_z, centers_y, centers_x, sparse=False, indexing='ij')
    sample_pset = ParticleSet(fieldset=fieldset, pclass=SampleParticle, lon=np.array(p_center_x).flatten(), lat=np.array(p_center_y).flatten(), depth=np.array(p_center_z).flatten(), time=sample_time)
    sample_kernel = sample_pset.Kernel(sample_uvw)
    sample_outname = out_fname + "_sampleuvw"
    sample_output_file = sample_pset.ParticleFile(name=os.path.join(odir,sample_outname+".nc"), outputdt=delta(minutes=outdt_minutes))
    if backwardSimulation:
        sample_pset.execute(sample_kernel, runtime=delta(days=time_in_days), dt=delta(minutes=-dt_minutes), output_file=sample_output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(minutes=outdt_minutes))
    else:
        sample_pset.execute(sample_kernel, runtime=delta(days=time_in_days), dt=delta(minutes=dt_minutes), output_file=sample_output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(minutes=outdt_minutes))
    sample_output_file.close()
    del sample_output_file
    del sample_pset
    del sample_kernel
    del fieldset
    print("UVW on NEMO grid sampled.")

    print("Load sampled data ...")
    sample_xarray = xr.open_dataset(os.path.join(odir, sample_outname + ".nc"))
    N_s = sample_xarray['lon'].shape[0]
    tN_s = sample_xarray['lon'].shape[1]
    print("N: {}, t_N: {}".format(N_s, tN_s))
    valid_array = np.maximum(np.max(np.array(sample_xarray['valid'][:, 0:2]), axis=1), 0).astype(np.bool_)
    if DBG_MSG:
        print("Valid array: any true ? {}; all true ? {}".format(valid_array.any(), valid_array.all()))
    ctime_array_s = sample_xarray['time'].data
    time_in_min_s = np.nanmin(ctime_array_s, axis=0)
    time_in_max_s = np.nanmax(ctime_array_s, axis=0)
    assert ctime_array_s.shape[1] == time_in_min.shape[0]
    # mask_array_s = np.array([True,] * ctime_array_s.shape[0])  # this array is used to separate valid ocean particles (True) from invalid ones (e.g. land; False)
    mask_array_s = valid_array
    for ti in range(ctime_array_s.shape[1]):
        replace_indices = np.isnan(ctime_array_s[:, ti])
        # if (ti < 3):
        #     reverse_replace = ~replace_indices
        #     mask_array_s &= reverse_replace
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
    elif 'depthu' in sample_xarray.keys():
        psZ = sample_xarray['depthu']  # to be loaded from pfile
    elif 'depthv' in sample_xarray.keys():
        psZ = sample_xarray['depthv']  # to be loaded from pfile
    elif 'depthw' in sample_xarray.keys():
        psZ = sample_xarray['depthw']  # to be loaded from pfile
    elif 'z' in sample_xarray.keys():
        psZ = sample_xarray['z']  # to be loaded from pfile
    psT = dtime_array_s
    global_psT = time_in_max_s -time_in_max_s[0]
    # this really need to convert to float
    # if isinstance(psT[0, 0], datetime.datetime) or isinstance(psT[0, 0], datetime.timedelta) or isinstance(psT[0, 0], np.timedelta64) or isinstance(psT[0, 0], np.datetime64) or np.float64(psT[0, 1] - psT[0, 0]) > (dt_minutes + dt_minutes / 2.0):
    ## if isinstance(psT[0,0], datetime.datetime) or isinstance(psT[0, 0], datetime.timedelta) or np.float64(psT[0, 1]-psT[0, 0]) > (dt_minutes+dt_minutes/2.0):
    #     print("psT.dtype before conversion: {}".format(psT.dtype))
    #     psT = (psT / ns_per_sec).astype(np.float64)  # to be loaded from pfile
    #     global_psT = (global_psT / ns_per_sec).astype(np.float64)
    #     print("psT.range and ft.dtype after conversion: ({}, {}) \t {}".format(psT[0].min(), psT[0].max(), psT.dtype))
    psT = convert_timearray(psT, outdt_minutes*60, ns_per_sec, debug=DBG_MSG, array_name="psT")
    global_psT = convert_timearray(global_psT, outdt_minutes*60, ns_per_sec, debug=DBG_MSG, array_name="global_psT")
    psU = sample_xarray['sample_u']  # to be loaded from pfile
    psV = sample_xarray['sample_v']  # to be loaded from pfile
    psW = sample_xarray['sample_w']  # to be loaded from pfile
    print("Sampled data loaded.")

    print("Interpolating UV on a regular-square grid ...")
    total_items = psT.shape[1]
    for ti in range(psT.shape[1]):
        us_local = np.expand_dims(psU[:, ti], axis=1)
        us_local[~mask_array_s, :] = 0
        vs_local = np.expand_dims(psV[:, ti], axis=1)
        vs_local[~mask_array_s, :] = 0
        ws_local = np.expand_dims(psW[:, ti], axis=1)
        ws_local[~mask_array_s, :] = 0
        if ti == 0 and DBG_MSG:
            print("us.shape {}; us_local.shape: {}; psU.shape: {}; p_center_y.shape: {}".format(us.shape, us_local.shape, psU.shape, p_center_y.shape))

        us[:, :] = np.reshape(us_local, p_center_x.shape)
        vs[:, :] = np.reshape(vs_local, p_center_y.shape)
        ws[:, :] = np.reshape(ws_local, p_center_z.shape)

        us_minmax = [min(us_minmax[0], us.min()), max(us_minmax[1], us.max())]
        us_statistics[0] += us.mean()
        us_statistics[1] += us.std()
        us_file_ds.resize((ti+1), axis=0)
        us_file_ds[ti, :, :] = us
        vs_minmax = [min(vs_minmax[0], vs.min()), max(vs_minmax[1], vs.max())]
        vs_statistics[0] += vs.mean()
        vs_statistics[1] += vs.std()
        vs_file_ds.resize((ti+1), axis=0)
        vs_file_ds[ti, :, :] = vs
        ws_minmax = [min(ws_minmax[0], ws.min()), max(ws_minmax[1], ws.max())]
        ws_statistics[0] += ws.mean()
        ws_statistics[1] += ws.std()
        ws_file_ds.resize((ti+1), axis=0)
        ws_file_ds[ti, :, :] = ws

        # del us_local
        # del vs_local
        current_item = ti
        workdone = current_item / total_items
        print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
    print("\nFinished UV-interpolation.")

    us_file_ds.attrs['min'] = us_minmax[0]
    us_file_ds.attrs['max'] = us_minmax[1]
    us_file_ds.attrs['mean'] = us_statistics[0] / float(fT.shape[1])
    us_file_ds.attrs['std'] = us_statistics[1] / float(fT.shape[1])
    us_file.close()
    vs_file_ds.attrs['min'] = vs_minmax[0]
    vs_file_ds.attrs['max'] = vs_minmax[1]
    vs_file_ds.attrs['mean'] = vs_statistics[0] / float(fT.shape[1])
    vs_file_ds.attrs['std'] = vs_statistics[1] / float(fT.shape[1])
    vs_file.close()
    ws_file_ds.attrs['min'] = ws_minmax[0]
    ws_file_ds.attrs['max'] = ws_minmax[1]
    ws_file_ds.attrs['mean'] = ws_statistics[0] / float(fT.shape[1])
    ws_file_ds.attrs['std'] = ws_statistics[1] / float(fT.shape[1])
    ws_file.close()

    del centers_x
    del centers_y
    del centers_z
    del xval
    del yval
    del zval
    del fT
    del global_fT
    del dtime_array
    del ctime_array
