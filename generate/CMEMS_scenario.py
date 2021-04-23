"""
Author: Dr. Christian Kehl
Date: 11-02-2020
"""

from parcels import AdvectionEE, AdvectionRK45, AdvectionRK4, AdvectionRK4DiffusionEM, AdvectionRK4DiffusionM1
from parcels import FieldSet, ScipyParticle, JITParticle, Variable, AdvectionRK4, StateCode, OperationCode, ErrorCode
# from parcels.particleset_benchmark import ParticleSet_Benchmark as BenchmarkParticleSet
# from parcels.particleset import ParticleSet as DryParticleSet
from parcels.particleset import ParticleSet as BenchmarkParticleSet
from parcels.field import Field, VectorField, NestedField, SummedField
# from parcels import plotTrajectoriesFile_loadedField
# from parcels import rng as random
from parcels import ParcelsRandom
from datetime import timedelta as delta
import math
from argparse import ArgumentParser
import datetime
import numpy as np
from numpy.random import default_rng
import xarray as xr
import fnmatch
import sys
import gc
import os
import time as ostime
# import matplotlib.pyplot as plt
from parcels.tools import perlin3d
from parcels.tools import perlin2d
from scipy.interpolate import interpn
from glob import glob
import h5py

try:
    from mpi4py import MPI
except:
    MPI = None
with_GC = False

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

#a = 1000 * 1e3
#b = 1000 * 1e3
#scalefac = 2.0
#tsteps = 61
#tscale = 6

# a = 9.6 * 1e3  # by definition: meters
# b = 4.8 * 1e3  # by definiton: meters
a = 3.6 * 1e2  # by definition: meters
b = 1.8 * 1e2  # by definiton: meters
tsteps = 122 # in steps
tstepsize = 3.0 # unitary
tscale = 24.0*60.0*60.0 # in seconds
# time[days] = (tsteps * tstepsize) * tscale
# ==== GYRE ROTATION SPEED ==== #
# gyre_rotation_speed = 60.0*24.0*60.0*60.0  # assume 1 rotation every 8.5 weeks
# gyre_rotation_speed = (366.0*24.0*60.0*60.0)/2.0  # assume 1 rotation every 26 weeks
# gyre_rotation_speed = 366.0*24.0*60.0*60.0  # assume 1 rotation every 52 weeks
# ==== INFO FROM NEMO-MEDUSA: realistic values are 0-2.5 [m/s] ==== #
# scalefac = (40.0 / (1000.0/60.0))  # 40 km/h
# scalefactor = (2.0 / (1000.0/60.0))  #  -> 40 kph far to fast on scale max. 2 km. -> 2 km/h
# scalefactor = (6.0 / (1000.0/60.0))  #  -> 40 kph far to fast on scale max. 2 km. -> 6 km/h
# ==== ONLY APPLY BELOW SCALING IF MESH IS FLAT AND (a, b) are below 100,000 [m] ==== #
# v_scale_small = 1./1000.0 # this is to adapt, cause 1 U = 1 m/s = 1 spatial unit / time unit; spatial scale; domain = 1920 m x 960 m -> scale needs to be adapted to to interpret speed on a 1920 km x 960 km grid

# we need to modify the kernel.execute / pset.execute so that it returns from the JIT
# in a given time WITHOUT writing to disk via outfie => introduce a pyloop_dt

def DeleteParticle(particle, fieldset, time):
    particle.delete()

def RenewParticle(particle, fieldset, time):
    xe = 3.6 * 1e2
    ye = 1.8 * 1e2
    # particle.lat = np.random.rand() * (-a) + (a/2.0)
    # particle.lon = np.random.rand() * (-b) + (b/2.0)
    if particle.lon >= (xe/2.0):
        particle.lon -= xe
    if particle.lon < (-xe/2.0):
        particle.lon += xe
    # particle.lat = min(particle.lat, ye/2.0 - ye/1000.0)
    # particle.lat = max(particle.lat, -ye/2.0 + ye/1000.0)
    if particle.lat >= (ye/2.0):
    #     particle.lat -= b
        particle.lat = ye / 2.0 - (abs(particle.lat) - ye/2.0)
    if particle.lat < (-ye/2.0):
    #     particle.lat += b
        particle.lat = -ye / 2.0 + (abs(particle.lat) - ye/2.0)

periodicBC = RenewParticle

def perIterGC():
    gc.collect()

def create_CMEMS_fieldset(datahead, periodic_wrap):
    ddir = os.path.join(datahead, "CMEMS/GLOBAL_REANALYSIS_PHY_001_030/")
    files = sorted(glob(ddir+"mercatorglorys12v1_gl12_mean_201607*.nc"))
    variables = {'U': 'uo', 'V': 'vo'}
    dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    # Kh_zonal = np.ones(U.shape, dtype=np.float32) * 0.5  # ?
    # Kh_meridional = np.ones(U.shape, dtype=np.float32) * 0.5  # ?
    # fieldset.add_constant_field("Kh_zonal", 1, mesh="flat")
    # fieldset.add_constant_field("Kh_meridonal", 1, mesh="flat")
    global ttotal
    ttotal = 31  # days
    # chs = False
    chs = 'auto'
    fieldset = None
    if periodic_wrap:
        fieldset = FieldSet.from_netcdf(files, variables, dimensions, chunksize=chs, time_periodic=delta(days=31))
    else:
        fieldset = FieldSet.from_netcdf(files, variables, dimensions, chunksize=chs, allow_time_extrapolation=True)
    global tsteps
    tsteps = len(fieldset.U.grid.time_full)
    global tstepsize
    tstepsize = int(math.floor(ttotal/tsteps))
    return fieldset

class SampleParticle(JITParticle):
    sample_u = Variable('sample_u', initial=0., dtype=np.float32, to_write=True)
    sample_v = Variable('sample_v', initial=0., dtype=np.float32, to_write=True)

def sample_uv(particle, fieldset, time):
    particle.sample_u = fieldset.U[time, particle.depth, particle.lat, particle.lon]
    particle.sample_v = fieldset.U[time, particle.depth, particle.lat, particle.lon]
    return StateCode.Success

class AgeParticle_JIT(JITParticle):
    age = Variable('age', dtype=np.float64, initial=0.0, to_write=True)
    life_expectancy = Variable('life_expectancy', dtype=np.float64, initial=np.finfo(np.float64).max, to_write=False)
    initialized_dynamic = Variable('initialized_dynamic', dtype=np.int32, initial=0, to_write=False)

class AgeParticle_SciPy(ScipyParticle):
    age = Variable('age', dtype=np.float64, initial=0.0)
    life_expectancy = Variable('life_expectancy', dtype=np.float64, initial=np.finfo(np.float64).max, to_write=False)
    initialized_dynamic = Variable('initialized_dynamic', dtype=np.int32, initial=0, to_write=False)

def initialize(particle, fieldset, time):
    if particle.initialized_dynamic < 1:
        np_scaler = math.sqrt(3.0 / 2.0)
        particle.life_expectancy = time + ParcelsRandom.uniform(.0, (fieldset.life_expectancy-time) * 2.0 / np_scaler)
        # particle.life_expectancy = time + ParcelsRandom.uniform(.0, (fieldset.life_expectancy-time)*math.sqrt(3.0 / 2.0))
        # particle.life_expectancy = time + ParcelsRandom.uniform(.0, fieldset.life_expectancy) * math.sqrt(3.0 / 2.0)
        # particle.life_expectancy = time+ParcelsRandom.uniform(.0, fieldset.life_expectancy) * ((3.0/2.0)**2.0)
        particle.initialized_dynamic = 1

def Age(particle, fieldset, time):
    if particle.state == StateCode.Evaluate:
        particle.age = particle.age + math.fabs(particle.dt)
    if particle.age > particle.life_expectancy:
        particle.delete()

def sample_regularly_jittered(lon_range, lat_range, res):
    """

    :param lon_range:
    :param lat_range:
    :param res: number of cells (in each direction) per arc degree or metre
    :return:
    """
    samples_lon = []
    samples_lat = []
    jitter = np.random.random(2) * 1/res
    lat_buckets = int(np.floor((lat_range[1]-lat_range[0])*res))-1
    lon_buckets = int(np.floor((lon_range[1]-lon_range[0])*res))-1
    for i in range(lat_buckets):
        for j in range(lon_buckets):
            # sample = [jitter[0]+lon_range[0]+(j*(1/res))/(lon_range[1]-lon_range[0]), jitter[1]+lat_range[0]+(i*(1/res))/(lat_range[1]-lat_range[0])]
            sample = [jitter[0] + lon_range[0] + (j * (1 / res)),
                      jitter[1] + lat_range[0] + (i * (1 / res))]
            samples_lon.append(sample[0])
            samples_lat.append(sample[1])
    # samples_lon = np.unique(samples_lon)
    # samples_lat = np.unique(samples_lat)
    return samples_lon, samples_lat

def rsample(low, high, size, sample_string):
    sample = None
    rng_sampler = default_rng()
    if sample_string == 'uniform':
        sample = rng_sampler.uniform(low, high, size)
    elif sample_string == 'gaussian':
        sample = rng_sampler.normal(0.0, 0.5, size) + 0.5
        sample[sample < 0] = 0.0
        sample[sample > 1] = 1.0
        sample = sample*(high-low) + low
    elif sample_string == 'triangular':
        mid = low + (high-low)/2.0
        sample = rng_sampler.triangular(low, mid, high, size)
    elif sample_string == 'vonmises':
        sample = rng_sampler.vonmises(0, 1/math.sqrt(2.0), size)+0.5
        sample[sample < 0] = 0.0
        sample[sample > 1] = 1.0
        sample = sample*(high-low) + low
    return sample

def sample_irregularly(lon_range, lat_range, res=None, rsampler_str='uniform', nparticle=None):
    """

    :param lon_range:
    :param lat_range:
    :param res: square-root of particles per sauqre-arc degree or metre
    :param nparticle:
    :return:
    """
    samples_lon = []
    samples_lat = []
    if res != None:
        llon = np.floor(lon_range[0])
        llat = np.floor(lat_range[0])
        lat_buckets = int(np.floor((lat_range[1]-lat_range[0])))-1
        lon_buckets = int(np.floor((lon_range[1]-lon_range[0])))-1
        deg_scale = 1.0
        local_nparticle = int(res**2)
        if res<1.0:
            deg_scale = int(np.round(1/res))
            local_nparticle = 1
        for i in range(lat_buckets):
            for j in range(lon_buckets):
                local_samples = rsample([llon+(j*deg_scale), llat+(i*deg_scale)], [llon+(j+1)*deg_scale, llat+((i+1)*deg_scale)], local_nparticle, rsampler_str)
                samples_lon.append(local_samples[0,:])
                samples_lat.append(local_samples[1,:])
    else:
        assert type(nparticle) in (int, np.int32, np.uint32, np.int64, np.uint64)
        samples_lon = np.random.uniform(lon_range[0], lon_range[1], nparticle)
        samples_lat = np.random.uniform(lat_range[0], lat_range[1], nparticle)
    return samples_lon, samples_lat

def sample_particles(lon_range, lat_range, res=None, rsampler_str='regular_jitter', nparticle=None):
    if rsampler_str == 'regular_jitter':
        assert res is not None
        return sample_regularly_jittered(lon_range, lat_range, res)
    else:
        return sample_irregularly(lon_range, lat_range, res, rsampler_str, nparticle)

age_ptype = {'scipy': AgeParticle_SciPy, 'jit': AgeParticle_JIT}

# ====
# start example: python3 doublegyre_scenario.py -f NNvsGeostatistics/data/file.txt -t 30 -dt 720 -ot 1440 -im 'rk4' -N 2**12 -sres 2 -sm 'regular_jitter'
#                python3 doublegyre_scenario.py -f NNvsGeostatistics/data/file.txt -t 366 -dt 720 -ot 2880 -im 'rk4' -N 2**12 -sres 2.5 -gres 5 -sm 'regular_jitter' -fsx 360 -fsy 180
# ====
if __name__=='__main__':
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-f", "--filename", dest="filename", type=str, default="file.txt", help="(relative) (text) file path of the csv hyper parameters")
    # parser.add_argument("-b", "--backwards", dest="backwards", action='store_true', default=False, help="enable/disable running the simulation backwards")
    # parser.add_argument("-p", "--periodic", dest="periodic", action='store_true', default=False, help="enable/disable periodic wrapping (else: extrapolation)")
    # parser.add_argument("-r", "--release", dest="release", action='store_true', default=False, help="continuously add particles via repeatdt (default: False)")
    # parser.add_argument("-rt", "--releasetime", dest="repeatdt", type=int, default=720, help="repeating release rate of added particles in Minutes (default: 720min = 12h)")
    # parser.add_argument("-a", "--aging", dest="aging", action='store_true', default=False, help="Removed aging particles dynamically (default: False)")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=1, help="runtime in days (default: 1)")
    parser.add_argument("-dt", "--deltatime", dest="dt", type=int, default=720, help="computational delta_t time stepping in minutes (default: 720min = 12h)")
    parser.add_argument("-ot", "--outputtime", dest="outdt", type=int, default=1440, help="repeating release rate of added particles in minutes (default: 1440min = 24h)")
    parser.add_argument("-im", "--interp_mode", dest="interp_mode", choices=['rk4','rk45', 'ee', 'em', 'm1'], default="jit", help="interpolation mode = [rk4, rk45, ee (Eulerian Estimation), em (Euler-Maruyama), m1 (Milstein-1)]")
    # parser.add_argument("-x", "--xarray", dest="use_xarray", action='store_true', default=False, help="use xarray as data backend")
    # parser.add_argument("-w", "--writeout", dest="write_out", action='store_true', default=False, help="write data in outfile")
    # parser.add_argument("-d", "--delParticle", dest="delete_particle", action='store_true', default=False, help="switch to delete a particle (True) or reset a particle (default: False).")
    # parser.add_argument("-A", "--animate", dest="animate", action='store_true', default=False, help="animate the particle trajectories during the run or not (default: False).")
    # parser.add_argument("-V", "--visualize", dest="visualize", action='store_true', default=False, help="Visualize particle trajectories at the end (default: False). Requires -w in addition to take effect.")
    parser.add_argument("-N", "--n_particles", dest="nparticles", type=str, default="2**6", help="number of particles to generate and advect (default: 2e6)")
    # parser.add_argument("-sN", "--start_n_particles", dest="start_nparticles", type=str, default="96", help="(optional) number of particles generated per release cycle (if --rt is set) (default: 96)")
    # parser.add_argument("-m", "--mode", dest="compute_mode", choices=['jit','scipy'], default="jit", help="computation mode = [JIT, SciPp]")
    parser.add_argument("-sres", "--sample_resolution", dest="sres", type=str, default="2", help="number of particle samples per arc-dgree (default: 2)")
    parser.add_argument("-gres", "--grid_resolution", dest="gres", type=str, default="10", help="number of cells per arc-degree or metre (default: 10)")
    parser.add_argument("-sm", "--samplemode", dest="sample_mode", choices=['regular_jitter','uniform','gaussian','triangular','vonmises'], default='regular_jitter', help="sampling distribution mode = [regular_jitter, (irregular) uniform, (irregular) gaussian, (irregular) triangular, (irregular) vonmises]")
    parser.add_argument("-fsx", "--field_sx", dest="field_sx", type=int, default="480", help="number of original field cells in x-direction")
    parser.add_argument("-fsy", "--field_sy", dest="field_sy", type=int, default="240", help="number of original field cells in y-direction")
    args = parser.parse_args()

    ParticleSet = BenchmarkParticleSet
    # if args.dryrun:
    #     ParticleSet = DryParticleSet

    filename=args.filename
    field_sx = args.field_sx
    field_sy = args.field_sy
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

    branch = "soa_benchmark"
    computer_env = "local/unspecified"
    scenario = "doublegyre"
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
        dirread_top = os.path.join(datahead, 'CMEMS/GLOBAL_REANALYSIS_PHY_001_030/')
        computer_env = "Gemini"
    elif fnmatch.fnmatchcase(os.uname()[1], "*.bullx*"):  # Cartesius
        CARTESIUS_SCRATCH_USERNAME = 'ckehluu'
        headdir = "/scratch/shared/{}/experiments".format(CARTESIUS_SCRATCH_USERNAME)
        odir = headdir
        datahead = "/projects/0/topios/hydrodynamic_data"
        dirread_top = os.path.join(datahead, 'CMEMS/GLOBAL_REANALYSIS_PHY_001_030/')
        computer_env = "Cartesius"
    else:
        headdir = "/var/scratch/experiments"
        odir = headdir
        dirread_pal = headdir
        datahead = "/data"
        dirread_top = os.path.join(datahead, 'CMEMS/GLOBAL_REANALYSIS_PHY_001_030/')
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

    np.random.seed(0)
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
    fieldset = create_CMEMS_fieldset(datahead=datahead, periodic_wrap=True)

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
        fieldset.add_constant('life_expectancy', delta(days=time_in_days).total_seconds())
    else:
        # making sure we do track age, but life expectancy is a hard full simulation time #
        fieldset.add_constant('life_expectancy', delta(days=time_in_days).total_seconds())
        age_ptype[(compute_mode).lower()].life_expectancy.initial = delta(days=time_in_days).total_seconds()
        age_ptype[(compute_mode).lower()].initialized_dynamic.initial = 1

    if repeatdtFlag:
        add_scaler = start_scaler/2.0
        addParticleN = Nparticle/2.0
        refresh_cycle = (delta(days=time_in_days).total_seconds() / (addParticleN/start_N_particles)) / cycle_scaler
        if agingParticles:
            refresh_cycle /= cycle_scaler
        repeatRateMinutes = int(refresh_cycle/60.0) if repeatRateMinutes == 720 else repeatRateMinutes

    print("Sampling the grid and creating the particle set now ...")
    if backwardSimulation:
        # ==== backward simulation ==== #
        if agingParticles:
            if repeatdtFlag:
                startlon, startlat = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), int(np.floor(sres*start_scaler)), sample_mode, None)
                repeatlon, repeatlat = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), int(np.floor(sres*add_scaler)), sample_mode, None)
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=startlon, lat=startlat, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                psetA = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=repeatlon, lat=repeatlat, time=simStart)
                pset.add(psetA)
            else:
                lons, lats = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), sres, sample_mode, None)
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=lons, lat=lats, time=simStart)
        else:
            if repeatdtFlag:
                startlon, startlat = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), int(np.floor(sres*start_scaler)), sample_mode, None)
                repeatlon, repeatlat = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), int(np.floor(sres*add_scaler)), sample_mode, None)
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=startlon, lat=startlat, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                psetA = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=repeatlon, lat=repeatlat, time=simStart)
                pset.add(psetA)
            else:
                lons, lats = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), sres, sample_mode, None)
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=lons, lat=lats, time=simStart)
    else:
        # ==== forward simulation ==== #
        if agingParticles:
            if repeatdtFlag:
                startlon, startlat = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), int(np.floor(sres*start_scaler)), sample_mode, None)
                repeatlon, repeatlat = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), int(np.floor(sres*add_scaler)), sample_mode, None)
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=startlon, lat=startlat, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                psetA = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=repeatlon, lat=repeatlat, time=simStart)
                pset.add(psetA)
            else:
                lons, lats = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), sres, sample_mode, None)
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=lons, lat=lats, time=simStart)
        else:
            if repeatdtFlag:
                startlon, startlat = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), int(np.floor(sres*start_scaler)), sample_mode, None)
                repeatlon, repeatlat = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), int(np.floor(sres*add_scaler)), sample_mode, None)
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=startlon, lat=startlat, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                psetA = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=repeatlon, lat=repeatlat, time=simStart)
                pset.add(psetA)
            else:
                lons, lats = sample_particles((-a/2.0, a/2.0), (-b/2.0, b/2.0), sres, sample_mode, None)
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(compute_mode).lower()], lon=lons, lat=lats, time=simStart)
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
    out_fname = "benchmark_CMEMS"
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
        output_file = pset.ParticleFile(name=os.path.join(odir,out_fname+".nc"), outputdt=delta(minutes=outdt_minutes))

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
        kernelfunc = AdvectionRK4DiffusionEM
    elif interp_mode == 'm1':
        kernelfunc = AdvectionRK4DiffusionM1

    kernels = pset.Kernel(kernelfunc,delete_cfiles=True)
    # if agingParticles:
    if True:
        kernels += pset.Kernel(initialize, delete_cfiles=True)
        kernels += pset.Kernel(Age, delete_cfiles=True)
    kernels += pset.Kernel(periodicBC, delete_cfiles=True)  # insert here to correct for boundary conditions right after advection)

    postProcessFuncs.append(perIterGC)
    if backwardSimulation:
        # ==== backward simulation ==== #
        if animate_result:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=-dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=outdt_minutes), moviedt=delta(hours=6), movie_background_field=fieldset.U)
        else:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=-dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=outdt_minutes))
    else:
        # ==== forward simulation ==== #
        if animate_result:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=outdt_minutes), moviedt=delta(hours=6), movie_background_field=fieldset.U)
        else:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=outdt_minutes))

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
    xsteps = int(np.floor(a * gres))
    ysteps = int(np.floor(b * gres))
    # ------------------------------ #
    # -- ISSUE: REMOVE OR REPLACE NAN's -- #

    data_xarray = xr.open_dataset(os.path.join(odir, out_fname + ".nc"))
    N = data_xarray['lon'].data.shape[0]
    tN = data_xarray['lon'].data.shape[1]
    print("N: {}, t_N: {}".format(N, tN))
    np.set_printoptions(linewidth=160)
    ns_per_sec = np.timedelta64(1, 's')  # nanoseconds in an sec
    print("ns_per_sec = {}".format((ns_per_sec/np.timedelta64(1, 'ns')).astype(np.float64)))
    sec_per_day = 86400.0
    ctime_array = data_xarray['time'].data
    time_in_min = np.nanmin(ctime_array, axis=0)
    time_in_max = np.nanmax(ctime_array, axis=0)
    print("Times:\n\tmin = {}\n\tmax = {}".format(time_in_min, time_in_max))
    assert ctime_array.shape[1] == time_in_min.shape[0]
    for ti in range(ctime_array.shape[1]):
        replace_indices = np.isnan(ctime_array[:, ti])
        ctime_array[replace_indices, ti] = time_in_max[ti]
    print("time info from file before baselining: shape = {} type = {} range = ({}, {})".format(ctime_array.shape, type(ctime_array[0, 0]), np.min(ctime_array[0]), np.max(ctime_array[0])))
    # timebase = ctime_array[:, 0]
    timebase = time_in_max[0]
    # dtime_array = np.transpose(ctime_array.transpose() - timebase)
    dtime_array = ctime_array - timebase
    print("time info from file after baselining: \n\tshape = {}".format(dtime_array.shape))
    print("\ttype = {}".format(type(dtime_array[0, 0])))
    print("\trange = {}".format( (np.nanmin(dtime_array), np.nanmax(dtime_array)) ))
    # print(dtime_array.dtype)
    # print(ns_per_sec.dtype)
    # pX = data_xarray['lon'].data[indices, :]  # only plot the first 32 particles
    # pY = data_xarray['lat'].data[indices, :]  # only plot the first 32 particles

    fX = data_xarray['lon'].data  # to be loaded from pfile
    fY = data_xarray['lat'].data  # to be loaded from pfile
    fZ = None
    if 'depth' in data_xarray.keys():
        fZ = data_xarray['depth'].data  # to be loaded from pfile
    elif 'z' in data_xarray.keys():
        fZ = data_xarray['z'].data  # to be loaded from pfile
    fT = dtime_array
    global_fT = time_in_max - time_in_max[0]
    # this really need to convert to float
    if isinstance(fT[0,0], datetime.datetime) or isinstance(fT[0, 0], datetime.timedelta) or isinstance(fT[0, 0], np.timedelta64) or isinstance(fT[0, 0], np.datetime64) or np.float64(fT[0, 1]-fT[0, 0]) > (dt_minutes+dt_minutes/2.0):
        print("fT.dtype before conversion: {}".format(fT.dtype))
        fT = (fT / ns_per_sec).astype(np.float64)  # to be loaded from pfile
        global_fT = (global_fT / ns_per_sec).astype(np.float64)
        print("fT.range and ft.dtype after conversion: ({}, {}) \t {}".format(fT[0].min(), fT[0].max(), fT.dtype))
    fA = data_xarray['age'].data  # to be loaded from pfile | age
    for ti in range(fA.shape[1]):
        replace_indices = np.isnan(fA[:, ti])
        replace_values = None
        if ti > 0:
            replace_values = np.fmax(fA[:, ti-1], .0)
        else:
            replace_values = np.zeros((fA.shape[0],), dtype=fA.dtype)
        fA[replace_indices, ti] = replace_values[replace_indices]
    print("original fA.range and fA.dtype (before conversion): ({}, {}) \t {}".format(fA[0].min(), fA[0].max(), fA.dtype))
    if not isinstance(fA[0,0], np.float32):
        # fA = np.transpose(fA.transpose()-timebase)
        fA = fA - timebase
    if isinstance(fA[0, 0], datetime.datetime) or isinstance(fA[0, 0], datetime.timedelta) or isinstance(fT[0, 0], np.timedelta64) or isinstance(fT[0, 0], np.datetime64):  # or np.float32((fA[0, 1]-fA[0, 0])) > (dt_minutes+dt_minutes/2.0):
        print("fA.dtype before conversion: {}".format(fA.dtype))
        fA = (fA / ns_per_sec).astype(np.float32)
        print("fA.range and fA.dtype (after conversion): ({}, {}) \t {}".format(float(fA.min()), float(fA.max()), fA.dtype))

    # pcounts = np.zeros((fT.shape[1], xval.shape[0], yval.shape[0]), dtype=np.int32)
    pcounts = np.zeros((ysteps, xsteps), dtype=np.int32)
    pcounts_minmax = [0., 0.]
    pcounts_statistics = [0., 0.]
    pcounts_file = h5py.File(os.path.join(odir, "particlecount.h5"), "w")
    pcounts_file_ds = pcounts_file.create_dataset("pcount", shape=(1, pcounts.shape[0], pcounts.shape[1]), dtype=pcounts.dtype, maxshape=(fT.shape[1], pcounts.shape[0], pcounts.shape[1]), compression="gzip", compression_opts=4)
    pcounts_file_ds.attrs['unit'] = "count (scalar)"
    pcounts_file_ds.attrs['name'] = 'particle_count'

    # density = np.zeros((fT.shape[1], xval.shape[0], yval.shape[0]), dtype=np.float32)
    density = np.zeros((ysteps, xsteps), dtype=np.float32)
    density_minmax = [0., 0.]
    density_statistics = [0., 0.]
    density_file = h5py.File(os.path.join(odir, "density.h5"), "w")
    density_file_ds = density_file.create_dataset("density", shape=(1, density.shape[0], density.shape[1]), dtype=density.dtype, maxshape=(fT.shape[1], density.shape[0], density.shape[1]), compression="gzip", compression_opts=4)
    density_file_ds.attrs['unit'] = "pts / arc_deg^2"
    density_file_ds.attrs['name'] = 'density'

    # rel_density = np.zeros((fT.shape[1], xval.shape[0], yval.shape[0]), dtype=np.float32)
    rel_density = np.zeros((ysteps, xsteps), dtype=np.float32)
    rel_density_minmax = [0., 0.]
    rel_density_statistics = [0., 0.]
    rel_density_file = h5py.File(os.path.join(odir, "rel_density.h5"), "w")
    rel_density_file_ds = rel_density_file.create_dataset("rel_density", shape=(1, rel_density.shape[0], rel_density.shape[1]), dtype=rel_density.dtype, maxshape=(fT.shape[1], rel_density.shape[0], rel_density.shape[1]), compression="gzip", compression_opts=4)
    rel_density_file_ds.attrs['unit'] = "pts_percentage / arc_deg^2"
    rel_density_file_ds.attrs['name'] = 'relative_density'

    # lifetime = np.zeros((fT.shape[1], xval.shape[0], yval.shape[0]), dtype=np.float32)
    lifetime = np.zeros((ysteps, xsteps), dtype=np.float32)
    lifetime_minmax = [0., 0.]
    lifetime_statistics = [0., 0.]
    lifetime_file = h5py.File(os.path.join(odir, "lifetime.h5"), "w")
    lifetime_file_ds = lifetime_file.create_dataset("lifetime", shape=(1, lifetime.shape[0], lifetime.shape[1]), dtype=lifetime.dtype, maxshape=(fT.shape[1], lifetime.shape[0], lifetime.shape[1]), compression="gzip", compression_opts=4)
    lifetime_file_ds.attrs['unit'] = "avg. lifetime"
    lifetime_file_ds.attrs['name'] = 'lifetime'

    A = float(gres**2)
    total_items = fT.shape[0] * fT.shape[1]
    for ti in range(fT.shape[1]):
        pcounts[:, :] = 0
        density[:, :] = 0
        rel_density[:, :] = 0
        lifetime[:, :] = 0
        tlifetime = np.zeros((ysteps, xsteps), dtype=np.float32)
        xpts = np.floor((fX[:, ti]+(a/2.0))*gres).astype(np.int32).flatten()
        xpts = np.maximum(np.minimum(xpts, xsteps - 1), 0)
        ypts = np.floor((fY[:, ti]+(b/2.0))*gres).astype(np.int32).flatten()
        ypts = np.maximum(np.minimum(ypts, ysteps - 1), 0)
        if ti == 0:
            print("xpts: {}".format(xpts))
            print("ypts: {}".format(ypts))
        for pi in range(xpts.shape[0]):
            try:
                pcounts[ypts[pi], xpts[pi]] += 1
                tlifetime[ypts[pi], xpts[pi]] += fA[pi, ti]
            except (IndexError, ) as error_msg:
                # we don't abort here cause the brownian-motion wiggle of AvectionRK4EulerMarujama always edges on machine precision, which can np.floor(..) make go over-size
                print("\nError trying to index point ({}, {}) with indices ({}, {})".format(fX[pi, ti], fY[pi, ti], xpts[pi], ypts[pi]))
            if (pi % 100) == 0:
                current_item = (ti*fT.shape[0]) + pi
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
    print("\nField Interpolation done.")
    # rel_density = density / float(N)
    data_xarray.close()
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

    particle_file = h5py.File(os.path.join(odir, "particles.h5"), "w")
    px_ds = particle_file.create_dataset("p_x", data=fX, compression="gzip", compression_opts=4)
    py_ds = particle_file.create_dataset("p_y", data=fY, compression="gzip", compression_opts=4)
    if fZ is not None:
        pz_ds = particle_file.create_dataset("p_z", data=fZ, compression="gzip", compression_opts=4)
    pt_ds = particle_file.create_dataset("p_t", data=fT, compression="gzip", compression_opts=4)
    # write attribute: time_in_min = np.nanmin(ctime_array, axis=0)
    # write attribute: time_in_max = np.nanmax(ctime_array, axis=0)
    page_ds = particle_file.create_dataset("p_age", data=fA, compression="gzip", compression_opts=4)
    particle_file.close()

    del rel_density
    del lifetime
    del density
    del pcounts
    del fX
    del fY
    del fZ
    del fA

    # xval = np.arange(start=-a*0.5, stop=a*0.5, step=step, dtype=np.float32)[0:-1]
    # yval = np.arange(start=-b*0.5, stop=b*0.5, step=step, dtype=np.float32)[0:-1]
    # centers_x = xval + step/2.0
    # centers_y = yval + step/2.0
    # us = np.zeros((fT.shape[1], centers_y.shape[0], centers_x.shape[0]))
    # vs = np.zeros((fT.shape[1], centers_y.shape[0], centers_x.shape[0]))
    # mgrid = (ftimes, flats, flons)
    # p_center_t, p_center_y, p_center_x = np.meshgrid(fT[0], centers_y, centers_x, sparse=False, indexing='ij')
    # gcenters = (p_center_t.flatten(), p_center_y.flatten(), p_center_x.flatten())
    # # print("gcenters dims = ({}, {}, {})".format(gcenters[0].shape, gcenters[1].shape, gcenters[2].shape))
    # # print("mgrid dims = ({}, {}, {})".format(mgrid[0].shape, mgrid[1].shape, mgrid[2].shape))
    # # print("u dims = ({}, {}, {})".format(U.shape[0], U.shape[1], U.shape[2]))
    # # print("v dims = ({}, {}, {})".format(V.shape[0], V.shape[1], V.shape[2]))
    # us_local = interpn(mgrid, U, gcenters, method='linear', fill_value=.0)
    # vs_local = interpn(mgrid, V, gcenters, method='linear', fill_value=.0)
    # # print("us_local dims: {}".format(us_local.shape))
    # # print("vs_local dims: {}".format(vs_local.shape))
    # us_local = np.reshape(us_local, p_center_t.shape)
    # vs_local = np.reshape(vs_local, p_center_t.shape)
    # # print("us_local dims: {}".format(us_local.shape))
    # # print("vs_local dims: {}".format(vs_local.shape))
    # us[:, :, :] = us_local
    # vs[:, :, :] = vs_local
    #
    # us_file = h5py.File(os.path.join(odir, "hydrodynamic_U.h5"), "w")
    # us_file_ds = us_file.create_dataset("uo", data=us, compression="gzip", compression_opts=4)
    # us_file.close()
    #
    # vs_file = h5py.File(os.path.join(odir, "hydrodynamic_V.h5"), "w")
    # vs_file_ds = vs_file.create_dataset("vo", data=vs, compression="gzip", compression_opts=4)
    # vs_file.close()
    #
    # grid_file = h5py.File(os.path.join(odir, "grid.h5"), "w")
    # grid_lon_ds = grid_file.create_dataset("longitude", data=centers_x, compression="gzip", compression_opts=4)
    # grid_lat_ds = grid_file.create_dataset("latitude", data=centers_y, compression="gzip", compression_opts=4)
    # # grid_lon_ds = grid_file.create_dataset("longitude", data=xval_start, compression="gzip", compression_opts=4)
    # # grid_lat_ds = grid_file.create_dataset("latitude", data=yval_start, compression="gzip", compression_opts=4)
    # grid_time_ds = grid_file.create_dataset("times", data=fT[0], compression="gzip", compression_opts=4)
    # grid_file.close()


    xval = np.arange(start=-a*0.5, stop=a*0.5, step=step, dtype=np.float32)
    yval = np.arange(start=-b*0.5, stop=b*0.5, step=step, dtype=np.float32)
    centers_x = xval + step/2.0
    centers_y = yval + step/2.0
    us = np.zeros((centers_y.shape[0], centers_x.shape[0]))
    vs = np.zeros((centers_y.shape[0], centers_x.shape[0]))

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
    grid_time_ds = grid_file.create_dataset("times", data=global_fT, compression="gzip", compression_opts=4)
    grid_time_ds.attrs['unit'] = "seconds"
    grid_time_ds.attrs['name'] = 'time'
    grid_time_ds.attrs['min'] = global_fT.min()
    grid_time_ds.attrs['max'] = global_fT.max()
    grid_file.close()

    us_minmax = [0., 0.]
    us_statistics = [0., 0.]
    us_file = h5py.File(os.path.join(odir, "hydrodynamic_U.h5"), "w")
    us_file_ds = us_file.create_dataset("uo", shape=(1, us.shape[0], us.shape[1]), dtype=us.dtype, maxshape=(fT.shape[1], us.shape[0], us.shape[1]), compression="gzip", compression_opts=4)
    us_file_ds.attrs['unit'] = "m/s"
    us_file_ds.attrs['name'] = 'meridional_velocity'

    vs_minmax = [0., 0.]
    vs_statistics = [0., 0.]
    vs_file = h5py.File(os.path.join(odir, "hydrodynamic_V.h5"), "w")
    vs_file_ds = vs_file.create_dataset("vo", shape=(1, vs.shape[0], vs.shape[1]), dtype=vs.dtype, maxshape=(fT.shape[1], vs.shape[0], vs.shape[1]), compression="gzip", compression_opts=4)
    vs_file_ds.attrs['unit'] = "m/s"
    vs_file_ds.attrs['name'] = 'zonal_velocity'

    print("Sampling UV on CMEMS grid ...")
    # sample_lats = centers_y
    # sample_lons = centers_x
    sample_time = 0
    # sample_func = sample_uv
    fieldset = create_CMEMS_fieldset(datahead=datahead, periodic_wrap=True)
    p_center_y, p_center_x = np.meshgrid(centers_y, centers_x, sparse=False, indexing='ij')
    sample_pset = ParticleSet(fieldset=fieldset, pclass=SampleParticle, lon=np.array(p_center_x).flatten(), lat=np.array(p_center_y).flatten(), time=sample_time)
    sample_kernel = sample_pset.Kernel(sample_uv)
    sample_outname = out_fname + "_sampleuv"
    sample_output_file = sample_pset.ParticleFile(name=os.path.join(odir,sample_outname+".nc"), outputdt=delta(minutes=outdt_minutes))
    if backwardSimulation:
        sample_pset.execute(sample_kernel, runtime=delta(days=time_in_days), dt=delta(minutes=-dt_minutes), output_file=sample_output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(minutes=outdt_minutes))
    else:
        sample_pset.execute(sample_kernel, runtime=delta(days=time_in_days), dt=delta(minutes=dt_minutes), output_file=sample_output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(minutes=outdt_minutes))
    sample_output_file.close()
    del sample_output_file
    del sample_pset
    del sample_kernel
    print("UV on CMEMS grid sampled.")

    print("Load sampled data ...")
    sample_xarray = xr.open_dataset(os.path.join(odir, sample_outname + ".nc"))
    N_s = sample_xarray['lon'].shape[0]
    tN_s = sample_xarray['lon'].shape[1]
    print("N: {}, t_N: {}".format(N_s, tN_s))
    ctime_array_s = sample_xarray['time'].data
    time_in_min_s = np.nanmin(ctime_array_s, axis=0)
    time_in_max_s = np.nanmax(ctime_array_s, axis=0)
    assert ctime_array_s.shape[1] == time_in_min.shape[0]
    for ti in range(ctime_array_s.shape[1]):
        replace_indices = np.isnan(ctime_array_s[:, ti])
        ctime_array_s[replace_indices, ti] = time_in_max_s[ti]
    print("time info from file before baselining: shape = {} type = {} range = ({}, {})".format(ctime_array_s.shape, type(ctime_array_s[0 ,0]), np.min(ctime_array_s[0]), np.max(ctime_array_s[0])))
    # timebase_s = ctime_array_s[:, 0]
    timebase_s = time_in_max_s[0]
    # dtime_array_s = np.transpose(ctime_array_s.transpose() - timebase_s)
    dtime_array_s = ctime_array_s - timebase_s
    print("time info from file after baselining: shape = {} type = {} range = {}".format( dtime_array_s.shape, type(dtime_array_s[0 ,0]), (np.min(dtime_array_s), np.max(dtime_array_s)) ))
    # print(dtime_array.dtype)
    # print(ns_per_sec.dtype)
    # pX = data_xarray['lon'].data[indices, :]  # only plot the first 32 particles
    # pY = data_xarray['lat'].data[indices, :]  # only plot the first 32 particles

    psX = sample_xarray['lon']  # to be loaded from pfile
    psY = sample_xarray['lat']  # to be loaded from pfile
    psZ = None
    if 'depth' in sample_xarray.keys():
        psZ = sample_xarray['depth']  # to be loaded from pfile
    elif 'z' in sample_xarray.keys():
        psZ = sample_xarray['z']  # to be loaded from pfile
    psT = dtime_array_s
    global_psT = time_in_max_s -time_in_max_s[0]
    # this really need to convert to float
    if isinstance(psT[0, 0], datetime.datetime) or isinstance(psT[0, 0], datetime.timedelta) or isinstance(psT[0, 0], np.timedelta64) or isinstance(psT[0, 0], np.datetime64) or np.float64(psT[0, 1] - psT[0, 0]) > (dt_minutes + dt_minutes / 2.0):
    # if isinstance(psT[0,0], datetime.datetime) or isinstance(psT[0, 0], datetime.timedelta) or np.float64(psT[0, 1]-psT[0, 0]) > (dt_minutes+dt_minutes/2.0):
        print("psT.dtype before conversion: {}".format(psT.dtype))
        psT = (psT / ns_per_sec).astype(np.float64)  # to be loaded from pfile
        global_psT = (global_psT / ns_per_sec).astype(np.float64)
        print("psT.range and ft.dtype after conversion: ({}, {}) \t {}".format(psT[0].min(), psT[0].max(), psT.dtype))
    psU = sample_xarray['sample_u']  # to be loaded from pfile
    psV = sample_xarray['sample_v']  # to be loaded from pfile
    print("Sampled data loaded.")

    print("Interpolating UV on a regular-square grid ...")
    total_items = psT.shape[1]
    for ti in range(psT.shape[1]):
        us_local = np.expand_dims(psU[:, ti], axis=1)
        vs_local = np.expand_dims(psV[:, ti], axis=1)
        if ti == 0:
            print("us.shape {}; us_local.shape: {}; psU.shape: {}; p_center_y.shape: {}".format(us.shape, us_local.shape, psU.shape, p_center_y.shape))

        us[:, :] = np.reshape(us_local, p_center_y.shape)
        vs[:, :] = np.reshape(vs_local, p_center_y.shape)

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

    del centers_x
    del centers_y
    del xval
    del yval
    # del xval_start
    # del yval_start
    del fT
    del global_fT
    del dtime_array
    del ctime_array
