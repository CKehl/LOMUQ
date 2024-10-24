"""
Author: Dr. Christian Kehl
Date: 11-02-2020
"""

from parcels import AdvectionEE, AdvectionRK45, AdvectionRK4_3D
from parcels import FieldSet, ScipyParticle, JITParticle, Variable, AdvectionRK4, StateCode, OperationCode, ErrorCode
# from parcels.particleset_benchmark import ParticleSet_Benchmark as BenchmarkParticleSet
# from parcels.particleset import ParticleSet as DryParticleSet
from parcels import BenchmarkParticleSetSOA, BenchmarkParticleSetAOS, BenchmarkParticleSetNodes
from parcels import ParticleSetSOA, ParticleSetAOS, ParticleSetNodes
from parcels import GenerateID_Service, SequentialIdGenerator, LibraryRegisterC  # noqa
from parcels.field import VectorField, NestedField, SummedField
# from parcels import plotTrajectoriesFile_loadedField
# from parcels import rng as random
from parcels import ParcelsRandom
from parcels.tools import logger
from datetime import timedelta as delta
from datetime import datetime
from argparse import ArgumentParser
import numpy as np
import fnmatch
# import psutil
# import dask as da
# import dask.array as daArray
# import matplotlib.pyplot as plt
from glob import glob
import time as ostime
import math
import sys
import os
import gc
try:
    from mpi4py import MPI
except:
    MPI = None

with_GC = False

import warnings
import xarray as xr
warnings.simplefilter("ignore", category=xr.SerializationWarning)

pset = None
pset_modes = ['soa', 'aos', 'nodes']
ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
pset_types_dry = {'soa': {'pset': ParticleSetSOA},  # , 'pfile': ParticleFileSOA, 'kernel': KernelSOA
                  'aos': {'pset': ParticleSetAOS},  # , 'pfile': ParticleFileAOS, 'kernel': KernelAOS
                  'nodes': {'pset': ParticleSetNodes}}  # , 'pfile': ParticleFileNodes, 'kernel': KernelNodes
pset_types = {'soa': {'pset': BenchmarkParticleSetSOA},
              'aos': {'pset': BenchmarkParticleSetAOS},
              'nodes': {'pset': BenchmarkParticleSetNodes}}
method = {'RK4': AdvectionRK4, 'EE': AdvectionEE, 'RK45': AdvectionRK45}
global_t_0 = 0
odir = ""


def create_OFAM_fieldset(datahead, basefile_str, periodic_wrap, cache=True, chs=None):
    # ddir = os.path.join(datahead, "OFAM3_2017_daily/2D/")
    # files = sorted(glob(ddir+"ocean_eta_t_1994_*.nc"))
    if type(basefile_str) == dict:
        files = {'U': sorted(glob(os.path.join(datahead, basefile_str['U']))),
                 'V': sorted(glob(os.path.join(datahead, basefile_str['V'])))}
    else:
        files = sorted(glob(os.path.join(datahead, basefile_str)))
    # files = sorted(glob(os.path.join(datahead, "ocean_eta_t_1994_*.nc")))
    # logger.info("Parsed files: {}".format(files))
    variables = {'U': 'u', 'V': 'v'}
    dimensions = {'lon': 'xu_ocean', 'lat': 'yu_ocean', 'time': 'Time'}
    if periodic_wrap:
        return FieldSet.from_netcdf(files, variables, dimensions, chunksize=chs, time_periodic=delta(days=122))
    else:
        return FieldSet.from_netcdf(files, variables, dimensions, chunksize=chs, allow_time_extrapolation=True)


class AgeParticle_JIT(JITParticle):
    age = Variable('age', dtype=np.float64, initial=0.0)
    life_expectancy = Variable('life_expectancy', dtype=np.float64, initial=np.finfo(np.float64).max)
    initialized_dynamic = Variable('initialized_dynamic', dtype=np.int32, initial=0)

class AgeParticle_SciPy(ScipyParticle):
    age = Variable('age', dtype=np.float64, initial=0.0)
    life_expectancy = Variable('life_expectancy', dtype=np.float64, initial=np.finfo(np.float64).max)
    initialized_dynamic = Variable('initialized_dynamic', dtype=np.int32, initial=0)

age_ptype = {'scipy': AgeParticle_SciPy, 'jit': AgeParticle_JIT}

def periodicBC(particle, fieldSet, time):
    if particle.lon > 180.0:
        particle.lon -= 360.0
    if particle.lon < -180.0:
        particle.lon += 360.0
    particle.lat = min(particle.lat, 90.0)
    particle.lat = max(particle.lat, -80.0)
    # if particle.lat > 90.0:
    #     particle.lat -= 170.0
    # if particle.lat < -80.0:
    #     particle.lat += 170.0

def initialize(particle, fieldset, time):
    if particle.initialized_dynamic < 1:
        particle.life_expectancy = time + ParcelsRandom.uniform(.0, fieldset.life_expectancy) * math.sqrt(3.0 / 2.0)
        # particle.life_expectancy = time+ParcelsRandom.uniform(.0, fieldset.life_expectancy) * ((3.0/2.0)**2.0)
        particle.initialized_dynamic = 1

def Age(particle, fieldset, time):
    if particle.state == StateCode.Evaluate:
        particle.age = particle.age + math.fabs(particle.dt)
    if particle.age > particle.life_expectancy:
        particle.delete()

def DeleteParticle(particle, fieldset, time):
    particle.delete()

def RenewParticle(particle, fieldset, time):
    particle.lat = np.random.rand() * 360.0 -180.0
    particle.lon = np.random.rand() * 170.0 -80.0

def perIterGC():
    gc.collect()

if __name__=='__main__':
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-i", "--imageFileName", dest="imageFileName", type=str, default="mpiChunking_plot_MPI.png", help="image file name of the plot")
    parser.add_argument("-b", "--backwards", dest="backwards", action='store_true', default=False, help="enable/disable running the simulation backwards")
    parser.add_argument("-p", "--periodic", dest="periodic", action='store_true', default=False, help="enable/disable periodic wrapping (else: extrapolation)")
    parser.add_argument("-r", "--release", dest="release", action='store_true', default=False, help="continuously add particles via repeatdt (default: False)")
    parser.add_argument("-rt", "--releasetime", dest="repeatdt", type=int, default=720, help="repeating release rate of added particles in Minutes (default: 720min = 12h)")
    parser.add_argument("-a", "--aging", dest="aging", action='store_true', default=False, help="Removed aging particles dynamically (default: False)")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=1, help="runtime in days (default: 1)")
    parser.add_argument("-x", "--xarray", dest="use_xarray", action='store_true', default=False, help="use xarray as data backend")
    parser.add_argument("-w", "--writeout", dest="write_out", action='store_true', default=False, help="write data in outfile")
    parser.add_argument("-d", "--delParticle", dest="delete_particle", action='store_true', default=False, help="switch to delete a particle (True) or reset a particle (default: False).")
    parser.add_argument("-A", "--animate", dest="animate", action='store_true', default=False, help="animate the particle trajectories during the run or not (default: False).")
    parser.add_argument("-V", "--visualize", dest="visualize", action='store_true', default=False, help="Visualize particle trajectories at the end (default: False). Requires -w in addition to take effect.")
    parser.add_argument("-N", "--n_particles", dest="nparticles", type=str, default="2**6", help="number of particles to generate and advect (default: 2e6)")
    parser.add_argument("-sN", "--start_n_particles", dest="start_nparticles", type=str, default="96", help="(optional) number of particles generated per release cycle (if --rt is set) (default: 96)")
    parser.add_argument("-m", "--mode", dest="compute_mode", choices=['jit','scipy'], default="jit", help="computation mode = [JIT, SciPy]")
    parser.add_argument("-tp", "--type", dest="pset_type", default="soa", help="particle set type = [SOA, AOS, Nodes]")
    parser.add_argument("-G", "--GC", dest="useGC", action='store_true', default=False, help="using a garbage collector (default: false)")
    parser.add_argument("-chs", "--chunksize", dest="chs", type=int, default=0, help="defines the chunksize level: 0=None, 1='auto', 2=fine tuned; default: 0")
    args = parser.parse_args()

    pset_type = str(args.pset_type).lower()
    assert pset_type in pset_types
    ParticleSet = pset_types[pset_type]['pset']
    if args.dryrun:
        ParticleSet = pset_types_dry[pset_type]['pset']

    imageFileName=args.imageFileName
    periodicFlag=args.periodic
    backwardSimulation = args.backwards
    repeatdtFlag=args.release
    repeatRateMinutes=args.repeatdt
    time_in_days = args.time_in_days
    use_xarray = args.use_xarray
    agingParticles = args.aging
    with_GC = args.useGC

    # ======================================================= #
    # new ID generator things
    # ======================================================= #
    idgen = None
    c_lib_register = None
    if pset_type == 'nodes':
        idgen = GenerateID_Service(SequentialIdGenerator)
        idgen.setDepthLimits(0., 1.0)
        idgen.setTimeLine(0, delta(days=time_in_days).total_seconds())
        c_lib_register = LibraryRegisterC()

    Nparticle = int(float(eval(args.nparticles)))
    target_N = Nparticle
    start_N_particles = int(float(eval(args.start_nparticles)))
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        if mpi_comm.Get_rank() == 0:
            if agingParticles and not repeatdtFlag:
                sys.stdout.write("N: {} ( {} )\n".format(Nparticle, int(Nparticle * (3.0 / 2.0)**2.0)))
            else:
                sys.stdout.write("N: {}\n".format(Nparticle))
    else:
        if agingParticles and not repeatdtFlag:
            sys.stdout.write("N: {} ( {} )\n".format(Nparticle, int(Nparticle * (3.0 / 2.0)**2.0)))
        else:
            sys.stdout.write("N: {}\n".format(Nparticle))

    dt_minutes = 60
    nowtime = datetime.now()
    ParcelsRandom.seed(nowtime.microsecond)

    branch = "benchmarking"
    computer_env = "local/unspecified"
    scenario = "OFAM"
    headdir = ""
    odir = ""
    dirread_pal = ""
    datahead = ""
    dirread_top = ""
    dirread_top_bgc = ""
    basefile_str = {}
    if os.uname()[1] in ['science-bs35', 'science-bs36']:  # Gemini
        # headdir = "/scratch/{}/experiments/palaeo-parcels".format(os.environ['USER'])
        headdir = "/scratch/{}/experiments/parcels_benchmarking".format("ckehl")
        odir = headdir
        datahead = "/data/oceanparcels/input_data"
        dirread_top = os.path.join(datahead, 'OFAM3_2017_daily/3D/')
        basefile_str = {'U': "ocean_u_1993_*.nc", 'V': "ocean_v_1993_*.nc"}
        computer_env = "Gemini"
    elif os.uname()[1] in ["lorenz.science.uu.nl",] or fnmatch.fnmatchcase(os.uname()[1], "node*"):  # Lorenz
        CARTESIUS_SCRATCH_USERNAME = 'ckehl'
        headdir = "/storage/shared/oceanparcels/output_data/data_{}/experiments/parcels_benchmarking".format(CARTESIUS_SCRATCH_USERNAME)
        odir = headdir
        datahead = "/storage/shared/oceanparcels/input_data/"
        dirread_top = os.path.join(datahead, 'OFAM3_2017_daily/3D/')
        basefile_str = {'U': "ocean_u_1993_*.nc", 'V': "ocean_v_1993_*.nc"}
        computer_env = "Lorenz"
    elif fnmatch.fnmatchcase(os.uname()[1], "*.bullx*"):  # Cartesius
        CARTESIUS_SCRATCH_USERNAME = 'ckehluu'
        headdir = "/scratch/shared/{}/experiments/parcels_benchmarking".format(CARTESIUS_SCRATCH_USERNAME)
        odir = headdir
        datahead = "/projects/0/topios/hydrodynamic_data"
        dirread_top = os.path.join(datahead, 'OFAM3_2017_daily/3D/')
        basefile_str = {'U': "ocean_u_1993_*.nc", 'V': "ocean_v_1993_*.nc"}
        computer_env = "Cartesius"
    elif fnmatch.fnmatchcase(os.uname()[1], "int*.snellius.*") or fnmatch.fnmatchcase(os.uname()[1], "fcn*") or fnmatch.fnmatchcase(os.uname()[1], "tcn*") or fnmatch.fnmatchcase(os.uname()[1], "gcn*") or fnmatch.fnmatchcase(os.uname()[1], "hcn*"):  # Snellius
        SNELLIUS_SCRATCH_USERNAME = 'ckehluu'
        headdir = "/scratch-shared/{}/experiments/parcels_benchmarking".format(SNELLIUS_SCRATCH_USERNAME)
        odir = headdir
        datahead = "/projects/0/topios/hydrodynamic_data"
        dirread_top = os.path.join(datahead, 'OFAM3_2017_daily/3D/')
        basefile_str = {'U': "ocean_u_1993_*.nc", 'V': "ocean_v_1993_*.nc"}
        computer_env = "Snellius"
    elif fnmatch.fnmatchcase(os.uname()[1], "PROMETHEUS"):  # Prometheus computer - use USB drive
        SNELLIUS_SCRATCH_USERNAME = 'christian'
        headdir = "/media/{}/DATA/data/hydrodynamics/OFAM/sim/".format(SNELLIUS_SCRATCH_USERNAME)
        odir = headdir
        datahead = "/media/{}/DATA/data/hydrodynamics"
        dirread_top = os.path.join(datahead, "OFAM3", "daily", "3D")
        basefile_str = {'U': "ocean_u_1993_*.nc", 'V': "ocean_v_1993_*.nc"}
        computer_env = "Prometheus"
    else:
        headdir = "/var/scratch/experiments"
        odir = headdir
        dirread_pal = headdir
        datahead = "/data"
        dirread_top = os.path.join(datahead, 'OFAM3_2017_daily/3D/')
        basefile_str = {'U': "ocean_u_1993_*.nc", 'V': "ocean_v_1993_*.nc"}
    print("running {} on {} (uname: {}) - branch '{}' - (target) N: {} - argv: {}".format(scenario, computer_env, os.uname()[1], branch, target_N, sys.argv[1:]))

    if os.path.sep in imageFileName:
        head_dir = os.path.dirname(imageFileName)
        if head_dir[0] == os.path.sep:
            odir = head_dir
        else:
            odir = os.path.join(odir, head_dir)
            imageFileName = os.path.split(imageFileName)[1]

    func_time = []
    mem_used_GB = []

    np.random.seed(nowtime.microsecond)
    chunksize = None
    if args.chs == 0:
        chunksize = False
    elif args.chs == 1:
        chunksize = 'auto'
    elif args.chs > 1:
        chunksize = {'U': {'lon': ('longitude', 128), 'lat': ('latitude', 96), 'depth': ('depth', 25), 'time': ('time', 1)},
                     'V': {'lon': ('longitude', 128), 'lat': ('latitude', 96), 'depth': ('depth', 25), 'time': ('time', 1)}}
    fieldset = create_OFAM_fieldset(dirread_top, basefile_str, periodic_wrap=periodicFlag, chs=chunksize)

    if args.compute_mode == 'scipy':
        Nparticle = 2**10

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            #global_t_0 = MPI.Wtime()
            global_t_0 = ostime.process_time()
    else:
        #global_t_0 = ostime.time()
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

    addParticleN = 1
    # np_scaler = math.sqrt(3.0/2.0)
    np_scaler = (3.0 / 2.0)**2.0       # **
    # np_scaler = 3.0 / 2.0
    # cycle_scaler = math.sqrt(3.0/2.0)
    cycle_scaler = (3.0 / 2.0)**2.0    # **
    # cycle_scaler = 3.0 / 2.0
    if agingParticles:
        if not repeatdtFlag:
            Nparticle = int(Nparticle * np_scaler)
        fieldset.add_constant('life_expectancy', delta(days=time_in_days).total_seconds())
    if repeatdtFlag:
        addParticleN = Nparticle/2.0
        refresh_cycle = (delta(days=time_in_days).total_seconds() / (addParticleN/start_N_particles)) / cycle_scaler
        if agingParticles:
            refresh_cycle /= cycle_scaler
        repeatRateMinutes = int(refresh_cycle/60.0) if repeatRateMinutes == 720 else repeatRateMinutes

    if backwardSimulation:
        # ==== backward simulation ==== #
        if agingParticles:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * 360.0 -180.0, lat=np.random.rand(start_N_particles, 1) * 160.0 - 80.0, time=simStart, repeatdt=delta(minutes=repeatRateMinutes), idgen=idgen, c_lib_register=c_lib_register)
                if pset_type != 'nodes':
                    psetA = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(int(Nparticle/2.0), 1) * 360.0 -180.0, lat=np.random.rand(int(Nparticle/2.0), 1) * 160.0 - 80.0, time=simStart)
                    pset.add(psetA)
                else:
                    lonlat_field = np.random.rand(int(addParticleN), 2)
                    lonlat_field[0] = lonlat_field[0] * 360.0 -180.0
                    lonlat_field[1] = lonlat_field[1] * 160.0 - 80.0
                    time_field = np.ones((int(addParticleN), 1), dtype=np.float64) * simStart
                    # pdata = np.concatenate( (lonlat_field, time_field), axis=1 )
                    pdata = {'lon': lonlat_field[:, 0], 'lat': lonlat_field[:, 1], 'time': time_field}
                    pset.add(pdata)
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * 360.0 -180.0, lat=np.random.rand(Nparticle, 1) * 160.0 - 80.0, time=simStart, idgen=idgen, c_lib_register=c_lib_register)
        else:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * 360.0 -180.0, lat=np.random.rand(start_N_particles, 1) * 160.0 - 80.0, time=simStart, repeatdt=delta(minutes=repeatRateMinutes), idgen=idgen, c_lib_register=c_lib_register)
                if pset_type != 'nodes':
                    psetA = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(int(Nparticle/2.0), 1) * 360.0 -180.0, lat=np.random.rand(int(Nparticle/2.0), 1) * 160.0 - 80.0, time=simStart)
                    pset.add(psetA)
                else:
                    lonlat_field = np.random.rand(int(addParticleN), 2)
                    lonlat_field[0] = lonlat_field[0] * 360.0 -180.0
                    lonlat_field[1] = lonlat_field[1] * 160.0 - 80.0
                    time_field = np.ones((int(addParticleN), 1), dtype=np.float64) * simStart
                    # pdata = np.concatenate( (lonlat_field, time_field), axis=1 )
                    pdata = {'lon': lonlat_field[:, 0], 'lat': lonlat_field[:, 1], 'time': time_field}
                    pset.add(pdata)
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * 360.0 -180.0, lat=np.random.rand(Nparticle, 1) * 160.0 - 80.0, time=simStart, idgen=idgen, c_lib_register=c_lib_register)
    else:
        # ==== forward simulation ==== #
        if agingParticles:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * 360.0 -180.0, lat=np.random.rand(start_N_particles, 1) * 160.0 - 80.0, time=simStart, repeatdt=delta(minutes=repeatRateMinutes), idgen=idgen, c_lib_register=c_lib_register)
                if pset_type != 'nodes':
                    psetA = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(int(Nparticle/2.0), 1) * 360.0 -180.0, lat=np.random.rand(int(Nparticle/2.0), 1) * 160.0 - 80.0, time=simStart)
                    pset.add(psetA)
                else:
                    lonlat_field = np.random.rand(int(addParticleN), 2)
                    lonlat_field[0] = lonlat_field[0] * 360.0 -180.0
                    lonlat_field[1] = lonlat_field[1] * 160.0 - 80.0
                    time_field = np.ones((int(addParticleN), 1), dtype=np.float64) * simStart
                    # pdata = np.concatenate( (lonlat_field, time_field), axis=1 )
                    pdata = {'lon': lonlat_field[:, 0], 'lat': lonlat_field[:, 1], 'time': time_field}
                    pset.add(pdata)
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * 360.0 -180.0, lat=np.random.rand(Nparticle, 1) * 160.0 - 80.0, time=simStart, idgen=idgen, c_lib_register=c_lib_register)
        else:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * 360.0 -180.0, lat=np.random.rand(start_N_particles, 1) * 160.0 - 80.0, time=simStart, repeatdt=delta(minutes=repeatRateMinutes), idgen=idgen, c_lib_register=c_lib_register)
                if pset_type != 'nodes':
                    psetA = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(int(Nparticle/2.0), 1) * 360.0 -180.0, lat=np.random.rand(int(Nparticle/2.0), 1) * 160.0 - 80.0, time=simStart)
                    pset.add(psetA)
                else:
                    lonlat_field = np.random.rand(int(addParticleN), 2)
                    lonlat_field[0] = lonlat_field[0] * 360.0 -180.0
                    lonlat_field[1] = lonlat_field[1] * 160.0 - 80.0
                    time_field = np.ones((int(addParticleN), 1), dtype=np.float64) * simStart
                    # pdata = np.concatenate( (lonlat_field, time_field), axis=1 )
                    pdata = {'lon': lonlat_field[:, 0], 'lat': lonlat_field[:, 1], 'time': time_field}
                    pset.add(pdata)
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * 360.0 -180.0, lat=np.random.rand(Nparticle, 1) * 160.0 - 80.0, time=simStart, idgen=idgen, c_lib_register=c_lib_register)


    output_file = None
    out_fname = "benchmark_CMEMS"
    if args.write_out:
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
        output_file = pset.ParticleFile(name=os.path.join(odir,out_fname+".nc"), outputdt=delta(hours=24))
    delete_func = RenewParticle
    if args.delete_particle:
        delete_func=DeleteParticle
    postProcessFuncs = []

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            #starttime = MPI.Wtime()
            starttime = ostime.process_time()
    else:
        #starttime = ostime.time()
        starttime = ostime.process_time()
    kernels = pset.Kernel(AdvectionRK4,delete_cfiles=True)
    kernels += pset.Kernel(periodicBC, delete_cfiles=True)
    if agingParticles:
        kernels += pset.Kernel(initialize, delete_cfiles=True)
        kernels += pset.Kernel(Age, delete_cfiles=True)
    if with_GC:
        postProcessFuncs.append(perIterGC)
    if backwardSimulation:
        # ==== backward simulation ==== #
        if args.animate:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=-dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12), moviedt=delta(hours=6), movie_background_field=fieldset.U)
        else:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=-dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12))
    else:
        # ==== forward simulation ==== #
        if args.animate:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12), moviedt=delta(hours=6), movie_background_field=fieldset.U)
        else:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12))
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            #endtime = MPI.Wtime()
            endtime = ostime.process_time()
    else:
        #endtime = ostime.time()
        endtime = ostime.process_time()

    if args.write_out:
        output_file.close()

    if not args.dryrun:
        size_Npart = len(pset.nparticle_log)
        Npart = pset.nparticle_log.get_param(size_Npart-1)
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            Npart = mpi_comm.reduce(Npart, op=MPI.SUM, root=0)
            if mpi_comm.Get_rank() == 0:
                if size_Npart>0:
                    sys.stdout.write("final # particles: {}\n".format( Npart ))
                sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime-starttime))
                avg_time = np.mean(np.array(pset.total_log.get_values(), dtype=np.float64))
                sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time*1000.0))
        else:
            if size_Npart > 0:
                sys.stdout.write("final # particles: {}\n".format( Npart ))
            sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime - starttime))
            avg_time = np.mean(np.array(pset.total_log.get_values(), dtype=np.float64))
            sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time * 1000.0))

        # if args.write_out:
        #     output_file.close()
        #     if args.visualize:
        #         if MPI:
        #             mpi_comm = MPI.COMM_WORLD
        #             if mpi_comm.Get_rank() == 0:
        #                 plotTrajectoriesFile_loadedField(os.path.join(odir, out_fname+".nc"), tracerfield=fieldset.U)
        #         else:
        #             plotTrajectoriesFile_loadedField(os.path.join(odir, out_fname+".nc"),tracerfield=fieldset.U)

        if MPI:
            mpi_comm = MPI.COMM_WORLD
            # mpi_comm.Barrier()
            Nparticles = mpi_comm.reduce(np.array(pset.nparticle_log.get_params()), op=MPI.SUM, root=0)
            Nmem = mpi_comm.reduce(np.array(pset.mem_log.get_params()), op=MPI.SUM, root=0)
            if mpi_comm.Get_rank() == 0:
                pset.plot_and_log(memory_used=Nmem, nparticles=Nparticles, target_N=target_N, imageFilePath=imageFileName, odir=odir, xlim_range=[0, 730], ylim_range=[0, 150])
        else:
            pset.plot_and_log(target_N=target_N, imageFilePath=imageFileName, odir=odir, xlim_range=[0, 730], ylim_range=[0, 150])

    del pset
    if idgen is not None:
        idgen.close()
        del idgen
    if c_lib_register is not None:
        c_lib_register.clear()
        del c_lib_register
