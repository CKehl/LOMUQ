"""
Author: Dr. Christian Kehl
Date: 14-03-2024
"""

from parcels import AdvectionEE, AdvectionRK45, AdvectionRK4, AdvectionDiffusionEM, AdvectionDiffusionM1
from parcels import FieldSet, ScipyParticle, JITParticle, Variable, AdvectionRK4, StateCode, OperationCode, ErrorCode
from parcels.particleset import ParticleSet as BenchmarkParticleSet
from parcels.tools.converters import Geographic, GeographicPolar
# from parcels.field import Field, VectorField, NestedField, SummedField
# from parcels.grid import RectilinearZGrid
from parcels import ParcelsRandom
from numpy.random import default_rng
from datetime import timedelta, datetime
from argparse import ArgumentParser
import math
# import datetime
import numpy as np
import xarray as xr
import fnmatch
import sys
import gc
import os
import time as ostime
# from scipy.interpolate import interpn
from glob import glob
import h5py

try:
    from mpi4py import MPI
except:
    MPI = None
with_GC = False
DBG_MSG = False

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
method = {'RK4': AdvectionRK4, 'EE': AdvectionEE, 'RK45': AdvectionRK45}
global_t_0 = 0

day_start = datetime(2020,1,1,12,00)
startdate = '2020-01-01'
K = 13.39


a = 3.6 * 1e2  # by definition: meters
b = 1.8 * 1e2  # by definiton: meters
tsteps = 122 # in steps
tstepsize = 3.0 # unitary
tscale = 24.0*60.0*60.0 # in seconds

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

def create_fieldset(datahead, periodic_wrap):
    ddir = os.path.join(datahead, "CMEMS/GLOBAL_REANALYSIS_PHY_001_030/")
    files = sorted(glob(ddir+"mercatorglorys12v1_gl12_mean_201607*.nc"))
    variables = {'U': 'uo', 'V': 'vo'}
    dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    global ttotal
    ttotal = 31  # days
    # chs = False
    chs = 'auto'
    fieldset = None
    if periodic_wrap:
        fieldset = FieldSet.from_netcdf(files, variables, dimensions, time_periodic=timedelta(days=31), mesh='spherical')
    else:
        fieldset = FieldSet.from_netcdf(files, variables, dimensions, allow_time_extrapolation=True, mesh='spherical')
    global tsteps
    tsteps = len(fieldset.U.grid.time_full)
    global tstepsize
    tstepsize = int(math.floor(ttotal/tsteps))
    return fieldset

def create_SMOC_fieldset(datahead, periodic_wrap):
    ddir = os.path.join(datahead, "SMOC/currents/")
    files = sorted(glob(ddir+"SMOC_202201*.nc"))
    print(files)
    variables = {'U': 'uo', 'V': 'vo'}
    dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    global ttotal
    ttotal = 31  # days
    # chs = False
    chs = 'auto'
    fieldset = None
    if periodic_wrap:
        fieldset = FieldSet.from_netcdf(files, variables, dimensions, chunksize=False, time_periodic=timedelta(days=31), mesh='spherical')
    else:
        fieldset = FieldSet.from_netcdf(files, variables, dimensions, chunksize=False, allow_time_extrapolation=True, mesh='spherical')
    global tsteps
    tsteps = len(fieldset.U.grid.time_full)
    global tstepsize
    tstepsize = int(math.floor(ttotal/tsteps))
    return fieldset


def create_fieldset_tidal_currents(datahead, name, filename):
    '''
    Create fieldset for a given type of tide (name = M2, S2, K1, or O1)
    '''
    files_eastward = os.path.join(datahead, "eastward_velocity")
    files_northward = os.path.join(datahead, "northward_velocity")
    filename_U = os.path.join(files_eastward, '%s.nc' % filename)
    filename_V = os.path.join(files_northward, '%s.nc' % filename)

    filenames = {'Ua%s' % name: filename_U,
                 'Ug%s' % name: filename_U,
                 'Va%s' % name: filename_V,
                 'Vg%s' % name: filename_V}
    variables = {'Ua%s' % name: 'Ua',
                 'Ug%s' % name: 'Ug',
                 'Va%s' % name: 'Va',
                 'Vg%s' % name: 'Vg'}
    dimensions = {'lat': 'lat',
                  'lon': 'lon'}

    fieldset_tmp = FieldSet.from_netcdf(filenames, variables, dimensions, mesh='spherical')
    return fieldset_tmp


class SampleParticle(JITParticle):
    valid = Variable('valid', dtype=np.int32, initial=-1, to_write=True)
    sample_u = Variable('sample_u', initial=0., dtype=np.float32, to_write=True)
    sample_v = Variable('sample_v', initial=0., dtype=np.float32, to_write=True)
    U_tide = Variable('U_tide', dtype=np.float32, initial=0., to_write=True)
    V_tide = Variable('V_tide', dtype=np.float32, initial=0., to_write=True)
    pre_lon = Variable('pre_lon', dtype=np.float32, initial=0., to_write=False)
    pre_lat = Variable('pre_lat', dtype=np.float32, initial=0., to_write=False)


def DeleteParticle(particle, fieldset, time):
    if particle.valid < 0:
        particle.valid = 0
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
        particle.lat = ye / 2.0 - (math.fabs(particle.lat) - ye/2.0)
    if particle.lat < (-ye/2.0):
    #     particle.lat += b
        particle.lat = -ye / 2.0 + (math.fabs(particle.lat) - ye/2.0)


periodicBC = RenewParticle


def perIterGC():
    gc.collect()


def sample_uv(particle, fieldset, time):
    if particle.valid < 0:
        if (math.isnan(particle.time) == True):
            particle.valid = 0
        else:
            particle.valid = 1
    if particle.valid > 0:
        (hu, hv) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        particle.sample_u = hu
        particle.sample_v = hv
        # particle.sample_u = fieldset.U[time, particle.depth, particle.lat, particle.lon]
        # particle.sample_v = fieldset.V[time, particle.depth, particle.lat, particle.lon]


def TidalMotionM2S2K1O1(particle, fieldset, time):
    """
    Kernel that calculates tidal currents U and V due to M2, S2, K1 and O1 tide at particle location and time
    and advects the particle in these currents (using Euler forward scheme)
    Calculations based on Doodson (1921) and Schureman (1958)
    Also: Sterl (2019)
    """
    # Number of Julian centuries that have passed between t0 and time
    t = ((time + fieldset.t0rel) / 86400.0) / 36525.0

    # Define constants to compute astronomical variables T, h, s, N (all in degrees) (source: FES2014 code)
    cT0 = 180.0
    ch0 = 280.1895
    cs0 = 277.0248
    cN0 = 259.1568
    cN1 = -1934.1420
    deg2rad = math.pi / 180.0

    # Calculation of factors T, h, s at t0 (source: Doodson (1921))
    # T0 = math.fmod(cT0, 360.0) * deg2rad
    # h0 = math.fmod(ch0, 360.0) * deg2rad
    # s0 = math.fmod(cs0, 360.0) * deg2rad
    T0 = cT0 * deg2rad
    h0 = ch0 * deg2rad
    s0 = cs0 * deg2rad

    # Calculation of V(t0) (source: Schureman (1958))
    V_M2 = 2 * T0 + 2 * h0 - 2 * s0
    V_S2 = 2 * T0
    V_K1 = T0 + h0 - 0.5 * math.pi
    V_O1 = T0 + h0 - 2 * s0 + 0.5 * math.pi
    # nonlinear
    V_M4 = 4 * T0 - 4 * s0 + 4 * h0
    V_MS4 = 4 * T0 - 2 * s0 + 2 * h0
    V_S4 = 4 * T0

    # Calculation of factors N, I, nu, xi at time (source: Schureman (1958))
    # Since these factors change only very slowly over time, we take them as constant over the time step dt
    N = math.fmod(cN0 + cN1 * t, 360.0) * deg2rad
    I = math.acos(0.91370 - 0.03569 * math.cos(N))
    tanN = math.tan(0.5 * N)
    at1 = math.atan(1.01883 * tanN)
    at2 = math.atan(0.64412 * tanN)
    nu = at1 - at2
    xi = -at1 - at2 + N
    nuprim = math.atan(math.sin(2 * I) * math.sin(nu) / (math.sin(2 * I) * math.cos(nu) + 0.3347))

    # Calculation of u, f at current time (source: Schureman (1958))
    u_M2 = 2 * xi - 2 * nu
    f_M2 = (math.cos(0.5 * I)) ** 4 / 0.9154
    u_S2 = 0
    f_S2 = 1
    u_K1 = -nuprim
    f_K1 = math.sqrt(0.8965 * (math.sin(2 * I)) ** 2 + 0.6001 * math.sin(2 * I) * math.cos(nu) + 0.1006)
    u_O1 = 2 * xi - nu
    f_O1 = math.sin(I) * (math.cos(0.5 * I)) ** 2 / 0.3800
    # nonlinear
    u_M4 = 4 * xi - 4 * nu
    f_M4 = (f_M2) ** 2
    u_MS4 = 2 * xi - 2 * nu
    f_MS4 = f_M2
    u_S4 = 0
    f_S4 = 1

    # Euler forward method to advect particle in tidal currents

    lon0, lat0 = (particle.lon, particle.lat)

    # Zonal amplitudes and phaseshifts at particle location and time
    Uampl_M2_1 = f_M2 * fieldset.UaM2[time, particle.depth, lat0, lon0]
    Upha_M2_1 = V_M2 + u_M2 - fieldset.UgM2[time, particle.depth, lat0, lon0]
    Uampl_S2_1 = f_S2 * fieldset.UaS2[time, particle.depth, lat0, lon0]
    Upha_S2_1 = V_S2 + u_S2 - fieldset.UgS2[time, particle.depth, lat0, lon0]
    Uampl_K1_1 = f_K1 * fieldset.UaK1[time, particle.depth, lat0, lon0]
    Upha_K1_1 = V_K1 + u_K1 - fieldset.UgK1[time, particle.depth, lat0, lon0]
    Uampl_O1_1 = f_O1 * fieldset.UaO1[time, particle.depth, lat0, lon0]
    Upha_O1_1 = V_O1 + u_O1 - fieldset.UgO1[time, particle.depth, lat0, lon0]

    # nonlinear
    Uampl_M4_1 = f_M4 * fieldset.UaM4[time, particle.depth, lat0, lon0]
    Upha_M4_1 = V_M4 + u_M4 - fieldset.UgM4[time, particle.depth, lat0, lon0]
    Uampl_MS4_1 = f_MS4 * fieldset.UaMS4[time, particle.depth, lat0, lon0]
    Upha_MS4_1 = V_MS4 + u_MS4 - fieldset.UgMS4[time, particle.depth, lat0, lon0]
    Uampl_S4_1 = f_S4 * fieldset.UaS4[time, particle.depth, lat0, lon0]
    Upha_S4_1 = V_S4 + u_S4 - fieldset.UgS4[time, particle.depth, lat0, lon0]

    # Meridional amplitudes and phaseshifts at particle location and time
    Vampl_M2_1 = f_M2 * fieldset.VaM2[time, particle.depth, lat0, lon0]
    Vpha_M2_1 = V_M2 + u_M2 - fieldset.VgM2[time, particle.depth, lat0, lon0]
    Vampl_S2_1 = f_S2 * fieldset.VaS2[time, particle.depth, lat0, lon0]
    Vpha_S2_1 = V_S2 + u_S2 - fieldset.VgS2[time, particle.depth, lat0, lon0]
    Vampl_K1_1 = f_K1 * fieldset.VaK1[time, particle.depth, lat0, lon0]
    Vpha_K1_1 = V_K1 + u_K1 - fieldset.VgK1[time, particle.depth, lat0, lon0]
    Vampl_O1_1 = f_O1 * fieldset.VaO1[time, particle.depth, lat0, lon0]
    Vpha_O1_1 = V_O1 + u_O1 - fieldset.VgO1[time, particle.depth, lat0, lon0]

    # nonlinear
    Vampl_M4_1 = f_M4 * fieldset.VaM4[time, particle.depth, lat0, lon0]
    Vpha_M4_1 = V_M4 + u_M4 - fieldset.VgM4[time, particle.depth, lat0, lon0]
    Vampl_MS4_1 = f_MS4 * fieldset.VaMS4[time, particle.depth, lat0, lon0]
    Vpha_MS4_1 = V_MS4 + u_MS4 - fieldset.VgMS4[time, particle.depth, lat0, lon0]
    Vampl_S4_1 = f_S4 * fieldset.VaS4[time, particle.depth, lat0, lon0]
    Vpha_S4_1 = V_S4 + u_S4 - fieldset.VgS4[time, particle.depth, lat0, lon0]

    # Zonal and meridional tidal currents; time + fieldset.t0rel = number of seconds elapsed between t0 and time
    Uvel_M2_1 = Uampl_M2_1 * math.cos(fieldset.omegaM2 * (time + fieldset.t0rel) + Upha_M2_1)
    Uvel_S2_1 = Uampl_S2_1 * math.cos(fieldset.omegaS2 * (time + fieldset.t0rel) + Upha_S2_1)
    Uvel_K1_1 = Uampl_K1_1 * math.cos(fieldset.omegaK1 * (time + fieldset.t0rel) + Upha_K1_1)
    Uvel_O1_1 = Uampl_O1_1 * math.cos(fieldset.omegaO1 * (time + fieldset.t0rel) + Upha_O1_1)

    # nonlinear
    Uvel_M4_1 = Uampl_M4_1 * math.cos(fieldset.omegaM4 * (time + fieldset.t0rel) + Upha_M4_1)
    Uvel_MS4_1 = Uampl_MS4_1 * math.cos(fieldset.omegaMS4 * (time + fieldset.t0rel) + Upha_MS4_1)
    Uvel_S4_1 = Uampl_S4_1 * math.cos(fieldset.omegaS4 * (time + fieldset.t0rel) + Upha_S4_1)

    Vvel_M2_1 = Vampl_M2_1 * math.cos(fieldset.omegaM2 * (time + fieldset.t0rel) + Vpha_M2_1)
    Vvel_S2_1 = Vampl_S2_1 * math.cos(fieldset.omegaS2 * (time + fieldset.t0rel) + Vpha_S2_1)
    Vvel_K1_1 = Vampl_K1_1 * math.cos(fieldset.omegaK1 * (time + fieldset.t0rel) + Vpha_K1_1)
    Vvel_O1_1 = Vampl_O1_1 * math.cos(fieldset.omegaO1 * (time + fieldset.t0rel) + Vpha_O1_1)

    # nonlinear
    Vvel_M4_1 = Vampl_M4_1 * math.cos(fieldset.omegaM4 * (time + fieldset.t0rel) + Vpha_M4_1)
    Vvel_MS4_1 = Vampl_MS4_1 * math.cos(fieldset.omegaMS4 * (time + fieldset.t0rel) + Vpha_MS4_1)
    Vvel_S4_1 = Vampl_S4_1 * math.cos(fieldset.omegaS4 * (time + fieldset.t0rel) + Vpha_S4_1)

    # Total zonal and meridional velocity, including nonlinear
    U1 = Uvel_M2_1 + Uvel_S2_1 + Uvel_K1_1 + Uvel_O1_1 + Uvel_M4_1 + Uvel_MS4_1 + Uvel_S4_1  # total zonal velocity
    V1 = Vvel_M2_1 + Vvel_S2_1 + Vvel_K1_1 + Vvel_O1_1 + Vvel_M4_1 + Vvel_MS4_1 + Vvel_S4_1  # total meridional velocity

    particle.U_tide = U1
    particle.V_tide = V1

def validate(particle, fieldset, time):
    if particle.valid < 0:
        particle.pre_lon = particle.lon
        particle.pre_lat = particle.lat
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
    if not isinstance(low, np.ndarray):
        low = np.array(low)
    if not isinstance(high, np.ndarray):
        high = np.array(high)
    msize = size
    if not isinstance(msize, tuple):
        msize = (size, 2)
    sample = None
    rng_sampler = default_rng()
    if sample_string == 'uniform':
        sample = rng_sampler.uniform(low, high, msize).transpose()
        # sample = np.random.uniform(low, high, msize).transpose()
    elif sample_string == 'gaussian':
        sample = rng_sampler.normal(0.0, 0.5, msize).transpose() + 0.5
        # sample = np.random.normal(0.0, 0.5, msize) + 0.5
        sample[sample < 0] = 0.0
        sample[sample > 1] = 1.0
        scalev = (high - low)
        scalerm = np.eye(2, dtype=sample.dtype)
        scalerm[0, 0] *= scalev[0]
        scalerm[1, 1] *= scalev[1]
        # sample = scalerm * sample  # + low
        sample = np.dot(scalerm, sample)
        sample[0, :] += low[0]
        sample[1, :] += low[1]
    elif sample_string == 'triangular':
        if np.all(low == high):
            print("low: {}, high: {}".format(low, high))
        mid = low + (high-low)/2.0
        sample = rng_sampler.triangular(low, mid, high, msize).transpose()
        # sample = np.random.triangular(low, mid, high, msize).transpose()
    elif sample_string == 'vonmises':
        sample = rng_sampler.vonmises(0, 1/math.sqrt(2.0), msize).transpose() + 0.5
        # sample = np.random.vonmises(0, 1 / math.sqrt(2.0), msize).transpose() + 0.5
        sample[sample < 0] = 0.0
        sample[sample > 1] = 1.0
        scalev = (high - low)
        scalerm = np.eye(2, dtype=sample.dtype)
        scalerm[0, 0] *= scalev[0]
        scalerm[1, 1] *= scalev[1]
        # sample = scalerm * sample  # + low
        sample = np.dot(scalerm, sample)
        sample[0, :] += low[0]
        sample[1, :] += low[1]
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
                # samples_lon.append(np.array(local_samples[0, :]).flatten())
                # samples_lat.append(np.array(local_samples[1, :]).flatten())
                samples_lon.append(local_samples[0, :])
                samples_lat.append(local_samples[1, :])
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


if __name__ == "__main__":
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-f", "--filename", dest="filename", type=str, default="file.txt", help="(relative) (text) file path of the csv hyper parameters")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=366, help="runtime in days (default: 1)")
    parser.add_argument("-dt", "--deltatime", dest="dt", type=int, default=300, help="computational delta_t time stepping in minutes (default: 300sec = 5min = 0.05h)")
    parser.add_argument("-ot", "--outputtime", dest="outdt", type=int, default=1800, help="output time stepping in seconds (default: 1800sec = 30min = 0.5h)")
    parser.add_argument("-im", "--interp_mode", dest="interp_mode", choices=['rk4','rk45', 'ee', 'em', 'm1'], default="jit", help="interpolation mode = [rk4, rk45, ee (Eulerian Estimation), em (Euler-Maruyama), m1 (Milstein-1)]")
    # parser.add_argument("-N", "--n_particles", dest="nparticles", type=str, default="2**6", help="number of particles to generate and advect (default: 2e6)")
    parser.add_argument("-gres", "--grid_resolution", dest="gres", type=str, default="10", help="number of cells per arc-degree or metre (default: 2). Fair range: 1 - 12")
    # parser.add_argument("-fsx", "--field_sx", dest="field_sx", type=int, default="480", help="number of original field cells in x-direction")
    # parser.add_argument("-fsy", "--field_sy", dest="field_sy", type=int, default="240", help="number of original field cells in y-direction")
    args = parser.parse_args()

    ParticleSet = BenchmarkParticleSet
    # if args.dryrun:
    #     ParticleSet = DryParticleSet

    filename=args.filename
    # field_sx = args.field_sx
    # field_sy = args.field_sy
    deleteBC = True
    animate_result = False
    visualize_results = False
    periodicFlag=True
    backwardSimulation = False
    repeatdtFlag=False
    time_in_days = args.time_in_days
    agingParticles = False
    writeout = True
    with_GC = True
    # Nparticle = int(float(eval(args.nparticles)))
    # target_N = Nparticle
    # addParticleN = 1
    # np_scaler = 3.0 / 2.0
    # cycle_scaler = 7.0 / 4.0
    # start_N_particles = Nparticle
    gres = int(float(eval(args.gres)))
    interp_mode = args.interp_mode
    compute_mode = 'jit'  # args.compute_mode

    dt_seconds = args.dt
    outdt_seconds = args.outdt
    nowtime = datetime.now()
    ParcelsRandom.seed(nowtime.microsecond)

    branch = "Tidal"
    computer_env = "local/unspecified"
    scenario = "resample"
    headdir = ""
    odir = ""
    dirread_pal = ""
    datahead = ""
    dirread_top = ""
    tidaldir = ""
    if os.uname()[1] in ['science-bs35', 'science-bs36', 'science-bs37']:  # Gemini
        headdir = "/scratch/{}/experiments".format("ckehl")
        odir = headdir
        datahead = "/data/oceanparcels/input_data"
        tidaldir = "/data/oceanparcels/input_data/FES2014"
        dirread_top = os.path.join(datahead, 'CMEMS/GLOBAL_REANALYSIS_PHY_001_030/')
        computer_env = "Gemini"
    elif fnmatch.fnmatchcase(os.uname()[1], "*.bullx*"):  # Cartesius
        CARTESIUS_SCRATCH_USERNAME = 'ckehluu'
        headdir = "/scratch/shared/{}/experiments".format(CARTESIUS_SCRATCH_USERNAME)
        odir = headdir
        datahead = "/projects/0/topios/hydrodynamic_data"
        tidaldir = "/projects/0/topios/FES2014"
        dirread_top = os.path.join(datahead, 'CMEMS/GLOBAL_REANALYSIS_PHY_001_030/')
        computer_env = "Cartesius"
    elif fnmatch.fnmatchcase(os.uname()[1], "PROMETHEUS"):  # Prometheus computer - use USB drive
        CARTESIUS_SCRATCH_USERNAME = 'christian'
        headdir = "/media/{}/DATA/data/hydrodynamics".format(CARTESIUS_SCRATCH_USERNAME)
        odir = headdir
        # datahead = "/media/{}/MyPassport/data/hydrodynamic".format(CARTESIUS_SCRATCH_USERNAME)  # CMEMS
        datahead = "/media/{}/DATA/data/hydrodynamics".format(CARTESIUS_SCRATCH_USERNAME)  # CMEMS
        tidaldir = "/media/{}/DATA/data/Tidal/FES2014".format(CARTESIUS_SCRATCH_USERNAME)
        # dirread_top = os.path.join(datahead, 'CMEMS/GLOBAL_REANALYSIS_PHY_001_030/')
        dirread_top = os.path.join(datahead, 'SMOC/currents')
        computer_env = "Prometheus"
    else:
        headdir = "/var/scratch/experiments"
        odir = headdir
        dirread_pal = headdir
        datahead = "/data"
        dirread_top = os.path.join(datahead, 'CMEMS/GLOBAL_REANALYSIS_PHY_001_030/')
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


    # ================================================================================================================ #
    #          P O S T - P R O C E S S I N G
    # ================================================================================================================ #
    step = 1.0/gres
    xsteps = int(np.floor(a * gres))
    # xsteps = int(np.ceil(a * gres))
    ysteps = int(np.floor(b * gres))
    # ysteps = int(np.ceil(b * gres))

    xval = np.arange(start=-a*0.5, stop=a*0.5, step=step, dtype=np.float32)
    yval = np.arange(start=-b*0.5, stop=b*0.5, step=step, dtype=np.float32)
    centers_x = xval + step/2.0
    centers_y = yval + step/2.0
    us = np.zeros((centers_y.shape[0], centers_x.shape[0]))
    vs = np.zeros((centers_y.shape[0], centers_x.shape[0]))
    seconds_per_day = 24.0*60.0*60.0
    num_t_samples = int(np.floor((time_in_days*seconds_per_day) / outdt_seconds))
    global_fT = np.linspace(start=.0, stop=time_in_days*seconds_per_day, num=num_t_samples, endpoint=True, dtype=np.float64)

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
    grid_time_ds.attrs['min'] = np.nanmin(global_fT)
    grid_time_ds.attrs['max'] = np.nanmax(global_fT)
    grid_file.close()

    # =================================================== #
    # ==== also create a NetCDF structured mesh sink ==== #
    # =================================================== #

    us_minmax = [0., 0.]
    us_statistics = [0., 0.]
    us_file = h5py.File(os.path.join(odir, "hydrodynamic_U.h5"), "w")
    us_file_ds = us_file.create_dataset("uo", shape=(1, us.shape[0], us.shape[1]), dtype=us.dtype, maxshape=(global_fT.shape[0], us.shape[0], us.shape[1]), compression="gzip", compression_opts=4)
    us_file_ds.attrs['unit'] = "m/s"
    us_file_ds.attrs['name'] = 'meridional_velocity'

    vs_minmax = [0., 0.]
    vs_statistics = [0., 0.]
    vs_file = h5py.File(os.path.join(odir, "hydrodynamic_V.h5"), "w")
    vs_file_ds = vs_file.create_dataset("vo", shape=(1, vs.shape[0], vs.shape[1]), dtype=vs.dtype, maxshape=(global_fT.shape[0], vs.shape[0], vs.shape[1]), compression="gzip", compression_opts=4)
    vs_file_ds.attrs['unit'] = "m/s"
    vs_file_ds.attrs['name'] = 'zonal_velocity'

    print("Sampling UV on CMEMS grid ...")
    # sample_lats = centers_y
    # sample_lons = centers_x
    sample_time = 0


    # sample_func = sample_uv
    fieldset = create_SMOC_fieldset(datahead=datahead, periodic_wrap=True)
    t0 = datetime(1900, 1, 1, 0, 0)  # origin of time = 1 January 1900, 00:00:00 UTC
    fieldset.add_constant('t0rel', (day_start - t0).total_seconds())  # number of seconds elapsed between t0 and starttime
    deg2rad = math.pi / 180.0  # factor to convert degrees to radians
    fieldset_m2 = create_fieldset_tidal_currents(datahead=tidaldir, name='M2', filename='m2')
    fieldset_m2.UaM2.set_scaling_factor(1e-2)
    fieldset_m2.UgM2.set_scaling_factor(deg2rad)  # convert from degrees to radians
    fieldset_m2.VaM2.set_scaling_factor(1e-2)  # cm/s to m/s
    fieldset_m2.VgM2.set_scaling_factor(deg2rad)
    fieldset_m2.UaM2.units = GeographicPolar()
    fieldset_m2.VaM2.units = Geographic()
    fieldset.add_field(fieldset_m2.UaM2)
    fieldset.add_field(fieldset_m2.UgM2)
    fieldset.add_field(fieldset_m2.VaM2)
    fieldset.add_field(fieldset_m2.VgM2)
    fieldset_s2 = create_fieldset_tidal_currents(datahead=tidaldir, name='S2', filename='s2')
    fieldset_s2.UaS2.set_scaling_factor(1e-2)
    fieldset_s2.UgS2.set_scaling_factor(deg2rad)  # convert from degrees to radians
    fieldset_s2.VaS2.set_scaling_factor(1e-2)  # cm/s to m/s
    fieldset_s2.VgS2.set_scaling_factor(deg2rad)
    fieldset_s2.UaS2.units = GeographicPolar()
    fieldset_s2.VaS2.units = Geographic()
    fieldset.add_field(fieldset_s2.UaS2)
    fieldset.add_field(fieldset_s2.UgS2)
    fieldset.add_field(fieldset_s2.VaS2)
    fieldset.add_field(fieldset_s2.VgS2)
    fieldset_k1 = create_fieldset_tidal_currents(datahead=tidaldir, name='K1', filename='k1')
    fieldset_k1.UaK1.set_scaling_factor(1e-2)
    fieldset_k1.UgK1.set_scaling_factor(deg2rad)  # convert from degrees to radians
    fieldset_k1.VaK1.set_scaling_factor(1e-2)  # cm/s to m/s
    fieldset_k1.VgK1.set_scaling_factor(deg2rad)
    fieldset_k1.UaK1.units = GeographicPolar()
    fieldset_k1.VaK1.units = Geographic()
    fieldset.add_field(fieldset_k1.UaK1)
    fieldset.add_field(fieldset_k1.UgK1)
    fieldset.add_field(fieldset_k1.VaK1)
    fieldset.add_field(fieldset_k1.VgK1)
    fieldset_o1 = create_fieldset_tidal_currents(datahead=tidaldir, name='O1', filename='o1')
    fieldset_o1.UaO1.set_scaling_factor(1e-2)
    fieldset_o1.UgO1.set_scaling_factor(deg2rad)  # convert from degrees to radians
    fieldset_o1.VaO1.set_scaling_factor(1e-2)  # cm/s to m/s
    fieldset_o1.VgO1.set_scaling_factor(deg2rad)
    fieldset_o1.UaO1.units = GeographicPolar()
    fieldset_o1.VaO1.units = Geographic()
    fieldset.add_field(fieldset_o1.UaO1)
    fieldset.add_field(fieldset_o1.UgO1)
    fieldset.add_field(fieldset_o1.VaO1)
    fieldset.add_field(fieldset_o1.VgO1)
    fieldset_m4 = create_fieldset_tidal_currents(datahead=tidaldir, name='M4', filename='m4')
    fieldset_m4.UaM4.set_scaling_factor(1e-2)
    fieldset_m4.UgM4.set_scaling_factor(deg2rad)  # convert from degrees to radians
    fieldset_m4.VaM4.set_scaling_factor(1e-2)  # cm/s to m/s
    fieldset_m4.VgM4.set_scaling_factor(deg2rad)
    fieldset_m4.UaM4.units = GeographicPolar()
    fieldset_m4.VaM4.units = Geographic()
    fieldset.add_field(fieldset_m4.UaM4)
    fieldset.add_field(fieldset_m4.UgM4)
    fieldset.add_field(fieldset_m4.VaM4)
    fieldset.add_field(fieldset_m4.VgM4)
    fieldset_ms4 = create_fieldset_tidal_currents(datahead=tidaldir, name='MS4', filename='ms4')
    fieldset_ms4.UaMS4.set_scaling_factor(1e-2)
    fieldset_ms4.UgMS4.set_scaling_factor(deg2rad)  # convert from degrees to radians
    fieldset_ms4.VaMS4.set_scaling_factor(1e-2)  # cm/s to m/s
    fieldset_ms4.VgMS4.set_scaling_factor(deg2rad)
    fieldset_ms4.UaMS4.units = GeographicPolar()
    fieldset_ms4.VaMS4.units = Geographic()
    fieldset.add_field(fieldset_ms4.UaMS4)
    fieldset.add_field(fieldset_ms4.UgMS4)
    fieldset.add_field(fieldset_ms4.VaMS4)
    fieldset.add_field(fieldset_ms4.VgMS4)
    fieldset_s4 = create_fieldset_tidal_currents(datahead=tidaldir, name='S4', filename='s4')
    fieldset_s4.UaS4.set_scaling_factor(1e-2)
    fieldset_s4.UgS4.set_scaling_factor(deg2rad)  # convert from degrees to radians
    fieldset_s4.VaS4.set_scaling_factor(1e-2)  # cm/s to m/s
    fieldset_s4.VgS4.set_scaling_factor(deg2rad)
    fieldset_s4.UaS4.units = GeographicPolar()
    fieldset_s4.VaS4.units = Geographic()
    fieldset.add_field(fieldset_s4.UaS4)
    fieldset.add_field(fieldset_s4.UgS4)
    fieldset.add_field(fieldset_s4.VaS4)
    fieldset.add_field(fieldset_s4.VgS4)
    omega_M2 = 28.9841042  # angular frequency of M2 in degrees per hour
    fieldset.add_constant('omegaM2', (omega_M2 * deg2rad) / 3600.0)  # angular frequency of M2 in radians per second
    omega_S2 = 30.0000000  # angular frequency of S2 in degrees per hour
    fieldset.add_constant('omegaS2', (omega_S2 * deg2rad) / 3600.0)  # angular frequency of S2 in radians per second
    omega_K1 = 15.0410686  # angular frequency of K1 in degrees per hour
    fieldset.add_constant('omegaK1', (omega_K1 * deg2rad) / 3600.0)  # angular frequency of K1 in radians per second
    omega_O1 = 13.9430356  # angular frequency of O1 in degrees per hour
    fieldset.add_constant('omegaO1', (omega_O1 * deg2rad) / 3600.0)  # angular frequency of O1 in radians per second
    # source for summation of frequencies: (Andersen, 1999)
    omega_M4 = omega_M2 + omega_M2  # angular frequency of S2 in degrees per hour
    fieldset.add_constant('omegaM4', (omega_M4 * deg2rad) / 3600.0)  # angular frequency of S2 in radians per second
    omega_MS4 = omega_M2 + omega_S2  # angular frequency of K1 in degrees per hour
    fieldset.add_constant('omegaMS4', (omega_MS4 * deg2rad) / 3600.0)  # angular frequency of K1 in radians per second
    omega_S4 = omega_S2 + omega_S2  # angular frequency of O1 in degrees per hour
    fieldset.add_constant('omegaS4', (omega_S4 * deg2rad) / 3600.0)  # angular frequency of O1 in radians per second

    out_fname = "tides"
    p_center_y, p_center_x = np.meshgrid(centers_y, centers_x, sparse=False, indexing='ij')
    sample_pset = ParticleSet(fieldset=fieldset, pclass=SampleParticle, lon=np.array(p_center_x).flatten(), lat=np.array(p_center_y).flatten(), time=sample_time)
    sample_kernel = sample_pset.Kernel(sample_uv, delete_cfiles=True)
    # sample_kernel += sample_pset.Kernel(TidalMotionM2S2K1O1, delete_cfiles=True)
    # sample_kernel += sample_pset.Kernel(validate, delete_cfiles=True)
    sample_outname = out_fname + "_sampleuv"
    sample_output_file = sample_pset.ParticleFile(name=os.path.join(odir,sample_outname+".nc"), outputdt=timedelta(seconds=outdt_seconds))
    delete_func = RenewParticle
    # if deleteBC:
    #     delete_func=DeleteParticle
    postProcessFuncs = []
    # postProcessFuncs.append(perIterGC)
    sample_pset.execute(sample_kernel, runtime=timedelta(days=time_in_days), dt=timedelta(seconds=dt_seconds), output_file=sample_output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=timedelta(seconds=outdt_seconds))
    sample_output_file.close()
    del sample_output_file
    del sample_pset
    del sample_kernel
    del fieldset_m2
    del fieldset_s2
    del fieldset_k1
    del fieldset_o1
    del fieldset_m4
    del fieldset_ms4
    del fieldset_s4
    del fieldset
    print("UV on CMEMS grid sampled.")

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
    assert ctime_array_s.shape[1] == global_fT.shape[0]
    mask_array_s = valid_array
    for ti in range(ctime_array_s.shape[1]):
        replace_indices = np.isnan(ctime_array_s[:, ti])

        ctime_array_s[replace_indices, ti] = time_in_max_s[ti]  # in this application, it should always work cause there's no delauyed release
    if DBG_MSG:
        print("time info from file before baselining: shape = {} type = {} range = ({}, {})".format(ctime_array_s.shape, type(ctime_array_s[0 ,0]), np.min(ctime_array_s[0]), np.max(ctime_array_s[0])))
    timebase_s = time_in_max_s[0]
    dtime_array_s = ctime_array_s - timebase_s
    if DBG_MSG:
        print("time info from file after baselining: shape = {} type = {} range = {}".format( dtime_array_s.shape, type(dtime_array_s[0 ,0]), (np.min(dtime_array_s), np.max(dtime_array_s)) ))

    psX = sample_xarray['lon']
    psY = sample_xarray['lat']
    psZ = None
    if 'depth' in sample_xarray.keys():
        psZ = sample_xarray['depth']
    elif 'z' in sample_xarray.keys():
        psZ = sample_xarray['z']
    np.set_printoptions(linewidth=160)
    ns_per_sec = np.timedelta64(1, 's')  # nanoseconds in an sec
    psT = dtime_array_s
    global_psT = time_in_max_s -time_in_max_s[0]
    psT = convert_timearray(psT, outdt_seconds, ns_per_sec, debug=DBG_MSG, array_name="psT")
    global_psT = convert_timearray(global_psT, outdt_seconds, ns_per_sec, debug=DBG_MSG, array_name="global_psT")
    psU = sample_xarray['sample_u']  # to be loaded from pfile
    psV = sample_xarray['sample_v']  # to be loaded from pfile
    print("Sampled data loaded.")

    print("Interpolating UV on a regular-square grid ...")
    total_items = psT.shape[1]
    for ti in range(psT.shape[1]):
        us_local = np.expand_dims(psU[:, ti], axis=1)
        us_local[~mask_array_s, :] = 0
        vs_local = np.expand_dims(psV[:, ti], axis=1)
        vs_local[~mask_array_s, :] = 0
        if ti == 0 and DBG_MSG:
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
    us_file_ds.attrs['mean'] = us_statistics[0] / float(psX.shape[0])
    us_file_ds.attrs['std'] = us_statistics[1] / float(psX.shape[0])
    us_file.close()
    vs_file_ds.attrs['min'] = vs_minmax[0]
    vs_file_ds.attrs['max'] = vs_minmax[1]
    vs_file_ds.attrs['mean'] = vs_statistics[0] / float(psX.shape[1])
    vs_file_ds.attrs['std'] = vs_statistics[1] / float(psX.shape[1])
    vs_file.close()

    del centers_x
    del centers_y
    del xval
    del yval
    del global_fT
