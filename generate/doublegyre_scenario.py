"""
Author: Dr. Christian Kehl
Date: 11-02-2020
"""

from parcels import AdvectionEE, AdvectionRK45, AdvectionRK4, DiffusionUniformKh, AdvectionDiffusionEM, AdvectionDiffusionM1
from parcels import FieldSet, RectilinearZGrid, ScipyParticle, JITParticle, Variable, StateCode, OperationCode, ErrorCode
# from parcels import ParticleSetAOS, ParticleSetSOA
# from parcels.particleset import ParticleSet as DryParticleSet
from parcels.particleset import ParticleSet as BenchmarkParticleSet
from parcels.field import Field, VectorField, NestedField, SummedField
# from parcels import plotTrajectoriesFile_loadedField
# from parcels import rng as random
from parcels import ParcelsRandom
from argparse import ArgumentParser
from datetime import timedelta as delta
from glob import glob
import math
import datetime
import numpy as np
from numpy.random import default_rng

import xarray as xr
import dask.array as da

import fnmatch
import sys
import gc
import os
import time as ostime
from scipy.interpolate import interpn
import h5py

from multiprocessing import Pool

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
#Nparticle = int(math.pow(2,19)) # equals to Nparticle = 524288

# a = 3.6 * 1e2  # by definition: meters
a = 359.0
# b = 1.8 * 1e2  # by definiton: meters
b = 179.0
c = 2.1 * 1e3  # by definiton: meters
tsteps = 122 # in steps
tstepsize = 6.0 # unitary
tscale = 12.0*60.0*60.0 # in seconds
# time[days] = (tsteps * tstepsize) * tscale
# ==== yearly rotation - initial estimate ==== #
# gyre_rotation_speed = 60.0*24.0*60.0*60.0  # assume 1 rotation every 8.5 weeks
# ==== yearly rotation - 3D use ==== #
gyre_rotation_speed = 30.5*24.0*60.0*60.0  # assume 1 rotation every 4.02 weeks
# ==== yearly rotation - 2D use ==== #
# gyre_rotation_speed = 366.0*24.0*60.0*60.0  # assume 1 rotation every 52 weeks

# ==== INFO FROM NEMO-MEDUSA: realistic values are 0-2.5 [m/s] ==== #
# scalefac = (40.0 / (1000.0/ (60.0 * 60.0)))  # 40 km/h
scalefactor = ((4.0*1000) / (60.0*60.0))  # 4 km/h
vertical_scale = (800.0 / (24*60.0*60.0))  # 800 m/d
# ==== ONLY APPLY BELOW SCALING IF MESH IS FLAT AND (a, b) are below 100,000 [m] ==== #
v_scale_small = 1./1000.0 # this is to adapt, cause 1 U = 1 m/s = 1 spatial unit / time unit; spatial scale; domain = 1920 m x 960 m -> scale needs to be adapted to to interpret speed on a 1920 km x 960 km grid



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

def DeleteParticle(particle, fieldset, time):
    particle.delete()


def RenewParticle(particle, fieldset, time):
    EA = fieldset.east_lim
    WE = fieldset.west_lim
    dlon = EA - WE
    NO = fieldset.north_lim
    SO = fieldset.south_lim
    dlat = NO - SO
    if particle.lon < WE:
        particle.lon += dlon
    if particle.lon > EA:
        particle.lon -= dlon
    if particle.lat < SO:
        particle.lat += dlat
    if particle.lat > NO:
        particle.lat -= dlat
    if fieldset.isThreeD > 0.0:
        TO = fieldset.top
        BO = fieldset.bottom
        if particle.depth < TO:
            particle.depth += 1.0
        if particle.depth > BO:
            particle.depth -= 1.0

    # particle.lon = WE + ParcelsRandom.random() * (EA - WE)
    # particle.lat = SO + ParcelsRandom.random() * (NO - SO)
    # if fieldset.isThreeD > 0.0:
    #     TO = fieldset.top
    #     BO = fieldset.bottom
    #     particle.depth = TO + ((BO-TO) / 0.75) + ((BO-TO) * -0.5 * ParcelsRandom.random())


def WrapClipParticle(particle, fieldSet, time):
    EA = fieldset.east_lim
    WE = fieldset.west_lim
    dlon = EA - WE
    NO = fieldset.north_lim
    SO = fieldset.south_lim
    dlat = NO - SO
    # if particle.lon < WE:
    if particle.lon < -(dlon/2.0):
        # particle.lon += dlon
        particle.lon += math.fabs(particle.lon - dlon)
    # if particle.lon > EA:
    if particle.lon > (dlon/2,0):
        # particle.lon -= dlon
        particle.lon -= math.fabs(particle.lon - dlon)
    if particle.lat < SO:
        # particle.lat += dlat
        particle.lat += math.fabs(particle.lat - dlat)
    if particle.lat > NO:
        # particle.lat -= dlat
        particle.lat -= math.fabs(particle.lat - dlat)


periodicBC = WrapClipParticle


def perIterGC():
    gc.collect()


# initialize a worker in the process pool
def worker_init(lon, lat, depth, freqs, a, b, c, epsilon, A):
    global var_lon
    var_lon = lon
    global var_lat
    var_lat = lat
    global var_depth
    var_depth = depth
    global var_freqs
    var_freqs = freqs
    global var_a
    var_a = a
    global var_b
    var_b = b
    global var_c
    var_c = c
    global var_epsilon
    var_epsilon = epsilon
    global var_Alpha
    var_Alpha = A


# lon, lat, depth, freq, a, b, c, epsilon, A
def doublgyre3D_func(ti, i, j, k):
    freq = var_freqs[ti]
    x1 = ((var_lon[i] + (var_a / 2.0)) / var_a)
    x2 = ((var_lat[j] + (var_b / 2.0)) / var_b)
    x3 = (var_depth[k] / var_c)
    f_xt = (var_epsilon * np.sin(freq) * x1 ** 2.0) + (1.0 - (2.0 * var_epsilon * np.sin(freq))) * x1
    val_u = -np.pi * var_Alpha * np.sin(np.pi * f_xt) * np.cos(np.pi * x2)
    val_v = np.pi * var_Alpha * np.cos(np.pi * f_xt) * np.sin(np.pi * x2) * (
            2 * var_epsilon * np.sin(freq) * x1 + 1 - 2 * var_epsilon * np.sin(freq))
    val_w = 0.2 * x3 * (1.0 - x3) * (x3 - var_epsilon * np.sin(4 * np.pi * freq) - 0.5)
    val_w = val_w * -1.0
    return (ti, i, j, k, val_u, val_v, val_w)


def doublegyre_waves3D(xdim=960, ydim=480, zdim=20, periodic_wrap=False, write_out=False, steady=False, mesh='flat'):
    """Implemented following Froyland and Padberg (2009), 10.1016/j.physd.2009.03.002
       Also: Ph.D. Thesis A.S. Parkash, p.35 - 37
       see: https://vcg.iwr.uni-heidelberg.de/plugins/vcg_-_double_gyre_3d"""
    A = 0.3  # range: [0 .. 1]
    epsilon = 0.25  # range: [0 .. 1]
    omega = 2 * np.pi * 1.0  # periodicity: [0.1 .. 10.0]

    scalefac = scalefactor
    if 'flat' in mesh and np.maximum(a, b) < 100000:
        scalefac *= v_scale_small

    lon = np.linspace(-a*0.5, a*0.5, xdim, dtype=np.float32)
    lonrange = lon.max()-lon.min()
    sys.stdout.write("lon field: {}\n".format(lon.size))
    lat = np.linspace(-b*0.5, b*0.5, ydim, dtype=np.float32)
    latrange = lat.max() - lat.min()
    sys.stdout.write("lat field: {}\n".format(lat.size))
    depth = np.linspace(0.0, c, zdim, dtype=np.float32)
    depthrange = depth.max() - depth.min()
    sys.stdout.write("depth field: {}\n".format(depth.size))
    totime = (tsteps * tstepsize) * tscale
    times = np.linspace(0., totime, tsteps, dtype=np.float64)
    sys.stdout.write("time field: {}\n".format(times.size))
    dx, dy, dz = lon[2]-lon[1], lat[2]-lat[1], depth[2]-depth[1]

    U = np.zeros((times.size, depth.size, lat.size, lon.size), dtype=np.float32)
    V = np.zeros((times.size, depth.size, lat.size, lon.size), dtype=np.float32)
    W = np.zeros((times.size, depth.size, lat.size, lon.size), dtype=np.float32)
    freqs = np.ones(times.size, dtype=np.float32)
    if not steady:
        for ti in range(times.shape[0]):
            time_f = times[ti] / gyre_rotation_speed
            freqs[ti] *= omega * time_f
    else:
        freqs = (freqs * 0.5) * omega

    # # xcoords, ycoords, zcoords = np.meshgrid(lon, lat, depth, sparse=False, indexing='ij')
    # for ti in range(times.shape[0]):
    #     freq = freqs[ti]
    #     for i in range(lon.shape[0]):
    #         for j in range(lat.shape[0]):
    #             for k in range(depth.shape[0]):
    #                 x1 = ((lon[i] + (a/2.0)) / a)
    #                 x2 = ((lat[j] + (b/2.0)) / b)
    #                 x3 = (depth[k] / c)
    #                 f_xt = (epsilon * np.sin(freq) * x1**2.0) + (1.0 - (2.0 * epsilon * np.sin(freq))) * x1
    #                 U[ti, k, j, i] = -np.pi * A * np.sin(np.pi * f_xt) * np.cos(np.pi * x2)
    #                 V[ti, k, j, i] = np.pi * A * np.cos(np.pi * f_xt) * np.sin(np.pi * x2) * (2 * epsilon * np.sin(freq) * x1 + 1 - 2 * epsilon * np.sin(freq))
    #                 W[ti, k, j, i] = 0.2 * x3 * (1.0 - x3) * (x3 - epsilon * np.sin(4 * np.pi * freq) - 0.5)

    # items = zip(range(times.shape[0]), range(lon.shape[0]), range(lat.shape[0]), range(depth.shape[0]))
    # coords_t, coords_x, coords_y, coords_z = np.meshgrid(times, lon, lat, depth, sparse=False, indexing='ij', copy=False)
    idx_t, idx_x, idx_y, idx_z = np.meshgrid(range(times.shape[0]), range(lon.shape[0]), range(lat.shape[0]), range(depth.shape[0]), sparse=False, indexing='ij',copy=False)
    items = zip(idx_t, idx_x, idx_y, idx_z)
    with Pool(initializer=worker_init, initargs=(lon, lat, depth, freqs, a, b, c, epsilon, A)) as pool:
        for result in pool.starmap(doublgyre3D_func, items):
            (ti, i, j, k, val_u, val_v, val_w) = result
            U[ti, k, j, i] = val_u
            V[ti, k, j, i] = val_v
            W[ti, k, j, i] = val_w


    U *= scalefac
    # U = np.transpose(U, (0, 2, 1))
    V *= scalefac
    # V = np.transpose(V, (0, 2, 1))
    W *= scalefac
    # W = np.transpose(W, (0, 2, 1))
    Kh_zonal = np.ones(U.shape, dtype=np.float32) * 0.5
    Kh_meridional = np.ones(U.shape, dtype=np.float32) * 0.5


    data = {'U': U, 'V': V, 'W': W, 'Kh_zonal': Kh_zonal, 'Kh_meridional': Kh_meridional}
    dimensions = {'time': times, 'depth': depth, 'lon': lon, 'lat': lat}
    fieldset = None
    if periodic_wrap:
        fieldset = FieldSet.from_data(data, dimensions, mesh=mesh, transpose=False, time_periodic=delta(days=366))
    else:
        fieldset = FieldSet.from_data(data, dimensions, mesh=mesh, transpose=False, allow_time_extrapolation=True)
    # fieldset.add_constant_field("Kh_zonal", 1, mesh="flat")
    # fieldset.add_constant_field("Kh_meridonal", 1, mesh="flat")
    fieldset.add_constant("dres", 0.01)
    if write_out:
        fieldset.write(filename=write_out)
    return lon, lat, depth, times, U, V, W, fieldset


def doublegyre_from_numpy(xdim=960, ydim=480, periodic_wrap=False, write_out=False, steady=False, mesh='flat', anisotropic_diffusion=False):
    """Implemented following Froyland and Padberg (2009), 10.1016/j.physd.2009.03.002"""
    A = 0.3  # range: [0 .. 1]
    epsilon = 0.25  # range: [0 .. 1]
    omega = 2 * np.pi * 1.0  # periodicity: [0.1 .. 10.0]

    scalefac = scalefactor
    if 'flat' in mesh and np.maximum(a, b) > 370.0 and np.maximum(a, b) < 100000:
        scalefac *= v_scale_small

    lon = np.linspace(-a*0.5, a*0.5, xdim, dtype=np.float32)
    lonrange = lon.max()-lon.min()
    sys.stdout.write("lon field: {}\n".format(lon.size))
    lat = np.linspace(-b*0.5, b*0.5, ydim, dtype=np.float32)
    latrange = lat.max() - lat.min()
    sys.stdout.write("lat field: {}\n".format(lat.size))
    totime = (tsteps * tstepsize) * tscale
    times = np.linspace(0., totime, tsteps, dtype=np.float64)
    sys.stdout.write("time field: {}\n".format(times.size))
    dx, dy = lon[2]-lon[1], lat[2]-lat[1]

    U = np.zeros((times.size, lat.size, lon.size), dtype=np.float32)
    V = np.zeros((times.size, lat.size, lon.size), dtype=np.float32)
    freqs = np.ones(times.size, dtype=np.float32)
    if not steady:
        for ti in range(times.shape[0]):
            time_f = times[ti] / gyre_rotation_speed
            # time_f = np.fmod((times[ti])/gyre_rotation_speed, 2.0)
            # time_f = np.fmod((times[ti]/gyre_rotation_speed), 2*np.pi)
            # freqs[ti] = omega * np.cos(time_f) * 2.0
            freqs[ti] *= omega * time_f
            # freqs[ti] *= time_f
    else:
        freqs = (freqs * 0.5) * omega

    for ti in range(times.shape[0]):
        freq = freqs[ti]
        # print(freq)
        for i in range(lon.shape[0]):
            for j in range(lat.shape[0]):
                x1 = ((lon[i]*2.0 + a) / a) # - dx / 2
                x2 = ((lat[j]*2.0 + b) / (2.0*b)) # - dy / 2
                f_xt = (epsilon * np.sin(freq) * x1**2.0) + (1.0 - (2.0 * epsilon * np.sin(freq))) * x1
                U[ti, j, i] = -np.pi * A * np.sin(np.pi * f_xt) * np.cos(np.pi * x2)
                V[ti, j, i] = np.pi * A * np.cos(np.pi * f_xt) * np.sin(np.pi * x2) * (2 * epsilon * np.sin(freq) * x1 + 1 - 2 * epsilon * np.sin(freq))

    U *= scalefac
    # U = np.transpose(U, (0, 2, 1))
    V *= scalefac
    # V = np.transpose(V, (0, 2, 1))
    Kh_zonal = None
    Kh_meridional = None
    if anisotropic_diffusion:
        print("Generating anisotropic diffusion fields ...")
        Kh_zonal = np.ones((lat.size, lon.size), dtype=np.float32) * 0.5 * 100.
        Kh_meridional = np.empty((lat.size, lon.size), dtype=np.float32)
        alpha = 1.  # Profile steepness
        L = 1.  # Basin scale
        # Ny = lat.shape[0]  # Number of grid cells in y_direction (101 +2, one level above and one below, where fields are set to zero)
        # dy = 1.03 / Ny  # Spatial resolution
        # y = np.linspace(-0.01, 1.01, 103)  # y-coordinates for grid
        # y_K = np.linspace(0., 1., 101)  # y-coordinates used for setting diffusivity
        beta = np.zeros(lat.shape[0])  # Placeholder for fraction term in K(y) formula

        # for yi in range(len(y_K)):
        for yi in range(lat.shape[0]):
            yk = ((lat[yi]*2.0 + b) / (2.0*b))
            if yk < L / 2:
                beta[yi] = yk * np.power(L - 2 * yk, 1 / alpha)
            elif yk >= L / 2:
                beta[yi] = (L - yk) * np.power(2 * yk - L, 1 / alpha)
        Kh_meridional_profile = 0.1 * (2 * (1 + alpha) * (1 + 2 * alpha)) / (alpha ** 2 * np.power(L, 1 + 1 / alpha)) * beta
        for i in range(lon.shape[0]):
            for j in range(lat.shape[0]):
                Kh_meridional[j, i] = Kh_meridional_profile[j] * 100.
    else:
        print("Generating isotropic diffusion value ...")
        # Kh_zonal = np.ones((ydim, xdim), dtype=np.float32) * np.random.uniform(0.85, 1.15) * 100.0  # in m^2/s
        # Kh_meridional = np.ones((ydim, xdim), dtype=np.float32) * np.random.uniform(0.7, 1.3) * 100.0  # in m^2/s
        # mesh_conversion = 1.0 / 1852. / 60 if fieldset.U.grid.mesh == 'spherical' else 1.0
        Kh_zonal = np.random.uniform(0.85, 1.15) * 100.0  # in m^2/s
        Kh_meridional = np.random.uniform(0.7, 1.3) * 100.0  # in m^2/s


    data = {'U': U, 'V': V}
    dimensions = {'time': times, 'lon': lon, 'lat': lat}
    fieldset = None
    if periodic_wrap:
        fieldset = FieldSet.from_data(data, dimensions, mesh=mesh, transpose=False, time_periodic=delta(days=366))
    else:
        fieldset = FieldSet.from_data(data, dimensions, mesh=mesh, transpose=False, allow_time_extrapolation=True)

    if write_out:
        fieldset.write(filename=write_out)
    if anisotropic_diffusion:
        Kh_grid = RectilinearZGrid(lon=fieldset.U.lon, lat=fieldset.U.lat, mesh=mesh)
        # fieldset.add_field(Field("Kh_zonal", Kh_zonal, lon=lon, lat=lat, to_write=False, mesh=mesh, transpose=False))
        # fieldset.add_field(Field("Kh_meridional", Kh_meridional, lon=lon, lat=lat, to_write=False, mesh=mesh, transpose=False))
        fieldset.add_field(Field("Kh_zonal", Kh_zonal, grid=Kh_grid, to_write=False, mesh=mesh, transpose=False, allow_time_extrapolation=True))
        fieldset.add_field(Field("Kh_meridional", Kh_meridional, grid=Kh_grid, to_write=False, mesh=mesh, transpose=False, allow_time_extrapolation=True))
        fieldset.add_constant("dres", max(lat[1]-lat[0], lon[1]-lon[0]))
    else:
        fieldset.add_constant_field("Kh_zonal", Kh_zonal, mesh=mesh)
        fieldset.add_constant_field("Kh_meridional", Kh_meridional, mesh=mesh)
        # fieldset.add_constant("Kh_zonal", Kh_zonal)
        # fieldset.add_constant("Kh_meridional", Kh_meridional)

    return lon, lat, times, U, V, fieldset

# def doublegyre_from_numpy_3D(xdim=960, ydim=480, periodic_wrap=False, write_out=False, steady=False, mesh='flat', anisotropic_diffusion=False):
#     """Implemented following Froyland and Padberg (2009), 10.1016/j.physd.2009.03.002"""
#     A = 0.3
#     epsilon = 0.25
#     omega = 2 * np.pi
#
#     scalefac = scalefactor
#     if 'flat' in mesh and np.maximum(a, b) > 370.0 and np.maximum(a, b) < 100000:
#         scalefac *= v_scale_small
#
#     lon = np.linspace(-a*0.5, a*0.5, xdim, dtype=np.float32)
#     lonrange = lon.max()-lon.min()
#     sys.stdout.write("lon field: {}\n".format(lon.size))
#     lat = np.linspace(-b*0.5, b*0.5, ydim, dtype=np.float32)
#     latrange = lat.max() - lat.min()
#     sys.stdout.write("lat field: {}\n".format(lat.size))
#     totime = (tsteps * tstepsize) * tscale
#     times = np.linspace(0., totime, tsteps, dtype=np.float64)
#     sys.stdout.write("time field: {}\n".format(times.size))
#     dx, dy = lon[2]-lon[1], lat[2]-lat[1]
#
#     U = np.zeros((times.size, lat.size, lon.size), dtype=np.float32)
#     V = np.zeros((times.size, lat.size, lon.size), dtype=np.float32)
#     freqs = np.ones(times.size, dtype=np.float32)
#     if not steady:
#         for ti in range(times.shape[0]):
#             time_f = times[ti] / gyre_rotation_speed
#             # time_f = np.fmod((times[ti])/gyre_rotation_speed, 2.0)
#             # time_f = np.fmod((times[ti]/gyre_rotation_speed), 2*np.pi)
#             # freqs[ti] = omega * np.cos(time_f) * 2.0
#             freqs[ti] *= omega * time_f
#             # freqs[ti] *= time_f
#     else:
#         freqs = (freqs * 0.5) * omega
#
#     for ti in range(times.shape[0]):
#         freq = freqs[ti]
#         # print(freq)
#         for i in range(lon.shape[0]):
#             for j in range(lat.shape[0]):
#                 x1 = ((lon[i] + (a/2.0)) / a)  # - dx / 2
#                 x2 = ((lat[j] + (b/2.0)) / b)  # - dy / 2 || / (2.0*)
#                 f_xt = (epsilon * np.sin(freq) * x1**2.0) + (1.0 - (2.0 * epsilon * np.sin(freq))) * x1
#                 U[ti, j, i] = -np.pi * A * np.sin(np.pi * f_xt) * np.cos(np.pi * x2)
#                 V[ti, j, i] = np.pi * A * np.cos(np.pi * f_xt) * np.sin(np.pi * x2) * (2 * epsilon * np.sin(freq) * x1 + 1 - 2 * epsilon * np.sin(freq))
#
#     U *= scalefac
#     # U = np.transpose(U, (0, 2, 1))
#     V *= scalefac
#     # V = np.transpose(V, (0, 2, 1))
#     Kh_zonal = np.ones(U.shape, dtype=np.float32) * 0.5
#     Kh_meridional = np.ones(U.shape, dtype=np.float32) * 0.5
#
#
#     data = {'U': U, 'V': V, 'Kh_zonal': Kh_zonal, 'Kh_meridional': Kh_meridional}
#     dimensions = {'time': times, 'lon': lon, 'lat': lat}
#     fieldset = None
#     if periodic_wrap:
#         fieldset = FieldSet.from_data(data, dimensions, mesh=mesh, transpose=False, time_periodic=delta(days=366))
#     else:
#         fieldset = FieldSet.from_data(data, dimensions, mesh=mesh, transpose=False, allow_time_extrapolation=True)
#     # fieldset.add_constant_field("Kh_zonal", 1, mesh="flat")
#     # fieldset.add_constant_field("Kh_meridonal", 1, mesh="flat")
#     fieldset.add_constant("dres", 0.01)
#     if write_out:
#         fieldset.write(filename=write_out)
#     return lon, lat, times, U, V, fieldset


def fieldset_from_file(periodic_wrap=False, filepath=None, simtime_days=None, diffusion=False):
    """
    Implemented following Froyland and Padberg (2009), 10.1016/j.physd.2009.03.002
    """
    if filepath is None:
        return None
    extra_fields = {}
    head_dir = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    fname, fext = os.path.splitext(basename)
    fext = ".nc" if len(fext) == 0 else fext
    flen = len(fname)
    field_fnames = glob(os.path.join(head_dir, fname+"*"+fext))
    for field_fname in field_fnames:
        field_fname, field_fext = os.path.splitext(os.path.basename(field_fname))
        field_indicator = field_fname[flen:]
        if field_indicator not in ["U", "V"]:
            extra_fields[field_indicator] = field_indicator
    if len(list(extra_fields.keys())) < 1:
        extra_fields = None

    a, b, c = 1.0, 1.0, 1.0
    simtime_days = 366 if simtime_days is None else simtime_days

    fieldset = None
    if periodic_wrap:
        fieldset = FieldSet.from_parcels(os.path.join(head_dir, fname), extra_fields=extra_fields, time_periodic=delta(days=simtime_days).total_seconds(), deferred_load=True, allow_time_extrapolation=False, chunksize='auto')
        # return FieldSet.from_xarray_dataset(ds, variables, dimensions, mesh='flat', time_periodic=delta(days=366))
    else:
        fieldset = FieldSet.from_parcels(os.path.join(head_dir, fname), extra_fields=extra_fields, time_periodic=None, deferred_load=True, allow_time_extrapolation=True, chunksize='auto')
        # return FieldSet.from_xarray_dataset(ds, variables, dimensions, mesh='flat', allow_time_extrapolation=True)
    #print(fieldset.U.grid.__dict__)
    ftimes = fieldset.U.grid.time
    flons = fieldset.U.lon
    a = flons[len(flons)-1] - flons[0]
    flats = fieldset.V.lat
    b = flats[len(flats) - 1] - flats[0]
    fdepths = None
    if "W" in extra_fields:
        fdepths = fieldset.W.depth
        c = fdepths[len(fdepths)-1] - fdepths[0]
        # fieldset.add_constant("top", depth[0] + 0.001)
        # fieldset.add_constant("bottom", depth[len(depth)-1] - 0.001)
    Kh_zonal = None
    Kh_meridional = None
    if diffusion:
        Kh_zonal = np.random.uniform(9.5, 10.5)  # in m^2/s
        Kh_meridional = np.random.uniform(7.5, 12.5)  # in m^2/s
        Kh_zonal, Kh_meridional = Kh_zonal * 100.0, Kh_meridional * 100.0  # because the mesh is flat
        fieldset.add_constant_field("Kh_zonal", Kh_zonal, mesh="flat")
        fieldset.add_constant_field("Kh_meridional", Kh_meridional, mesh="flat")
    return flons, flats, fdepths, ftimes, fieldset, a, b, c


class AgeParticle_JIT(JITParticle):
    age = Variable('age', dtype=np.float64, initial=0.0, to_write=True)
    age_d = Variable('age_d', dtype=np.float32, initial=0.0, to_write=True)
    life_expectancy = Variable('life_expectancy', dtype=np.float64, initial=np.finfo(np.float64).max, to_write=False)
    initialized_dynamic = Variable('initialized_dynamic', dtype=np.int32, initial=0, to_write=False)

class AgeParticle_SciPy(ScipyParticle):
    age = Variable('age', dtype=np.float64, initial=0.0)
    age_d = Variable('age_d', dtype=np.float32, initial=0.0, to_write=True)
    life_expectancy = Variable('life_expectancy', dtype=np.float64, initial=np.finfo(np.float64).max, to_write=False)
    initialized_dynamic = Variable('initialized_dynamic', dtype=np.int32, initial=0, to_write=False)

def initialize(particle, fieldset, time):
    if particle.initialized_dynamic < 1:
        np_scaler = math.sqrt(3.0 / 2.0)
        particle.life_expectancy = time + ParcelsRandom.uniform(.0, (fieldset.life_expectancy-time) * 2.0 / np_scaler)
        particle.initialized_dynamic = 1

def Age(particle, fieldset, time):
    if particle.state == StateCode.Evaluate:
        particle.age = particle.age + math.fabs(particle.dt)
        particle.age_d = particle.age/86400.0
    if particle.age > particle.life_expectancy:
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
#                python3 doublegyre_scenario.py -f vis_example/metadata.txt -t 366 -dt 60 -ot 720 -im 'rk4' -N 2**12 -sres 2.5 -gres 5 -sm 'regular_jitter' -fsx 360 -fsy 180
# ====
if __name__=='__main__':
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-f", "--filename", dest="filename", type=str, default="file.txt", help="(relative) (text) file path of the csv hyper parameters")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=1, help="runtime in days (default: 1)")
    parser.add_argument("-dt", "--deltatime", dest="dt", type=int, default=720, help="computational delta_t time stepping in minutes (default: 720min = 12h)")
    parser.add_argument("-ot", "--outputtime", dest="outdt", type=int, default=1440, help="repeating release rate of added particles in minutes (default: 1440min = 24h)")
    parser.add_argument("-im", "--interp_mode", dest="interp_mode", choices=['rk4','rk45', 'ee', 'em', 'm1', 'bm'], default="rk4", help="interpolation mode = [rk4, rk45, ee (Eulerian Estimation), em (Euler-Maruyama), m1 (Milstein-1), bm (Brownian Motion)]")
    # parser.add_argument("-w", "--writeout", dest="write_out", action='store_true', default=False, help="write data in outfile")
    parser.add_argument("-N", "--n_particles", dest="nparticles", type=str, default="2**6", help="number of particles to generate and advect (default: 2e6)")
    parser.add_argument("-sres", "--sample_resolution", dest="sres", type=str, default="2", help="number of particle samples per arc-dgree (default: 2)")
    parser.add_argument("-gres", "--grid_resolution", dest="gres", type=str, default="10", help="number of cells per arc-degree or metre (default: 10)")
    parser.add_argument("-sm", "--samplemode", dest="sample_mode", choices=['regular_jitter','uniform','gaussian','triangular','vonmises'], default='regular_jitter', help="sampling distribution mode = [regular_jitter, (irregular) uniform, (irregular) gaussian, (irregular) triangular, (irregular) vonmises]")
    parser.add_argument("-fsx", "--field_sx", dest="field_sx", type=int, default="480", help="number of original field cells in x-direction")
    parser.add_argument("-fsy", "--field_sy", dest="field_sy", type=int, default="240", help="number of original field cells in y-direction")
    parser.add_argument("-fsz", "--field_sz", dest="field_sz", type=int, default="20", help="number of original field cells in z-direction")
    parser.add_argument("-3D", "--threeD", dest="threeD", action='store_true', default=False, help="make a 3D-simulation (default: False).")
    args = parser.parse_args()

    ParticleSet = BenchmarkParticleSet
    # if args.dryrun:
    #     ParticleSet = DryParticleSet

    filename=args.filename
    field_sx = args.field_sx
    field_sy = args.field_sy
    field_sz = args.field_sz
    deleteBC = False
    animate_result = False
    visualize_results = False
    periodicFlag=True
    backwardSimulation = False
    repeatdtFlag=False
    repeatRateMinutes=720
    time_in_days = args.time_in_days
    time_in_years = int(float(time_in_days)/365.0)
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
    np.random.seed(0)
    use_3D = args.threeD
    use_W = False

    branch = "LOMUQ"
    computer_env = "local/unspecified"
    scenario = "doublegyre"
    if use_3D:
        scenario += "3D"
    odir = ""
    if os.uname()[1] in ['science-bs35', 'science-bs36']:  # Gemini
        # odir = "/scratch/{}/experiments".format(os.environ['USER'])
        odir = "/scratch/{}/experiments".format("ckehl")
        computer_env = "Gemini"
    elif fnmatch.fnmatchcase(os.uname()[1], "*.bullx*"):  # Cartesius
        CARTESIUS_SCRATCH_USERNAME = 'ckehluu'
        odir = "/scratch/shared/{}/experiments".format(CARTESIUS_SCRATCH_USERNAME)
        computer_env = "Cartesius"
    elif fnmatch.fnmatchcase(os.uname()[1], "int*.snellius.*") or fnmatch.fnmatchcase(os.uname()[1], "fcn*") or fnmatch.fnmatchcase(os.uname()[1], "tcn*") or fnmatch.fnmatchcase(os.uname()[1], "gcn*") or fnmatch.fnmatchcase(os.uname()[1], "hcn*"):  # Snellius
        SNELLIUS_SCRATCH_USERNAME = 'ckehluu'
        odir = "/scratch-shared/{}/experiments/".format(SNELLIUS_SCRATCH_USERNAME)
        computer_env = "Snellius"
    elif fnmatch.fnmatchcase(os.uname()[1], "PROMETHEUS"):  # Prometheus computer - use USB drive
        CARTESIUS_SCRATCH_USERNAME = 'christian'
        odir = "/media/{}/DATA/data/hydrodynamics/doublegyre/".format(CARTESIUS_SCRATCH_USERNAME)
        computer_env = "Prometheus"
    else:
        odir = "/var/scratch/experiments"
    print("running {} on {} (uname: {}) - branch '{}' - (target) N: {} - argv: {}".format(scenario, computer_env, os.uname()[1], branch, target_N, sys.argv[1:]))
    if not os.path.exists(odir):
        os.mkdir(odir)

    if os.path.sep in filename:
        head_dir = os.path.dirname(filename)
        if head_dir[0] == os.path.sep:
            odir = head_dir
        else:
            odir = os.path.join(odir, head_dir)
            filename = os.path.split(filename)[1]
    pfname, pfext = os.path.splitext(filename)
    if not os.path.exists(odir):
        os.makedirs(odir)

    func_time = []
    mem_used_GB = []

    fieldset = None
    flons = None
    flats = None
    fdepths = None
    ftimes = None
    U = None
    V = None
    W = None
    field_fpath = False
    if writeout:
        field_fpath = os.path.join(odir,"doublegyre")
    if field_fpath and os.path.exists(field_fpath+"U.nc"):
        flons, flats, fdepths, ftimes, fieldset, a, b, c = fieldset_from_file(periodic_wrap=periodicFlag, filepath=field_fpath+".nc")
        use_3D &= hasattr(fieldset, "W")
        U = fieldset.U
        V = fieldset.V
        if hasattr(fieldset, "W"):
            W = fieldset.W
            use_W = True
        if fdepths is None:
            use_3D = False
    else:
        if not use_3D:
            flons, flats, ftimes, U, V, fieldset = doublegyre_from_numpy(xdim=field_sx, ydim=field_sy, periodic_wrap=periodicFlag, write_out=field_fpath, mesh='spherical', anisotropic_diffusion=(interp_mode in ['em', 'm1']))
        else:
            flons, flats, fdepths, ftimes, U, V, W, fieldset = doublegyre_waves3D(xdim=field_sx, ydim=field_sy, zdim=field_sz, periodic_wrap=periodicFlag, write_out=field_fpath, mesh='spherical')
            use_W = True
    fieldset.add_constant("east_lim", +a * 0.5)
    fieldset.add_constant("west_lim", -a * 0.5)
    fieldset.add_constant("north_lim", +b * 0.5)
    fieldset.add_constant("south_lim", -b * 0.5)
    fieldset.add_constant("top", 0.0)
    fieldset.add_constant("bottom", c)
    fieldset.add_constant("isThreeD", 1.0 if use_3D else -1.0)

    # ================================================== #
    # ----- make running the particle-sim optional ----- #
    # -- skip the simulation if sim-file is available -- #
    # ================================================== #
    out_fname = "particles-doublegyre"
    if MPI and (MPI.COMM_WORLD.Get_size()>1):
        out_fname += "_MPI"
    else:
        out_fname += "_noMPI"
    if periodicFlag:
        out_fname += "_p"
    out_fname += "_n"+str(int(sres*a*sres*b))
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

    if not os.path.exists(os.path.join(odir, out_fname + ".nc")):
        # ==== ==== ==== ==== ==== #
        # == EXECUTE SIMULATION == #
        # ==== ==== ==== ==== ==== #
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

        step = 1.0/gres
        zstep = gres*10.0
        xsteps = int(np.floor(a * gres))
        # xsteps = int(np.ceil(a * gres))
        ysteps = int(np.floor(b * gres))
        # ysteps = int(np.ceil(b * gres))
        zsteps = int(np.floor(c * (1.0/(gres*10.0))))
        # zsteps = int(np.ceil(c * gres))
        if DBG_MSG:
            print("=========================================================")
            print("step: gres={}, value={}".format(gres, step))
            print("zstep: gres={}, value={}".format(gres, zstep))
            print("xsteps: a={}, gres={}, value={}".format(a, gres, xsteps))
            print("xsteps: b={}, gres={}, value={}".format(b, gres, ysteps))
            print("xsteps: c={}, gres={}, value={}".format(c, gres, zsteps))
            print("=========================================================")
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
        if writeout:
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
            kernelfunc = AdvectionDiffusionEM
        elif interp_mode == 'm1':
            kernelfunc = AdvectionDiffusionM1

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
                pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=-dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12), moviedt=delta(hours=6), movie_background_field=fieldset.U)
            else:
                pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=-dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12))
        else:
            # ==== forward simulation ==== #
            if animate_result:
                pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12), moviedt=delta(hours=6), movie_background_field=fieldset.U)
            else:
                pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12))

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
    zstep = gres*50.0
    xsteps = int(np.floor(a * gres))
    # xsteps = int(np.ceil(a * gres))
    ysteps = int(np.floor(b * gres))
    # ysteps = int(np.ceil(b * gres))
    zsteps = int(np.floor(c * (1.0/(gres*50.0))))
    # zsteps = int(np.ceil(c * gres))

    data_xarray = xr.open_dataset(os.path.join(odir, out_fname + ".nc"))
    N = data_xarray['lon'].data.shape[0]
    tN = data_xarray['lon'].data.shape[1]
    if DBG_MSG:
        print("N: {}, t_N: {}".format(N, tN))
    np.set_printoptions(linewidth=160)
    ns_per_sec = np.timedelta64(1, 's')  # nanoseconds in an sec
    if DBG_MSG:
        print("ns_per_sec = {}".format((ns_per_sec/np.timedelta64(1, 'ns')).astype(np.float64)))
    sec_per_day = 86400.0
    ctime_array = data_xarray['time'].data
    if DBG_MSG:
        print("time info from file before baselining: shape = {} type = {} range = ({}, {})".format(ctime_array.shape, type(ctime_array[0 ,0]), ctime_array[0].min(), ctime_array[0].max()))
    timebase = ctime_array[:, 0]
    # time_in_min = np.nanmin(ctime_array, axis=0)
    # time_in_max = np.nanmax(ctime_array, axis=0)
    # timebase = time_in_max[0]
    dtime_array = np.transpose(ctime_array.transpose() - timebase)
    if DBG_MSG:
        print("time info from file after baselining: shape = {} type = {} range = ({}, {})".format(dtime_array.shape, type(dtime_array[0 ,0]), dtime_array[0].min(), dtime_array[0].max()))
    # print(dtime_array.dtype)
    # print(ns_per_sec.dtype)
    # pX = data_xarray['lon'].data[indices, :]  # only plot the first 32 particles
    # pY = data_xarray['lat'].data[indices, :]  # only plot the first 32 particles

    fX = data_xarray['lon'].data  # to be loaded from pfile
    fY = data_xarray['lat'].data  # to be loaded from pfile
    fZ = None
    sample3D = False
    if 'depth' in data_xarray.keys():
        fZ = data_xarray['depth'].data  # to be loaded from pfile
        sample3D = True
    elif 'z' in data_xarray.keys():
        fZ = data_xarray['z'].data  # to be loaded from pfile
        sample3D = True
    fT = dtime_array
    fT = convert_timearray(fT, outdt_minutes*60, ns_per_sec, debug=DBG_MSG, array_name="fT")
    fA = data_xarray['age'].data  # to be loaded from pfile | age
    fAd = data_xarray['age_d'].data  # to be loaded from pfile | age_d
    if DBG_MSG:
        print("original fA.range and fA.dtype (before conversion): ({}, {}) \t {}".format(fA[0].min(), fA[0].max(), fA.dtype))
    if not isinstance(fA[0,0], np.float32) and not isinstance(fA[0,0], np.float64):
        fA = fA - timebase
    fA = convert_timearray(fA, outdt_minutes*60, ns_per_sec, debug=DBG_MSG, array_name="fA")

    # pcounts = np.zeros((fT.shape[1], xval.shape[0], yval.shape[0]), dtype=np.int32)
    pcounts = None
    if sample3D:
        pcounts = np.zeros((zsteps, ysteps, xsteps), dtype=np.int32)
    else:
        pcounts = np.zeros((ysteps, xsteps), dtype=np.int32)
    pcounts_minmax = [0., 0.]
    pcounts_statistics = [0., 0.]
    pcounts_file = h5py.File(os.path.join(odir, "particlecount.h5"), "w")
    pcounts_file_ds = None
    if sample3D:
        pcounts_file_ds = pcounts_file.create_dataset("pcount",
                                                      shape=(1, pcounts.shape[0], pcounts.shape[1], pcounts.shape[2]),
                                                      dtype=pcounts.dtype,
                                                      maxshape=(fT.shape[1], pcounts.shape[0], pcounts.shape[1], pcounts.shape[2]),
                                                      compression="gzip", compression_opts=4)
    else:
        pcounts_file_ds = pcounts_file.create_dataset("pcount", shape=(1, pcounts.shape[0], pcounts.shape[1]), dtype=pcounts.dtype, maxshape=(fT.shape[1], pcounts.shape[0], pcounts.shape[1]), compression="gzip", compression_opts=4)
    pcounts_file_ds.attrs['unit'] = "count (scalar)"
    pcounts_file_ds.attrs['name'] = 'particle_count'

    # density = np.zeros((fT.shape[1], xval.shape[0], yval.shape[0]), dtype=np.float32)
    density = None
    if sample3D:
        density = np.zeros((zsteps, ysteps, xsteps), dtype=np.float32)
    else:
        density = np.zeros((ysteps, xsteps), dtype=np.float32)
    density_minmax = [0., 0.]
    density_statistics = [0., 0.]
    density_file = h5py.File(os.path.join(odir, "density.h5"), "w")
    density_file_ds = None
    if sample3D:
        density_file_ds = density_file.create_dataset("density",
                                                      shape=(1, density.shape[0], density.shape[1], density.shape[2]),
                                                      dtype=density.dtype,
                                                      maxshape=(fT.shape[1], density.shape[0], density.shape[1], density.shape[2]),
                                                      compression="gzip", compression_opts=4)
    else:
        density_file_ds = density_file.create_dataset("density", shape=(1, density.shape[0], density.shape[1]), dtype=density.dtype, maxshape=(fT.shape[1], density.shape[0], density.shape[1]), compression="gzip", compression_opts=4)
    density_file_ds.attrs['unit'] = "pts / arc_deg^2"
    density_file_ds.attrs['name'] = 'density'

    # rel_density = np.zeros((fT.shape[1], xval.shape[0], yval.shape[0]), dtype=np.float32)
    rel_density = None
    if sample3D:
        rel_density = np.zeros((zsteps, ysteps, xsteps), dtype=np.float32)
    else:
        rel_density = np.zeros((ysteps, xsteps), dtype=np.float32)
    rel_density_minmax = [0., 0.]
    rel_density_statistics = [0., 0.]
    rel_density_file = h5py.File(os.path.join(odir, "rel_density.h5"), "w")
    rel_density_file_ds = None
    if sample3D:
        rel_density_file_ds = rel_density_file.create_dataset("rel_density",
                                                              shape=(1, rel_density.shape[0], rel_density.shape[1], rel_density.shape[2]),
                                                              dtype=rel_density.dtype,
                                                              maxshape=(fT.shape[1], rel_density.shape[0], rel_density.shape[1], rel_density.shape[2]),
                                                              compression="gzip", compression_opts=4)
    else:
        rel_density_file_ds = rel_density_file.create_dataset("rel_density", shape=(1, rel_density.shape[0], rel_density.shape[1]), dtype=rel_density.dtype, maxshape=(fT.shape[1], rel_density.shape[0], rel_density.shape[1]), compression="gzip", compression_opts=4)
    rel_density_file_ds.attrs['unit'] = "pts_percentage / arc_deg^2"
    rel_density_file_ds.attrs['name'] = 'relative_density'

    # lifetime = np.zeros((fT.shape[1], xval.shape[0], yval.shape[0]), dtype=np.float32)
    lifetime = None
    if sample3D:
        lifetime = np.zeros((zsteps, ysteps, xsteps), dtype=np.float32)
    else:
        lifetime = np.zeros((ysteps, xsteps), dtype=np.float32)
    lifetime_minmax = [0., 0.]
    lifetime_statistics = [0., 0.]
    lifetime_file = h5py.File(os.path.join(odir, "lifetime.h5"), "w")
    lifetime_file_ds = None
    if sample3D:
        lifetime_file_ds = lifetime_file.create_dataset("lifetime",
                                                        shape=(1, lifetime.shape[0], lifetime.shape[1], lifetime.shape[2]),
                                                        dtype=lifetime.dtype,
                                                        maxshape=(fT.shape[1], lifetime.shape[0], lifetime.shape[1], lifetime.shape[2]),
                                                        compression="gzip", compression_opts=4)
    else:
        lifetime_file_ds = lifetime_file.create_dataset("lifetime", shape=(1, lifetime.shape[0], lifetime.shape[1]), dtype=lifetime.dtype, maxshape=(fT.shape[1], lifetime.shape[0], lifetime.shape[1]), compression="gzip", compression_opts=4)
    lifetime_file_ds.attrs['unit'] = "avg. lifetime"
    lifetime_file_ds.attrs['name'] = 'lifetime'

    A = float(gres**2)
    total_items = fT.shape[0] * fT.shape[1]
    for ti in range(fT.shape[1]):
        if sample3D:
            pcounts[:, :, :] = 0
            density[:, :, :] = 0
            rel_density[:, :, :] = 0
            lifetime[:, :, :] = 0
            tlifetime = np.zeros((zsteps, ysteps, xsteps), dtype=np.float32)
        else:
            pcounts[:, :] = 0
            density[:, :] = 0
            rel_density[:, :] = 0
            lifetime[:, :] = 0
            tlifetime = np.zeros((ysteps, xsteps), dtype=np.float32)

        x_in = np.array(fX[:, ti])
        nonnan_x = ~np.isnan(x_in)
        y_in = np.array(fY[:, ti])
        nonnan_y = ~np.isnan(y_in)
        z_in, nonnan_z = (None, None)
        if sample3D:
            z_in = np.array(fZ[:, ti])
            nonnan_z = ~np.isnan(z_in)
        xpts, ypts, zpts = (None, None, None)
        if sample3D:
            x_in = x_in[np.logical_and(np.logical_and(nonnan_x, nonnan_y), nonnan_z)]
            y_in = y_in[np.logical_and(np.logical_and(nonnan_x, nonnan_y), nonnan_z)]
            z_in = z_in[np.logical_and(np.logical_and(nonnan_x, nonnan_y), nonnan_z)]
            xpts = (np.floor(x_in+(a/2.0))*gres).astype(np.int32).flatten()
            ypts = (np.floor(y_in+(b/2.0))*gres).astype(np.int32).flatten()
            zpts = (np.floor(z_in / gres)).astype(np.int32).flatten()
            assert xpts.shape[0] == ypts.shape[0], "Dimensions of xpts (={}) does not match ypts(={}).".format(xpts.shape[0], ypts.shape[0])
            assert xpts.shape[0] == zpts.shape[0], "Dimensions of xpts (={}) does not match zpts(={}).".format(xpts.shape[0], zpts.shape[0])
            xcondition = np.logical_and((xpts >= 0), (xpts < (xsteps - 1)))
            ycondition = np.logical_and((ypts >= 0), (ypts < (ysteps - 1)))
            zcondition = np.logical_and((zpts >= 0), (zpts < (zsteps - 1)))
            xpts = xpts[np.logical_and(np.logical_and(xcondition, ycondition), zcondition)]
            ypts = ypts[np.logical_and(np.logical_and(xcondition, ycondition), zcondition)]
            zpts = zpts[np.logical_and(np.logical_and(xcondition, ycondition), zcondition)]
        else:
            x_in = x_in[np.logical_and(nonnan_x, nonnan_y)]
            y_in = y_in[np.logical_and(nonnan_x, nonnan_y)]
            xpts = (np.floor(x_in+(a/2.0))*gres).astype(np.int32).flatten()
            ypts = (np.floor(y_in+(b/2.0))*gres).astype(np.int32).flatten()
            assert xpts.shape[0] == ypts.shape[0], "Dimensions of xpts (={}) does not match ypts(={}).".format(xpts.shape[0], ypts.shape[0])
            xcondition = np.logical_and((xpts >= 0), (xpts < (xsteps - 1)))
            ycondition = np.logical_and((ypts >= 0), (ypts < (ysteps - 1)))
            xpts = xpts[np.logical_and(xcondition, ycondition)]
            ypts = ypts[np.logical_and(xcondition, ycondition)]

        # xpts = np.floor((fX[:, ti]+(a/2.0))*gres).astype(np.int32).flatten()
        # xpts = np.maximum(np.minimum(xpts, xsteps - 1), 0)
        # ypts = np.floor((fY[:, ti]+(b/2.0))*gres).astype(np.int32).flatten()
        # ypts = np.maximum(np.minimum(ypts, ysteps - 1), 0)
        if ti == 0:
            print("xpts: {}".format(xpts))
            print("ypts: {}".format(ypts))
            if sample3D:
                print("zpts: {}".format(zpts))
        for pi in range(xpts.shape[0]):
            try:
                if sample3D:
                    pcounts[zpts[pi], ypts[pi], xpts[pi]] += 1
                    tlifetime[zpts[pi], ypts[pi], xpts[pi]] += fA[pi, ti]
                else:
                    pcounts[ypts[pi], xpts[pi]] += 1
                    tlifetime[ypts[pi], xpts[pi]] += fA[pi, ti]
            except (IndexError, ) as error_msg:
                # we don't abort here cause the brownian-motion wiggle of AvectionRK4EulerMarujama always edges on machine precision, which can np.floor(..) make go over-size
                if sample3D:
                    print("\nError trying to index point ({}, {}, {}) with indices ({}, {}, {})".format(fX[pi, ti], fY[pi, ti], fZ[pi, ti],
                                                                                                xpts[pi], ypts[pi], zpts[pi]))
                else:
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
        if sample3D:
            pcounts_file_ds[ti, :, :, :] = pcounts
        else:
            pcounts_file_ds[ti, :, :] = pcounts

        density_minmax = [min(density_minmax[0], density.min()), max(density_minmax[1], density.max())]
        density_statistics[0] += density.mean()
        density_statistics[1] += density.std()
        density_file_ds.resize((ti+1), axis=0)
        if sample3D:
            density_file_ds[ti, :, :, :] = density
        else:
            density_file_ds[ti, :, :] = density

        rel_density_minmax = [min(rel_density_minmax[0], rel_density.min()), max(rel_density_minmax[1], rel_density.max())]
        rel_density_statistics[0] += rel_density.mean()
        rel_density_statistics[1] += rel_density.std()
        rel_density_file_ds.resize((ti+1), axis=0)
        if sample3D:
            rel_density_file_ds[ti, :, :, :] = rel_density
        else:
            rel_density_file_ds[ti, :, :] = rel_density

        lifetime_minmax = [min(lifetime_minmax[0], lifetime.min()), max(lifetime_minmax[1], lifetime.max())]
        lifetime_statistics[0] += lifetime.mean()
        lifetime_statistics[1] += lifetime.std()
        lifetime_file_ds.resize((ti+1), axis=0)
        if sample3D:
            lifetime_file_ds[ti, :, :, :] = lifetime
        else:
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

    n_valid_pts = fX.shape[0]
    print("Number valid particles: {} (of total {})".format(n_valid_pts, fX.shape[0]))
    particle_file = h5py.File(os.path.join(odir, "particles.h5"), "w")
    px_ds = particle_file.create_dataset("p_x", data=fX, compression="gzip", compression_opts=4)
    px_ds.attrs['unit'] = "arc degree"
    px_ds.attrs['name'] = 'longitude'
    px_ds.attrs['min'] = fX.min()
    px_ds.attrs['max'] = fX.max()
    py_ds = particle_file.create_dataset("p_y", data=fY, compression="gzip", compression_opts=4)
    py_ds.attrs['unit'] = "arc degree"
    py_ds.attrs['name'] = 'latitude'
    py_ds.attrs['min'] = fY.min()
    py_ds.attrs['max'] = fY.max()
    pz_ds = None
    if fZ is not None:
        pz_ds = particle_file.create_dataset("p_z", data=fZ, compression="gzip", compression_opts=4)
        pz_ds.attrs['unit'] = "metres"
        pz_ds.attrs['name'] = 'depth'
        pz_ds.attrs['min'] = fZ.min()
        pz_ds.attrs['max'] = fZ.max()
    pt_ds = particle_file.create_dataset("p_t", data=fT, compression="gzip", compression_opts=4)
    pt_ds.attrs['unit'] = "seconds"
    pt_ds.attrs['name'] = 'time'
    pt_ds.attrs['min'] = fT.min()
    pt_ds.attrs['max'] = fT.max()
    # page_ds = particle_file.create_dataset("p_age", data=fA, compression="gzip", compression_opts=4)
    # page_ds.attrs['unit'] = "seconds"
    # page_ds.attrs['name'] = 'age'
    # page_ds.attrs['min'] = fA.min()
    # page_ds.attrs['max'] = fA.max()
    page_ds = particle_file.create_dataset("p_age_d", data=fAd, compression="gzip", compression_opts=4)
    page_ds.attrs['unit'] = "days"
    page_ds.attrs['name'] = 'age'
    page_ds.attrs['min'] = fAd.min()
    page_ds.attrs['max'] = fAd.max()
    particle_file.close()

    del rel_density
    del lifetime
    del density
    del pcounts
    del fX
    del fY
    if fZ is not None:
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
    zval = None
    if sample3D:
        zval = np.arange(start=0.0, stop=c, step=zstep, dtype=np.float32)
    centers_x = xval + step/2.0
    centers_y = yval + step/2.0
    centers_z = None
    if sample3D:
        centers_z = zval + zstep/2.0
    us, vs, ws = (None, None, None)
    if sample3D:
        us = np.zeros((centers_z.shape[0], centers_y.shape[0], centers_x.shape[0]))
        vs = np.zeros((centers_z.shape[0], centers_y.shape[0], centers_x.shape[0]))
        if use_W:
            ws = np.zeros((centers_z.shape[0], centers_y.shape[0], centers_x.shape[0]))
    else:
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
    grid_dep_ds = None
    if sample3D:
        grid_dep_ds = grid_file.create_dataset("depth", data=centers_z, compression="gzip", compression_opts=4)
        grid_dep_ds.attrs['unit'] = "metres"
        grid_dep_ds.attrs['name'] = 'depth'
        grid_dep_ds.attrs['min'] = centers_z.min()
        grid_dep_ds.attrs['max'] = centers_z.max()
    grid_time_ds = grid_file.create_dataset("times", data=fT[0], compression="gzip", compression_opts=4)
    grid_time_ds.attrs['unit'] = "seconds"
    grid_time_ds.attrs['name'] = 'time'
    grid_time_ds.attrs['min'] = fT[0].min()
    grid_time_ds.attrs['max'] = fT[0].max()
    grid_file.close()

    us_minmax = [0., 0.]
    us_statistics = [0., 0.]
    us_file = h5py.File(os.path.join(odir, "hydrodynamic_U.h5"), "w")
    if sample3D:
        us_file_ds = us_file.create_dataset("uo",
                                            shape=(1, us.shape[0], us.shape[1], us.shape[2]), dtype=us.dtype,
                                            maxshape=(fT.shape[1], us.shape[0], us.shape[1], us.shape[2]),
                                            compression="gzip",
                                            compression_opts=4)
    else:
        us_file_ds = us_file.create_dataset("uo", shape=(1, us.shape[0], us.shape[1]), dtype=us.dtype, maxshape=(fT.shape[1], us.shape[0], us.shape[1]), compression="gzip", compression_opts=4)
    us_file_ds.attrs['unit'] = "m/s"
    us_file_ds.attrs['name'] = 'meridional_velocity'

    vs_minmax = [0., 0.]
    vs_statistics = [0., 0.]
    vs_file = h5py.File(os.path.join(odir, "hydrodynamic_V.h5"), "w")
    if sample3D:
        vs_file_ds = vs_file.create_dataset("vo", shape=(1, vs.shape[0], vs.shape[1], vs.shape[2]), dtype=vs.dtype,
                                            maxshape=(fT.shape[1], vs.shape[0], vs.shape[1], vs.shape[2]),
                                            compression="gzip",
                                            compression_opts=4)
    else:
        vs_file_ds = vs_file.create_dataset("vo", shape=(1, vs.shape[0], vs.shape[1]), dtype=vs.dtype, maxshape=(fT.shape[1], vs.shape[0], vs.shape[1]), compression="gzip", compression_opts=4)
    vs_file_ds.attrs['unit'] = "m/s"
    vs_file_ds.attrs['name'] = 'zonal_velocity'

    ws_minmax, ws_statistics, ws_file, ws_file_ds = (None, None, None, None)
    if sample3D and use_W:
        ws_minmax = [0., 0.]
        ws_statistics = [0., 0.]
        ws_file = h5py.File(os.path.join(odir, "hydrodynamic_W.h5"), "w")
        ws_file_ds = ws_file.create_dataset("wo", shape=(1, ws.shape[0], ws.shape[1], ws.shape[2]), dtype=ws.dtype, maxshape=(fT.shape[1], ws.shape[0], ws.shape[1], ws.shape[2]), compression="gzip", compression_opts=4)
        ws_file_ds.attrs['unit'] = "m/s"
        ws_file_ds.attrs['name'] = 'vertical_velocity'

    print("Preparing field datatypes ...")
    if not ( isinstance(U, da.core.Array) or isinstance(U, np.ndarray) ):
        del U
        del V
        if W is not None:
            del W
        Upath = field_fpath + "U.nc"
        Ufile = xr.open_dataset(Upath, decode_cf=True, engine='netcdf4', chunks='auto')
        U = Ufile['vozocrtx'].data
        if DBG_MSG:
            print("U - shape: {}".format(U.shape))
        Vpath = field_fpath + "V.nc"
        Vfile = xr.open_dataset(Vpath, decode_cf=True, engine='netcdf4', chunks='auto')
        V = Vfile['vomecrty'].data
        if DBG_MSG:
            print("V - shape: {}".format(V.shape))
        if use_W:
            Wpath = field_fpath + "W.nc"
            Wfile = xr.open_dataset(Wpath, decode_cf=True, engine='netcdf4', chunks='auto')
            W = Wfile['W'].data
            if DBG_MSG:
                print("W - shape: {}".format(W.shape))
    if not isinstance(fdepths, np.ndarray):
        fdepths = np.array(fdepths)
    if not isinstance(flats, np.ndarray):
        flats = np.array(flats)
    if not isinstance(flons, np.ndarray):
        flons = np.array(flons)

    print("Interpolating UVW on a regular-square grid ...")
    total_items = fT.shape[1]
    for ti in range(fT.shape[1]):
        uvw_ti = ti
        if periodicFlag:
            uvw_ti = ti % U.shape[0]
        else:
            uvw_ti = min(ti, U.shape[0]-1)
        p_center_z = None
        if sample3D:
            fU = np.array(U[uvw_ti])
            if DBG_MSG:
                print("fU - shape: {}".format(fU.shape))
            fV = np.array(V[uvw_ti])
            if DBG_MSG:
                print("fV - shape: {}".format(fV.shape))
            fW = np.array(W[uvw_ti])
            if DBG_MSG:
                print("fW - shape: {}".format(fW.shape))
            us[:, :, :] = 0
            vs[:, :, :] = 0
            if use_W:
                ws[:, :, :] = 0
            mgrid = (fdepths, flats, flons)
            if DBG_MSG:
                print("input meshgrid - shape: ({}, {}, {})".format(mgrid[0].shape[0], mgrid[1].shape[0], mgrid[2].shape[0]))
            p_center_z, p_center_y, p_center_x = np.meshgrid(centers_z, centers_y, centers_x, sparse=False, indexing='ij', copy=False)
            gcenters = (p_center_z.flatten(), p_center_y.flatten(), p_center_x.flatten())
            if DBG_MSG:
                print("interpolate points - shape: ({}, {}, {})".format(gcenters[0].shape[0], gcenters[1].shape[0], gcenters[2].shape[0]))
            # print("gcenters dims = ({}, {}, {})".format(gcenters[0].shape, gcenters[1].shape, gcenters[2].shape))
            # print("mgrid dims = ({}, {}, {})".format(mgrid[0].shape, mgrid[1].shape, mgrid[2].shape))
            # print("u dims = ({}, {}, {})".format(U.shape[0], U.shape[1], U.shape[2]))
            # print("v dims = ({}, {}, {})".format(V.shape[0], V.shape[1], V.shape[2]))
            us_local = interpn(mgrid, fU, gcenters, method='linear', fill_value=.0)
            vs_local = interpn(mgrid, fV, gcenters, method='linear', fill_value=.0)
            ws_local = None
            if use_W:
                ws_local = interpn(mgrid, fW, gcenters, method='linear', fill_value=.0)
            # print("us_local dims: {}".format(us_local.shape))
            # print("vs_local dims: {}".format(vs_local.shape))

            # us_local = np.reshape(us_local, p_center_y.shape)
            # vs_local = np.reshape(vs_local, p_center_y.shape)
            # print("us_local dims: {}".format(us_local.shape))
            # print("vs_local dims: {}".format(vs_local.shape))
            us[:, :, :] = np.reshape(us_local, p_center_x.shape)
            vs[:, :, :] = np.reshape(vs_local, p_center_y.shape)
            if use_W:
                ws[:, :, :] = np.reshape(ws_local, p_center_z.shape)
            del fU
            fU = None
            del fV
            fV = None
            del fW
            fW = None
        else:
            us[:, :] = 0
            vs[:, :] = 0
            mgrid = (flats, flons)
            p_center_y, p_center_x = np.meshgrid(centers_y, centers_x, sparse=False, indexing='ij')
            gcenters = (p_center_y.flatten(), p_center_x.flatten())
            # print("gcenters dims = ({}, {}, {})".format(gcenters[0].shape, gcenters[1].shape, gcenters[2].shape))
            # print("mgrid dims = ({}, {}, {})".format(mgrid[0].shape, mgrid[1].shape, mgrid[2].shape))
            # print("u dims = ({}, {}, {})".format(U.shape[0], U.shape[1], U.shape[2]))
            # print("v dims = ({}, {}, {})".format(V.shape[0], V.shape[1], V.shape[2]))
            us_local = interpn(mgrid, U[uvw_ti], gcenters, method='linear', fill_value=.0)
            vs_local = interpn(mgrid, V[uvw_ti], gcenters, method='linear', fill_value=.0)
            # print("us_local dims: {}".format(us_local.shape))
            # print("vs_local dims: {}".format(vs_local.shape))

            # us_local = np.reshape(us_local, p_center_y.shape)
            # vs_local = np.reshape(vs_local, p_center_y.shape)
            # print("us_local dims: {}".format(us_local.shape))
            # print("vs_local dims: {}".format(vs_local.shape))
            us[:, :] = np.reshape(us_local, p_center_x.shape)
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
        if sample3D and use_W:
            ws_minmax = [min(ws_minmax[0], ws.min()), max(ws_minmax[1], ws.max())]
            ws_statistics[0] += ws.mean()
            ws_statistics[1] += ws.std()
            ws_file_ds.resize((ti+1), axis=0)
            ws_file_ds[ti, :, :] = ws

        del us_local
        del vs_local
        if sample3D and use_W:
            del ws_local
        # del gcenters
        if sample3D:
            del p_center_z
        del p_center_y
        del p_center_x
        # del mgrid
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
    if sample3D and use_W:
        ws_file_ds.attrs['min'] = ws_minmax[0]
        ws_file_ds.attrs['max'] = ws_minmax[1]
        ws_file_ds.attrs['mean'] = ws_statistics[0] / float(fT.shape[1])
        ws_file_ds.attrs['std'] = ws_statistics[1] / float(fT.shape[1])
        ws_file.close()

    # del gcenters
    del centers_x
    del centers_y
    if sample3D:
        del centers_z
    del xval
    del yval
    if sample3D:
        del zval
    # del xval_start
    # del yval_start
    del fT
    del dtime_array
    del ctime_array
