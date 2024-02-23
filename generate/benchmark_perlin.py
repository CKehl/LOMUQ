"""
Author: Dr. Christian Kehl
Date: 11-02-2020
"""

from parcels import AdvectionEE, AdvectionRK45, AdvectionRK4, AdvectionRK4_3D  # noqa
from parcels import FieldSet, ScipyParticle, JITParticle, Variable, StateCode, OperationCode, ErrorCode  # noqa
from parcels import ParticleSetSOA, ParticleSetAOS
from parcels import ParticleFileSOA, ParticleFileAOS
from parcels import KernelSOA, KernelAOS
from parcels.field import VectorField, NestedField, SummedField  # Field,
# from parcels import plotTrajectoriesFile_loadedField
# from parcels import rng as random
from parcels import ParcelsRandom
from datetime import timedelta as delta
import time as ostime
from glob import glob
import math
from argparse import ArgumentParser
import datetime
import numpy as np
import xarray as xr
import fnmatch
import sys
import gc
import os
import string
from parcels.tools import perlin3d
from parcels.tools import perlin2d

from scipy.interpolate import interpn
from scipy import ndimage

from PIL import Image
import matplotlib.pyplot as plt

try:
    from mpi4py import MPI
except:
    MPI = None
with_GC = False

# pset = None
pset_modes = ['soa', 'aos', 'nodes']
ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
pset_types = {'soa': {'pset': ParticleSetSOA, 'pfile': ParticleFileSOA, 'kernel': KernelSOA},
              'aos': {'pset': ParticleSetAOS, 'pfile': ParticleFileAOS, 'kernel': KernelAOS}}
method = {'RK4': AdvectionRK4, 'EE': AdvectionEE, 'RK45': AdvectionRK45}
global_t_0 = 0
fieldset = None
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

noctaves=4
perlinres=(1,20,10,1)  # (1,32,8)
shapescale=(4,3,3,4)  # (4,8,8)
shapescale3D=(4,1,1,2)  # (4,8,8)
#shapescale=(8,6,6) # formerly
perlin_persistence=0.65
img_shape = (int(math.pow(2,noctaves))*perlinres[1]*shapescale[1],
             int(math.pow(2,noctaves))*perlinres[2]*shapescale[2])
vol_shape = (int(math.pow(2,noctaves))*perlinres[1]*shapescale3D[1],
             int(math.pow(2,noctaves))*perlinres[2]*shapescale3D[2],
             int(math.pow(2,noctaves))*perlinres[3]*shapescale3D[3])
# sx = img_shape[0]/1000.0
# sy = img_shape[1]/1000.0
# sz = img_shape[2]/1000.0

# a = (100.0 * vol_shape[0])
# b = (100.0 * vol_shape[1])
# c = (10.0 * vol_shape[2])

tsteps = 61
tscale = 6
# scalefac = (40.0 / (1000.0/ (60.0 * 60.0)))  # 40 km/h
scalefac = ((4.0*1000) / (60.0*60.0))  # 4 km/h
vertical_scale = ( 800.0 / (24*60.0*60.0) )  # 800 m/d

nwaves_x = 2
nwaves_y = 2
perlin_power_x = 0.55
perlin_power_y = 0.45

# Idea for 4D: perlin3D creates a time-consistent 3D field
# Thus, we can use skimage to create shifted/rotated/morphed versions
# for the depth domain, so that perlin4D = [depth][time][lat][lon].
# then, we can do a transpose-op in numpy, to get [time][depth][lat][lon]

# we need to modify the kernel.execute / pset.execute so that it returns from the JIT
# in a given time WITHOUT writing to disk via outfie => introduce a pyloop_dt


def DeleteParticle(particle, fieldset, time):
    particle.delete()


def RenewParticle(particle, fieldset, time):
    EA = fieldset.east_lim
    WE = fieldset.west_lim
    NO = fieldset.north_lim
    SO = fieldset.south_lim

    particle.lon = WE + ParcelsRandom.random() * (EA - WE)
    particle.lat = SO + ParcelsRandom.random() * (NO - SO)
    if fieldset.isThreeD > 0.0:
        TO = fieldset.top
        BO = fieldset.bottom
        particle.depth = TO + ((BO-TO) / 0.75) + ((BO-TO) * -0.5 * ParcelsRandom.random())
    particle.delete()


def periodicBC(particle, fieldSet, time):
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


def perIterGC():
    gc.collect()


def reflect_top_bottom(particle, fieldset, time):
    span = fieldset.bottom - fieldset.top
    while particle.depth <= fieldset.top:
        particle.depth += 0.01 * span
    while particle.depth >= fieldset.bottom:
        particle.depth -= 0.01 * span


def wrap_top_bottom(particle, fieldset, time):
    span = fieldset.bottom - fieldset.top
    if particle.depth <= fieldset.top:
        particle.depth += span
    if particle.depth >= fieldset.bottom:
        particle.depth -= span


def constant_top_bottom(particle, fieldset, time):
    if particle.depth < fieldset.top:
        particle.depth = fieldset.top
    if particle.depth > fieldset.bottom:
        particle.depth = fieldset.bottom


def perlin_waves(periodic_wrap=False, write_out=False):
    U_operator = np.array([[-0.3, -0.6, -1.0, 0.0, 1.0, 0.6, 0.3],
                           [-0.3, -0.6, -1.0, 0.0, 1.0, 0.6, 0.3],
                           [-0.3, -0.6, -1.0, 0.0, 1.0, 0.6, 0.3],
                           [-0.3, -0.6, -1.0, 0.0, 1.0, 0.6, 0.3],
                           [-0.3, -0.6, -1.0, 0.0, 1.0, 0.6, 0.3],
                           [-0.3, -0.6, -1.0, 0.0, 1.0, 0.6, 0.3],
                           [-0.3, -0.6, -1.0, 0.0, 1.0, 0.6, 0.3]], dtype=np.float32)
    U_OP_2D = np.stack((U_operator*0.5, U_operator*1.0, U_operator*0.5), axis=2)
    U_OP_2D = np.transpose(U_OP_2D, [2,0,1])
    V_operator = np.array([[-0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3],
                           [-0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6],
                           [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [+1.0, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0],
                           [+0.6, +0.6, +0.6, +0.6, +0.6, +0.6, +0.6],
                           [+0.3, +0.3, +0.3, +0.3, +0.3, +0.3, +0.3]], dtype=np.float32)
    V_OP_2D = np.stack((V_operator*0.5, V_operator*1.0, V_operator*0.5), axis=2)
    V_OP_2D = np.transpose(V_OP_2D, [2,0,1])

    a = (100.0 * img_shape[0])
    b = (100.0 * img_shape[1])

    lon = np.linspace(-a*0.5, a*0.5, img_shape[0], dtype=np.float32)
    sys.stdout.write("lon field: {}\n".format(lon.size))
    lat = np.linspace(-b*0.5, b*0.5, img_shape[1], dtype=np.float32)
    sys.stdout.write("lat field: {}\n".format(lat.size))
    totime = tsteps*tscale*24.0*60.0*60.0
    time = np.linspace(0., totime, tsteps, dtype=np.float64)
    sys.stdout.write("time field: {}\n".format(time.size))

    velmag = perlin2d.generate_fractal_noise_temporal2d(img_shape, tsteps, (perlinres[1], perlinres[2]), noctaves, perlin_persistence, max_shift=((-1, 2), (-1, 2)))
    abs_velmag = np.linalg.norm(velmag)
    # velmag = velmag / abs_velmag

    pts = (time, lon, lat)
    wave = np.empty(velmag.shape, dtype=velmag.dtype)
    total_items = velmag.shape[0] * velmag.shape[1] * velmag.shape[2]
    workdone = 0
    ti = 0
    for t in range(velmag.shape[0]):
        wave_x = np.empty((velmag.shape[1], velmag.shape[2]), dtype=velmag.dtype)
        wave_y = np.empty((velmag.shape[1], velmag.shape[2]), dtype=velmag.dtype)
        # pts_x = np.stack((np.stack((np.ones(velmag.shape[2], dtype=velmag.dtype)*time[t], np.ones(velmag.shape[2], dtype=velmag.dtype)*lon[12]), axis=1), lat / 5.0), axis=1)
        # pts_y = np.stack((np.stack((np.ones(velmag.shape[1], dtype=velmag.dtype)*time[t], lon / 5.0), axis=1), np.ones(velmag.shape[1], dtype=velmag.dtype)*lat[12]), axis=1)
        pts_x = np.stack((np.ones(velmag.shape[2], dtype=velmag.dtype)*time[t], np.ones(velmag.shape[2], dtype=velmag.dtype)*lon[12], lat / 18.0), axis=1)
        pts_y = np.stack((np.ones(velmag.shape[1], dtype=velmag.dtype)*time[t], lon / 18.0, np.ones(velmag.shape[1], dtype=velmag.dtype)*lat[12]), axis=1)
        pValues_x = interpn(pts, velmag, pts_x)
        pValues_y = interpn(pts, velmag, pts_y)
        for i in range(velmag.shape[1]):
            for j in range(velmag.shape[2]):
                # sinx_coord = (math.pi * (i / velmag.shape[1]) * (nwaves_x*2.0))  #  / (img_shape[0])
                # siny_coord = (math.pi * (j / velmag.shape[2]) * (nwaves_y*2.0))  #  / (img_shape[1])

                # cosx_coord = (math.pi * (i / velmag.shape[1]) * (nwaves_x*2.0))  # / (img_shape[0])
                # cosy_coord = (math.pi * (j / velmag.shape[2]) * (nwaves_y*2.0))  #  / (img_shape[1])

                # cosx_coord = (i / velmag.shape[1]) + (perlin_power_x / 5.0) * velmag[t, 12, j]  #
                # cosy_coord = (j / velmag.shape[2]) + (perlin_power_y / 5.0) * velmag[t, i, 12]  #
                cosx_coord = (i / velmag.shape[1]) + (perlin_power_x / 1.0) * pValues_x[j]
                cosy_coord = (j / velmag.shape[2]) + (perlin_power_y / 1.0) * pValues_y[i]

                # wave_x = math.sin(sinx_coord + perlin_power_x * velmag[t, i, j])
                # wave_y = math.sin(siny_coord + perlin_power_y * velmag[t, i, j])

                # wave_x = math.cos(cosx_coord + perlin_power_x * velmag[t, i, j])
                # wave_y = math.cos(cosy_coord + perlin_power_y * velmag[t, i, j])

                wave_x[i, j] = math.sin((cosx_coord) * ((math.pi * (nwaves_x * 2.0)) / 1.0))
                wave_y[i, j] = math.sin((cosy_coord) * ((math.pi * (nwaves_y * 2.0)) / 1.0))
                # wave_x = np.nan_to_num(np.float_power(math.sin((cosx_coord) * ((math.pi * (nwaves_x * 2.0)) / 1.0)), 0.3), nan=0.0)
                # wave_y = np.nan_to_num(np.float_power(math.sin((cosy_coord) * ((math.pi * (nwaves_y * 2.0)) / 1.0)), 0.3), nan=0.0)

                # wave_x = (wave_x + 1.0) / 2.0
                # wave_y = (wave_y + 1.0) / 2.0
                # wave[t, i, j] = (wave_x - 0.5) * 2.0
                # wave[t, i, j] = (wave_y - 0.5) * 2.0

                # wave[t, i, j] = ((wave_x + wave_y) / 2.0) * velmag[t, i, j]
                # # wave[t, i, j] = (np.minimum(np.fabs(wave_x + wave_y), 1.0)) * velmag[t, i, j]
                # # wave[t, i, j] = (np.minimum(wave_x + wave_y, 1.0))  # * velmag[t, i, j]
                # # wave[t, i, j] = (np.maximum(wave_x, wave_y))  # * velmag[t, i, j]

                current_item = ti
                workdone = current_item / total_items
                ti += 1
        print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)

        with np.errstate(invalid='ignore'):
            wave_x = np.nan_to_num(np.float_power(wave_x, 0.3), nan=0.0)
            wave_y = np.nan_to_num(np.float_power(wave_y, 0.3), nan=0.0)
        wave[t, :, :] = ((wave_x + wave_y) / 2.0) * (velmag[t, :, :] / abs_velmag)
    print("\nGenerated NetCDF U/V data.")
    # print("wave: {}".format(wave[0,32:48,32:48]))

    # U = wave
    # U = np.gradient(wave*scalefac, 7, edge_order=1, axis=1)
    U = ndimage.convolve(wave*scalefac, U_OP_2D, mode='wrap')
    U = np.transpose(U, (0,2,1))
    # V = wave
    # V = np.gradient(wave*scalefac, 7, edge_order=1, axis=2)
    V = ndimage.convolve(wave*scalefac, V_OP_2D, mode='wrap')
    V = np.transpose(V, (0,2,1))
    wave = np.transpose(wave, (0,2,1))

    # U *= scalefac
    # V *= scalefac

    data = {'U': U, 'V': V}
    dimensions = {'time': time, 'lon': lon, 'lat': lat}
    fieldset = None
    if periodic_wrap:
        fieldset = FieldSet.from_data(data, dimensions, mesh='flat', transpose=False, time_periodic=delta(days=366))
    else:
        fieldset = FieldSet.from_data(data, dimensions, mesh='flat', transpose=False, allow_time_extrapolation=True)
    if write_out:
        fieldset.write(filename=write_out)

        wave_img_path = os.path.join(os.path.dirname(write_out), 'wave_snapshot.png')
        U_img_path = os.path.join(os.path.dirname(write_out), 'U_snapshot.png')
        V_img_path = os.path.join(os.path.dirname(write_out), 'V_snapshot.png')

        plt.imsave(wave_img_path, wave[0], cmap='gray', dpi=300)
        # wave_img = Image.fromarray(wave[0], 'I')
        # wave_img.save(wave_img_path)
        plt.imsave(U_img_path, U[0], cmap='gray', dpi=300)
        # U_img = Image.fromarray(U[0], 'I')
        # U_img.save(U_img_path)
        plt.imsave(V_img_path, V[0], cmap='gray', dpi=300)
        # V_img = Image.fromarray(V[0], 'I')
        # V_img.save(V_img_path)
    return fieldset, a, b


def perlin_waves3D(periodic_wrap=False, write_out=False):
    U_operator = np.array([[-0.3, -0.6, -1.0, 0.0, 1.0, 0.6, 0.3],
                           [-0.3, -0.6, -1.0, 0.0, 1.0, 0.6, 0.3],
                           [-0.3, -0.6, -1.0, 0.0, 1.0, 0.6, 0.3],
                           [-0.3, -0.6, -1.0, 0.0, 1.0, 0.6, 0.3],
                           [-0.3, -0.6, -1.0, 0.0, 1.0, 0.6, 0.3],
                           [-0.3, -0.6, -1.0, 0.0, 1.0, 0.6, 0.3],
                           [-0.3, -0.6, -1.0, 0.0, 1.0, 0.6, 0.3]], dtype=np.float32)
    U_OP_2D = np.stack((U_operator, U_operator, U_operator,
                        U_operator,
                        U_operator, U_operator, U_operator), axis=2)
    U_OP_3D = np.stack((U_OP_2D*0.5, U_OP_2D*1.0, U_OP_2D*0.5), axis=3)
    U_OP_3D = np.transpose(U_OP_3D, [3, 0, 1, 2])
    V_operator = np.array([[-0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3],
                           [-0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6],
                           [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [+1.0, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0],
                           [+0.6, +0.6, +0.6, +0.6, +0.6, +0.6, +0.6],
                           [+0.3, +0.3, +0.3, +0.3, +0.3, +0.3, +0.3]], dtype=np.float32)
    V_OP_2D = np.stack((V_operator, V_operator, V_operator,
                        V_operator,
                        V_operator, V_operator, V_operator), axis=2)
    V_OP_3D = np.stack((V_OP_2D*0.5, V_OP_2D*1.0, V_OP_2D*0.5), axis=3)
    V_OP_3D = np.transpose(V_OP_3D, [3, 0, 1, 2])
    W_operator = np.ones((7, 7), dtype=np.float32)
    W_OP_2D = np.stack((W_operator*-0.3, W_operator*-0.6, W_operator*1.0,
                        np.zeros((7, 7), dtype=np.float32),
                        W_operator*1.0, W_operator*0.6, W_operator*0.3), axis=2)
    W_OP_3D = np.stack((W_OP_2D*0.5, W_OP_2D*1.0, W_OP_2D*0.5), axis=3)
    W_OP_3D = np.transpose(W_OP_3D, [3, 0, 1, 2])

    a = (100.0 * vol_shape[0])
    b = (100.0 * vol_shape[1])
    c = (100.0 * vol_shape[2])

    lon = np.linspace(-a*0.5, a*0.5, vol_shape[0], dtype=np.float32)
    sys.stdout.write("lon field: {}\n".format(lon.size))
    lat = np.linspace(-b*0.5, b*0.5, vol_shape[1], dtype=np.float32)
    sys.stdout.write("lat field: {}\n".format(lat.size))
    depth = np.linspace(0.0, c, vol_shape[2], dtype=np.float32)
    sys.stdout.write("depth field: {}\n".format(depth.size))
    totime = tsteps*tscale*24.0*60.0*60.0
    time = np.linspace(0., totime, tsteps, dtype=np.float64)
    sys.stdout.write("time field: {}\n".format(time.size))

    velmag = perlin3d.generate_fractal_noise_temporal3d(vol_shape, tsteps, (perlinres[1], perlinres[2], perlinres[3]), noctaves, perlin_persistence, max_shift=((-1, 2), (-1, 2), (-1, 2)))
    abs_velmag = np.linalg.norm(velmag)
    # velmag = velmag / abs_velmag

    pts = (time, lon, lat, depth)
    mix = int(velmag.shape[3]/4)
    depthgrad = np.concatenate([np.linspace(0, 1.0, mix, dtype=np.float32), np.linspace(1.0, 0.0, (velmag.shape[3]-mix-1), dtype=np.float32), np.zeros(1, dtype=np.float32)])
    wave = np.empty(velmag.shape, dtype=velmag.dtype)
    sys.stdout.write("velmag shape: {}\n".format(velmag.shape))
    total_items = velmag.shape[0] * velmag.shape[1] * velmag.shape[2] #* velmag.shape[3]
    workdone = 0
    ti = 0
    for t in range(velmag.shape[0]):
        wave_vol = np.empty((velmag.shape[1], velmag.shape[2], velmag.shape[3]), dtype=velmag.dtype)
        pts_x = np.stack((np.ones(velmag.shape[2], dtype=velmag.dtype)*time[t], np.ones(velmag.shape[2], dtype=velmag.dtype)*lon[12], lat / 18.0, np.ones(velmag.shape[2], dtype=velmag.dtype)*depth[1]), axis=1)
        pts_y = np.stack((np.ones(velmag.shape[1], dtype=velmag.dtype)*time[t], lon / 18.0, np.ones(velmag.shape[1], dtype=velmag.dtype)*lat[12], np.ones(velmag.shape[1], dtype=velmag.dtype)*depth[1]), axis=1)
        pValues_x = interpn(pts, velmag, pts_x)
        pValues_y = interpn(pts, velmag, pts_y)
        for i in range(velmag.shape[1]):
            for j in range(velmag.shape[2]):
                cosx_coord = (i / velmag.shape[1]) + (perlin_power_x / 1.0) * pValues_x[j]
                cosy_coord = (j / velmag.shape[2]) + (perlin_power_y / 1.0) * pValues_y[i]

                wave_vol[i, j, :] = (math.sin((cosx_coord) * ((math.pi * (nwaves_x * 2.0)) / 1.0)) + math.sin((cosy_coord) * ((math.pi * (nwaves_y * 2.0)) / 1.0))) / 2.0

                current_item = ti
                workdone = current_item / total_items
                ti += 1
        print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
        with np.errstate(invalid='ignore'):
            wave_vol = np.nan_to_num(np.float_power(wave_vol, 0.3), nan=0.0)
        for k in range(velmag.shape[3]):
            wave_vol[:, :, k] *= depthgrad[k]
        wave[t, :, :, :] = (wave_vol) * (velmag[t, :, :] / abs_velmag)
    print("\nGenerated NetCDF U/V/W data.")

    # U = np.gradient(wave*scalefac, 7, edge_order=1, axis=1)
    U = ndimage.convolve(wave*scalefac, U_OP_3D, mode='reflect')
    U = np.transpose(U, (0,3,2,1))
    # V = np.gradient(wave*scalefac, 7, edge_order=1, axis=2)
    V = ndimage.convolve(wave*scalefac, V_OP_3D, mode='reflect')
    V = np.transpose(V, (0,3,2,1))
    # W = np.gradient(wave*scalefac, 7, edge_order=1, axis=2)
    W = ndimage.convolve(wave*vertical_scale, W_OP_3D, mode='nearest')
    W = np.transpose(W, (0,3,2,1))
    ds = depth.size-1
    # W[:, ds, :, :] = -np.fabs(W[:, ds, :, :])
    W[:, 0, :, :] = np.fabs(W[:, 0, :, :])
    wave = np.transpose(wave, (0,3,2,1))

    data = {'U': U, 'V': V, 'W': W}
    dimensions = {'time': time, 'depth': depth, 'lon': lon, 'lat': lat}
    fieldset = None
    if periodic_wrap:
        fieldset = FieldSet.from_data(data, dimensions, mesh='flat', transpose=False, time_periodic=delta(days=366))
    else:
        fieldset = FieldSet.from_data(data, dimensions, mesh='flat', transpose=False, allow_time_extrapolation=True)
    fieldset.add_constant("top", 0.001)
    fieldset.add_constant("bottom", c)
    if write_out:
        fieldset.write(filename=write_out)

        wave_img_path = os.path.join(os.path.dirname(write_out), 'wave_snapshot.png')
        U_img_path = os.path.join(os.path.dirname(write_out), 'U_snapshot.png')
        V_img_path = os.path.join(os.path.dirname(write_out), 'V_snapshot.png')
        W_img_path = os.path.join(os.path.dirname(write_out), 'W_snapshot.png')

        plt.imsave(wave_img_path, wave[0, 1], cmap='gray', dpi=300)
        # wave_img = Image.fromarray(wave[0, 0], 'I')
        # wave_img.save(wave_img_path)
        plt.imsave(U_img_path, U[0, 1], cmap='gray', dpi=300)
        # U_img = Image.fromarray(U[0, 0], 'I')
        # U_img.save(U_img_path)
        plt.imsave(V_img_path, V[0, 1], cmap='gray', dpi=300)
        # V_img = Image.fromarray(V[0, 0], 'I')
        # V_img.save(V_img_path)
        plt.imsave(W_img_path, np.squeeze(W[0, :, 8, :]), cmap='gray', dpi=300)
        # W_img = Image.fromarray(np.squeeze(W[0, :, 8, :]), 'I')
        # W_img.save(W_img_path)
    return fieldset, a, b, c


def perlin_fieldset_from_numpy(periodic_wrap=False, write_out=False):
    """Simulate a current from structured random noise (i.e. Perlin noise).
    we use the external package 'perlin-numpy' as field generator, see:
    https://github.com/pvigier/perlin-numpy

    Perlin noise was introduced in the literature here:
    Perlin, Ken (July 1985). "An Image Synthesizer". SIGGRAPH Comput. Graph. 19 (97–8930), p. 287–296.
    doi:10.1145/325165.325247, https://dl.acm.org/doi/10.1145/325334.325247

    :param write_out: False if no write-out; else the fieldset path+basename
    """

    a = (100.0 * img_shape[0])
    b = (100.0 * img_shape[1])

    # Coordinates of the test fieldset (on A-grid in deg)
    lon = np.linspace(-a*0.5, a*0.5, img_shape[0], dtype=np.float32)
    sys.stdout.write("lon field: {}\n".format(lon.size))
    lat = np.linspace(-b*0.5, b*0.5, img_shape[1], dtype=np.float32)
    sys.stdout.write("lat field: {}\n".format(lat.size))
    totime = tsteps*tscale*24.0*60.0*60.0
    time = np.linspace(0., totime, tsteps, dtype=np.float64)
    sys.stdout.write("time field: {}\n".format(time.size))

    # Define arrays U (zonal), V (meridional)
    U = perlin2d.generate_fractal_noise_temporal2d(img_shape, tsteps, (perlinres[1], perlinres[2]), noctaves, perlin_persistence, max_shift=((-1, 2), (-1, 2)))
    U = np.transpose(U, (0,2,1))
    # U = np.swapaxes(U, 1, 2)
    # print("U-statistics - min: {:10.7f}; max: {:10.7f}; avg. {:10.7f}; std_dev: {:10.7f}".format(U.min(initial=0), U.max(initial=0), U.mean(), U.std()))
    V = perlin2d.generate_fractal_noise_temporal2d(img_shape, tsteps, (perlinres[1], perlinres[2]), noctaves, perlin_persistence, max_shift=((-1, 2), (-1, 2)))
    V = np.transpose(V, (0,2,1))
    # V = np.swapaxes(V, 1, 2)
    # print("V-statistics - min: {:10.7f}; max: {:10.7f}; avg. {:10.7f}; std_dev: {:10.7f}".format(V.min(initial=0), V.max(initial=0), V.mean(), V.std()))

    # U = perlin3d.generate_fractal_noise_3d(img_shape, perlinres, noctaves, perlin_persistence) * scalefac
    # U = np.transpose(U, (0,2,1))
    # sys.stdout.write("U field shape: {} - [tdim][ydim][xdim]=[{}][{}][{}]\n".format(U.shape, time.shape[0], lat.shape[0], lon.shape[0]))
    # V = perlin3d.generate_fractal_noise_3d(img_shape, perlinres, noctaves, perlin_persistence) * scalefac
    # V = np.transpose(V, (0,2,1))
    # sys.stdout.write("V field shape: {} - [tdim][ydim][xdim]=[{}][{}][{}]\n".format(V.shape, time.shape[0], lat.shape[0], lon.shape[0]))

    U *= scalefac
    V *= scalefac

    data = {'U': U, 'V': V}
    dimensions = {'time': time, 'lon': lon, 'lat': lat}
    fieldset = None
    if periodic_wrap:
        fieldset = FieldSet.from_data(data, dimensions, mesh='flat', transpose=False, time_periodic=delta(days=366))
    else:
        fieldset = FieldSet.from_data(data, dimensions, mesh='flat', transpose=False, allow_time_extrapolation=True)
    if write_out:
        fieldset.write(filename=write_out)
    return fieldset, a, b


def fieldset_from_file(periodic_wrap=False, filepath=None):
    """Simulate a current from structured random noise (i.e. Perlin noise).
    we use the external package 'perlin-numpy' as field generator, see:
    https://github.com/pvigier/perlin-numpy

    Perlin noise was introduced in the literature here:
    Perlin, Ken (July 1985). "An Image Synthesizer". SIGGRAPH Comput. Graph. 19 (97–8930), p. 287–296.
    doi:10.1145/325165.325247, https://dl.acm.org/doi/10.1145/325334.325247
    """
    if filepath is None:
        return None
    # img_shape = (perlinres[0]*shapescale[0], int(math.pow(2,noctaves))*perlinres[1]*shapescale[1], int(math.pow(2,noctaves))*perlinres[2]*shapescale[2])

    # # Coordinates of the test fieldset (on A-grid in deg)
    # lon = np.linspace(0, a, img_shape[1], dtype=np.float32)
    # lat = np.linspace(0, b, img_shape[2], dtype=np.float32)
    # totime = img_shape[0] * 24.0 * 60.0 * 60.0
    # time = np.linspace(0., totime, img_shape[0], dtype=np.float32)

    # dimensions = {'time': time, 'lon': lon, 'lat': lat}
    # dims = ('time', 'lat', 'lon')

    # variables = {'U': 'Uxr', 'V': 'Vxr'}
    # dimensions = {'time': 'time', 'lat': 'lat', 'lon': 'lon'}
    extra_fields = {}
    head_dir = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    fname, fext = os.path.splitext(basename)
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
    fieldset = None
    if periodic_wrap:
        fieldset = FieldSet.from_parcels(filepath, extra_fields=extra_fields, time_periodic=delta(days=366), deferred_load=True, allow_time_extrapolation=False, chunksize='auto')
        # return FieldSet.from_xarray_dataset(ds, variables, dimensions, mesh='flat', time_periodic=delta(days=366))
    else:
        fieldset = FieldSet.from_parcels(filepath, extra_fields=extra_fields, time_periodic=None, deferred_load=True, allow_time_extrapolation=True, chunksize='auto')
        # return FieldSet.from_xarray_dataset(ds, variables, dimensions, mesh='flat', allow_time_extrapolation=True)
    lon = fieldset.U.lon
    a = lon[len(lon)-1] - lon[0]
    lat = fieldset.V.lat
    b = lat[len(lat) - 1] - lat[0]
    if "W" in extra_fields:
        depth = fieldset.W.depth
        c = depth[len(depth)-1] - depth[0]
        fieldset.add_constant("top", depth[0] + 0.001)
        fieldset.add_constant("bottom", depth[len(depth)-1] - 0.001)
    return fieldset, a, b, c


def getU():
    return fieldset.U


def getV():
    return fieldset.V


def getW():
    return fieldset.W if hasattr(fieldset, "W") else None


class AgeParticle_JIT(JITParticle):
    age = Variable('age', dtype=np.float64, initial=0.0)
    life_expectancy = Variable('life_expectancy', dtype=np.float64, initial=np.finfo(np.float64).max)
    initialized_dynamic = Variable('initialized_dynamic', dtype=np.int32, initial=0)


class AgeParticle_SciPy(ScipyParticle):
    age = Variable('age', dtype=np.float64, initial=0.0)
    life_expectancy = Variable('life_expectancy', dtype=np.float64, initial=np.finfo(np.float64).max)
    initialized_dynamic = Variable('initialized_dynamic', dtype=np.int32, initial=0)


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


age_ptype = {'scipy': AgeParticle_SciPy, 'jit': AgeParticle_JIT}

if __name__=='__main__':
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-i", "--imageFileName", dest="imageFileName", type=str, default="benchmark_perlin.png", help="image file name of the plot")
    parser.add_argument("-b", "--backwards", dest="backwards", action='store_true', default=False, help="enable/disable running the simulation backwards")
    parser.add_argument("-p", "--periodic", dest="periodic", action='store_true', default=False, help="enable/disable periodic wrapping (else: extrapolation)")
    parser.add_argument("-r", "--release", dest="release", action='store_true', default=False, help="continuously add particles via repeatdt (default: False)")
    parser.add_argument("-rt", "--releasetime", dest="repeatdt", type=int, default=720, help="repeating release rate of added particles in Minutes (default: 720min = 12h)")
    parser.add_argument("-a", "--aging", dest="aging", action='store_true', default=False, help="Removed aging particles dynamically (default: False)")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=1, help="runtime in days (default: 1)")
    parser.add_argument("-w", "--writeout", dest="write_out", action='store_true', default=False, help="write data in outfile")
    parser.add_argument("-d", "--delParticle", dest="delete_particle", action='store_true', default=False, help="switch to delete a particle (True) or reset a particle (default: False).")
    parser.add_argument("-A", "--animate", dest="animate", action='store_true', default=False, help="animate the particle trajectories during the run or not (default: False).")
    parser.add_argument("-V", "--visualize", dest="visualize", action='store_true', default=False, help="Visualize particle trajectories at the end (default: False). Requires -w in addition to take effect.")
    parser.add_argument("-N", "--n_particles", dest="nparticles", type=str, default="2**6", help="number of particles to generate and advect (default: 2e6)")
    parser.add_argument("-sN", "--start_n_particles", dest="start_nparticles", type=str, default="96", help="(optional) number of particles generated per release cycle (if --rt is set) (default: 96)")
    parser.add_argument("-m", "--mode", dest="compute_mode", choices=['jit','scipy'], default="jit", help="computation mode = [JIT, SciPy]")
    parser.add_argument("-tp", "--type", dest="pset_type", default="soa", help="particle set type = [SOA, AOS]")
    parser.add_argument("-G", "--GC", dest="useGC", action='store_true', default=False, help="using a garbage collector (default: false)")
    parser.add_argument("-3D", "--threeD", dest="threeD", action='store_true', default=False, help="make a 3D-simulation (default: False).")
    args = parser.parse_args()

    pset_type = str(args.pset_type).lower()
    assert pset_type in pset_types
    ParticleSet = pset_types[pset_type]['pset']

    imageFileName=args.imageFileName
    periodicFlag=args.periodic
    backwardSimulation = args.backwards
    repeatdtFlag=args.release
    repeatRateMinutes=args.repeatdt
    time_in_days = args.time_in_days
    time_in_years = int(float(time_in_days)/365.0)
    agingParticles = args.aging
    with_GC = args.useGC
    Nparticle = int(float(eval(args.nparticles)))
    target_N = Nparticle
    addParticleN = 1
    # np_scaler = math.sqrt(3.0/2.0)
    # np_scaler = (3.0 / 2.0)**2.0       # **
    np_scaler = 3.0 / 2.0
    # cycle_scaler = math.sqrt(3.0/2.0)
    # cycle_scaler = (3.0 / 2.0)**2.0    # **
    # cycle_scaler = 3.0 / 2.0
    cycle_scaler = 7.0 / 4.0
    start_N_particles = int(float(eval(args.start_nparticles)))

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

    dt_minutes = 60
    nowtime = datetime.datetime.now()
    ParcelsRandom.seed(nowtime.microsecond)

    a, b, c = 1.0, 1.0, 1.0
    use_3D = args.threeD

    # fnmatch.fnmatchcase(os.uname()[2], "*.el*_*.x86_64*")
    branch = "soa_benchmark"
    computer_env = "local/unspecified"
    scenario = "perlin"
    odir = ""
    if os.uname()[1] in ['science-bs35', 'science-bs36']:  # Gemini
        odir = "/scratch/{}/experiments/parcels_benchmarking/{}".format("ckehl", str(args.pset_type))
        computer_env = "Gemini"
    elif os.uname()[1] in ["lorenz.science.uu.nl",] or fnmatch.fnmatchcase(os.uname()[1], "node*"):  # Lorenz
        CARTESIUS_SCRATCH_USERNAME = 'ckehl'
        odir = "/storage/shared/oceanparcels/output_data/data_{}/experiments/parcels_benchmarking/{}".format(CARTESIUS_SCRATCH_USERNAME, str(args.pset_type))
        computer_env = "Lorenz"
    elif fnmatch.fnmatchcase(os.uname()[1], "*.bullx*"):  # Cartesius
        CARTESIUS_SCRATCH_USERNAME = 'ckehluu'
        odir = "/scratch/shared/{}/experiments/parcels_benchmarking/{}".format(CARTESIUS_SCRATCH_USERNAME, str(args.pset_type))
        computer_env = "Cartesius"
    elif fnmatch.fnmatchcase(os.uname()[1], "int*.snellius.*") or fnmatch.fnmatchcase(os.uname()[1], "fcn*") or fnmatch.fnmatchcase(os.uname()[1], "tcn*") or fnmatch.fnmatchcase(os.uname()[1], "gcn*") or fnmatch.fnmatchcase(os.uname()[1], "hcn*"):  # Snellius
        SNELLIUS_SCRATCH_USERNAME = 'ckehluu'
        odir = "/scratch-shared/{}/experiments/parcels_benchmarking/{}".format(SNELLIUS_SCRATCH_USERNAME, str(args.pset_type))
        computer_env = "Snellius"
    else:
        odir = "/var/scratch/experiments/{}".format(str(args.pset_type))
    print("running {} on {} (uname: {}) - branch '{}' - (target) N: {} - argv: {}".format(scenario, computer_env, os.uname()[1], branch, target_N, sys.argv[1:]))

    if os.path.sep in imageFileName:
        head_dir = os.path.dirname(imageFileName)
        if head_dir[0] == os.path.sep:
            odir = head_dir
        else:
            odir = os.path.join(odir, head_dir)
            imageFileName = os.path.split(imageFileName)[1]
    pfname, pfext = os.path.splitext(imageFileName)

    func_time = []
    mem_used_GB = []

    np.random.seed(0)
    global fieldset
    field_fpath = False
    if args.write_out:
        field_fpath = os.path.join(odir,"perlin")
    if field_fpath and os.path.exists(field_fpath+"U.nc"):
        fieldset, a, b, c = fieldset_from_file(periodic_wrap=periodicFlag, filepath=field_fpath)
        use_3D &= hasattr(fieldset, "W")
    else:
        if not use_3D:
            fieldset, a, b = perlin_waves(periodic_wrap=periodicFlag, write_out=field_fpath)
        else:
            fieldset, a, b, c = perlin_waves3D(periodic_wrap=periodicFlag, write_out=field_fpath)
    fieldset.add_constant("east_lim", +a * 0.5)
    fieldset.add_constant("west_lim", -a * 0.5)
    fieldset.add_constant("north_lim", +b * 0.5)
    fieldset.add_constant("south_lim", -b * 0.5)
    fieldset.add_constant("isThreeD", 1.0 if use_3D else -1.0)

    if args.compute_mode == 'scipy':
        Nparticle = 2**10

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

    pt_ndims = 3 if use_3D else 2
    if backwardSimulation:
        # ==== backward simulation ==== #
        if agingParticles:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * (-a) + (a/2.0), lat=np.random.rand(start_N_particles, 1) * (-b) + (b/2.0), depth=np.random.rand(start_N_particles, 1) * (-c/2) + (c * 0.75) if use_3D else None, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                if pset_type != 'nodes':
                    psetA = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(int(addParticleN), 1) * (-a) + (a/2.0), lat=np.random.rand(int(addParticleN), 1) * (-b) + (b/2.0), depth=np.random.rand(int(addParticleN), 1) * (-c/2) + (c * 0.75) if use_3D else None, time=simStart)
                    pset.add(psetA)
                else:
                    lonlat_field = np.random.rand(int(addParticleN), pt_ndims)
                    marr = np.array([a, b, c/2]) if use_3D else np.array([a, b])
                    lonlat_field *= marr
                    lonlat_field[:, 0] = -lonlat_field[:, 0] + (a / 2.0)
                    lonlat_field[:, 1] = -lonlat_field[:, 1] + (b / 2.0)
                    if use_3D:
                        lonlat_field[:, 2] = -lonlat_field[:, 2] + (c * 0.75)
                    time_field = np.ones((int(addParticleN), 1), dtype=np.float64) * simStart
                    # pdata = np.concatenate( (lonlat_field, time_field), axis=1 )
                    pdata = {'lon': lonlat_field[:, 0], 'lat': lonlat_field[:, 1], 'time': time_field}
                    pset.add(pdata)
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * (-a) + (a/2.0), lat=np.random.rand(Nparticle, 1) * (-b) + (b/2.0), depth=np.random.rand(Nparticle, 1) * (-c/2) + (c * 0.75) if use_3D else None, time=simStart)
        else:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * (-a) + (a/2.0), lat=np.random.rand(start_N_particles, 1) * (-b) + (b/2.0), depth=np.random.rand(start_N_particles, 1) * (-c/2) + (c * 0.75) if use_3D else None, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                if pset_type != 'nodes':
                    psetA = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(int(addParticleN), 1) * (-a) + (a/2.0), lat=np.random.rand(int(addParticleN), 1) * (-b) + (b/2.0), depth=np.random.rand(int(addParticleN), 1) * (-c/2) + (c * 0.75) if use_3D else None, time=simStart)
                    pset.add(psetA)
                else:
                    lonlat_field = np.random.rand(int(addParticleN), pt_ndims)
                    marr = np.array([a, b, c/2]) if use_3D else np.array([a, b])
                    lonlat_field *= marr
                    lonlat_field[:, 0] = -lonlat_field[:, 0] + (a / 2.0)
                    lonlat_field[:, 1] = -lonlat_field[:, 1] + (b / 2.0)
                    if use_3D:
                        lonlat_field[:, 2] = -lonlat_field[:, 2] + (c * 0.75)
                    time_field = np.ones((int(addParticleN), 1), dtype=np.float64) * simStart
                    # pdata = np.concatenate( (lonlat_field, time_field), axis=1 )
                    pdata = {'lon': lonlat_field[:, 0], 'lat': lonlat_field[:, 1], 'time': time_field}
                    pset.add(pdata)
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * (-a) + (a/2.0), lat=np.random.rand(Nparticle, 1) * (-b) + (b/2.0), depth=np.random.rand(Nparticle, 1) * (-c/2) + (c * 0.75) if use_3D else None, time=simStart)
    else:
        # ==== forward simulation ==== #
        if agingParticles:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * (-a) + (a/2.0), lat=np.random.rand(start_N_particles, 1) * (-b) + (b/2.0), depth=np.random.rand(start_N_particles, 1) * (-c/2) + (c * 0.75) if use_3D else None, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                if pset_type != 'nodes':
                    psetA = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(int(addParticleN), 1) * (-a) + (a/2.0), lat=np.random.rand(int(addParticleN), 1) * (-b) + (b/2.0), depth=np.random.rand(int(addParticleN), 1) * (-c/2) + (c * 0.75) if use_3D else None, time=simStart)
                    pset.add(psetA)
                else:
                    lonlat_field = np.random.rand(int(addParticleN), pt_ndims)
                    marr = np.array([a, b, c/2]) if use_3D else np.array([a, b])
                    lonlat_field *= marr
                    lonlat_field[:, 0] = -lonlat_field[:, 0] + (a / 2.0)
                    lonlat_field[:, 1] = -lonlat_field[:, 1] + (b / 2.0)
                    if use_3D:
                        lonlat_field[:, 2] = -lonlat_field[:, 2] + (c * 0.75)
                    time_field = np.ones((int(addParticleN), 1), dtype=np.float64) * simStart
                    # pdata = np.concatenate( (lonlat_field, time_field), axis=1 )
                    pdata = {'lon': lonlat_field[:, 0], 'lat': lonlat_field[:, 1], 'time': time_field}
                    pset.add(pdata)
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * (-a) + (a/2.0), lat=np.random.rand(Nparticle, 1) * (-b) + (b/2.0), depth=np.random.rand(Nparticle, 1) * (-c/2) + (c * 0.75) if use_3D else None, time=simStart)
        else:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * (-a) + (a/2.0), lat=np.random.rand(start_N_particles, 1) * (-b) + (b/2.0), depth=np.random.rand(start_N_particles, 1) * (-c/2) + (c * 0.75) if use_3D else None, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                if pset_type != 'nodes':
                    psetA = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(int(addParticleN), 1) * (-a) + (a/2.0), lat=np.random.rand(int(addParticleN), 1) * (-b) + (b/2.0), depth=np.random.rand(int(addParticleN), 1) * (-c/2) + (c * 0.75) if use_3D else None, time=simStart)
                    pset.add(psetA)
                else:
                    lonlat_field = np.random.rand(int(addParticleN), pt_ndims)
                    marr = np.array([a, b, c/2]) if use_3D else np.array([a, b])
                    lonlat_field *= marr
                    lonlat_field[:, 0] = -lonlat_field[:, 0] + (a / 2.0)
                    lonlat_field[:, 1] = -lonlat_field[:, 1] + (b / 2.0)
                    if use_3D:
                        lonlat_field[:, 2] = -lonlat_field[:, 2] + (c * 0.75)
                    time_field = np.ones((int(addParticleN), 1), dtype=np.float64) * simStart
                    # pdata = np.concatenate( (lonlat_field, time_field), axis=1 )
                    pdata = {'lon': lonlat_field[:, 0], 'lat': lonlat_field[:, 1], 'time': time_field}
                    pset.add(pdata)
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * (-a) + (a/2.0), lat=np.random.rand(Nparticle, 1) * (-b) + (b/2.0), depth=np.random.rand(Nparticle, 1) * (-c/2) + (c * 0.75) if use_3D else None, time=simStart)

    # ======== ======== End of FieldSet construction ======== ======== #
    output_file = None
    out_fname = "benchmark_perlin"
    if args.write_out:
        if MPI and (MPI.COMM_WORLD.Get_size()>1):
            out_fname += "_MPI" + "_n{}".format(MPI.COMM_WORLD.Get_size())
            pfname += "_MPI" + "_n{}".format(MPI.COMM_WORLD.Get_size())
        else:
            out_fname += "_noMPI"
            pfname += "_noMPI"
        if periodicFlag:
            out_fname += "_p"
            pfname += '_p'
        out_fname += "_n"+str(Nparticle)
        pfname += "_n"+str(Nparticle)
        # if time_in_years != 1:
        #     out_fname += '_%dy' % (time_in_years, )
        #     pfname += '_%dy' % (time_in_years, )
        out_fname += '_%dd' % (time_in_days, )
        pfname += '_%dd' % (time_in_days, )
        if use_3D:
            out_fname += "_3D"
            pfname += "_3D"
        if backwardSimulation:
            out_fname += "_bwd"
            pfname += "_bwd"
        else:
            out_fname += "_fwd"
            pfname += "_fwd"
        if repeatdtFlag:
            out_fname += "_add"
            pfname += "_add"
        if agingParticles:
            out_fname += "_age"
            pfname += "_age"
        if with_GC:
            out_fname += "_wGC"
            pfname += "_wGC"
        else:
            out_fname += "_woGC"
            pfname += "_woGC"
        output_file = pset.ParticleFile(name=os.path.join(odir, out_fname+".nc"), outputdt=delta(hours=24))
    imageFileName = pfname + pfext

    delete_func = RenewParticle
    if args.delete_particle:
        delete_func = DeleteParticle
    postProcessFuncs = None
    callbackdt = None
    if with_GC:
        postProcessFuncs = [perIterGC, ]
        callbackdt = delta(hours=12)

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            starttime = ostime.process_time()
    else:
        starttime = ostime.process_time()
    advect_K = AdvectionRK4
    if use_3D:
        advect_K = AdvectionRK4_3D
    kernels = pset.Kernel(advect_K ,delete_cfiles=True)
    if use_3D:
        kernels += pset.Kernel(reflect_top_bottom, delete_cfiles=True)
    kernels += pset.Kernel(periodicBC, delete_cfiles=True)
    if agingParticles:
        kernels += pset.Kernel(initialize, delete_cfiles=True)
        kernels += pset.Kernel(Age, delete_cfiles=True)
    if backwardSimulation:
        # ==== backward simulation ==== #
        if args.animate:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=-dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func, ErrorCode.ErrorThroughSurface: reflect_top_bottom, ErrorCode.ErrorInterpolation: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12), moviedt=delta(hours=6), movie_background_field=fieldset.U)
        else:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=-dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func, ErrorCode.ErrorThroughSurface: reflect_top_bottom, ErrorCode.ErrorInterpolation: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12))
    else:
        # ==== forward simulation ==== #
        if args.animate:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func, ErrorCode.ErrorThroughSurface: reflect_top_bottom, ErrorCode.ErrorInterpolation: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12), moviedt=delta(hours=6), movie_background_field=fieldset.U)
        else:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func, ErrorCode.ErrorThroughSurface: reflect_top_bottom, ErrorCode.ErrorInterpolation: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12))

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            endtime = ostime.process_time()
    else:
        endtime = ostime.process_time()

    if args.write_out:
        output_file.close()

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        # mpi_comm.Barrier()
        Npart = Nparticle
        Npart = mpi_comm.reduce(Npart, op=MPI.SUM, root=0)
        if mpi_comm.Get_rank() == 0:
            sys.stdout.write("final # particles: {}\n".format( Npart ))
            sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime-starttime))
    else:
        Npart = Nparticle
        sys.stdout.write("final # particles: {}\n".format( Npart ))
        sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime - starttime))


