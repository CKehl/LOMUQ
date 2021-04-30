import os
import h5py
import math
from scipy.io import netcdf
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib import patches
from matplotlib import colors
from matplotlib.colors import colorConverter
from matplotlib.animation import FuncAnimation, writers
from multivariate_colour_palettes import colour_scales
from DecayLine import DecayLine
from datetime import timedelta
from argparse import ArgumentParser
from TransparentCircles import TransparentCircles, TransparentEllipses

# the point-scatter plot blitting just DOESN'T FUCKING WORK ... I really tried all functions in the book here,
# but matplotlib just proves again uncooperative when co-plotting and blitting data on top of one another.
# we will require a real vis backend - did someone say 'VTK' ?! :-(

# One option remains: ArtistAnimation(...)
GLOBAL_MIN_DIST = 0.00005

def time_index_value(tx, _ft):
    # expect ft to be forward-linear
    f_dt = _ft[1] - _ft[0]
    if type(f_dt) is not np.float64:
        fdt = timedelta(f_dt).total_seconds()
    f_interp = tx / f_dt
    ti = int(math.floor(f_interp))
    return ti


def time_partion_value(tx, _ft):
    # expect ft to be forward-linear
    f_dt = _ft[1] - _ft[0]
    if type(f_dt) is not np.float64:
        f_dt = timedelta(f_dt).total_seconds()
    f_interp = tx / f_dt
    f_t = f_interp - math.floor(f_interp)
    return f_t

class DistanceTracker():
    _in_data_x = None
    _in_data_y = None
    _in_data_z = None
    _in_dt = None
    _out_dt = None
    _result = None
    _result_shape = None
    _timetrack = None
    _iter_delta = None
    _current_time = 0

    def __init__(self, in_data_x, in_data_y, in_data_z, in_dt, out_dt):
        self._in_data_x = in_data_x
        self._in_data_y = in_data_y
        self._in_data_z = in_data_z
        self._in_dt = in_dt
        self._out_dt = out_dt
        self._result_shape = self._in_data_x.shape
        self._result = np.ones(self._result_shape, dtype=np.float32) * GLOBAL_MIN_DIST
        self._iter_delta = int(np.ceil(self._out_dt / self._in_dt))
        self._timetrack = np.zeros(self._iter_delta, dtype=np.int32) - 1
        self._current_time = -1
        self._iter_counter = -1

    def __del__(self):
        self._in_data_x = None
        self._in_data_y = None
        self._in_data_z = None
        self._in_dt = None
        self._out_dt = None
        del self._result
        self._result = None
        del self._timetrack
        self._timetrack = None
        self._result_shape = None
        self._iter_delta = None
        self._current_time = None
        self._iter_counter = None

    @property
    def results(self):
        return self._result

    def compute_all(self):
        total_items = self._in_data_x.shape[1]
        for ti in range(total_items):
            self.compute(ti)
            current_item = ti
            workdone = current_item / total_items
            print("\rProgress - computing traveled distance per particle: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)

    def compute(self, iteration):
        prev_time_frac = self._current_time / self._out_dt
        prev_out_shift = int(math.floor(prev_time_frac))
        prev_out_interp = prev_time_frac - prev_out_shift
        # self._current_time += abs(self._in_dt)
        self._current_time = iteration * abs(self._in_dt)
        curr_time_frac = self._current_time / self._out_dt
        curr_out_shift = int(math.floor(curr_time_frac))
        curr_out_interp = curr_time_frac - curr_out_shift
        if prev_out_shift < curr_out_shift:
            dshift = curr_out_shift-max(0, prev_out_shift)
            for i in range(dshift):
                self._timetrack[1:] = self._timetrack[0:-1]
                self._iter_counter += 1
                self._timetrack[0] = self._iter_counter
        tt_sel_inds = np.where(self._timetrack >= 0)
        tt_sel = np.sort(self._timetrack[tt_sel_inds])
        lsize = tt_sel.shape[0]
        if curr_out_shift > 0 and lsize > 1:
            lx = self._in_data_x[:, tt_sel]
            ly = self._in_data_y[:, tt_sel]
            lz = self._in_data_z[:, tt_sel]
            local_distances = np.zeros((self._in_data_x.shape[0], lsize-1), dtype=np.int32)
            for i in range(lsize-1):
                local_distances[:, i] = np.sqrt( (lx[:, i+1]-lx[:, i])**2 + (ly[:, i+1]-ly[:, i])**2 + (lz[:, i+1]-lz[:, i])**2 )
            self._result[:, iteration] = np.sum(local_distances[:, :-1], axis=1) + curr_out_interp * local_distances[:, -1]

def ClipLogNorm_forward(x):
    x[np.isnan(x)] = np.finfo(x.dtype).min
    x[np.isinf(x)] = np.finfo(x.dtype).max
    x[np.isclose(x, 0)] = np.finfo(x.dtype).min
    return np.log10(x)

def ClipLogNorm_invert(x):
    x[np.isnan(x)] = np.finfo(x.dtype).min
    x[np.isinf(x)] = np.finfo(x.dtype).max
    x[np.isclose(x, 0)] = np.finfo(x.dtype).min
    valid_inds = ~np.isnan(x) & ~np.isinf(x) & ~np.isclose(x, 0)
    x[valid_inds] = x[valid_inds]**2
    return x

if __name__ == '__main__':
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-d", "--filedir", dest="filedir", type=str,
                        default="/var/scratch/experiments/NNvsGeostatistics/data",
                        help="head directory containing all input data and also are the store target for output files")
    parser.add_argument("-o", "--outname", dest="outname", type=str,
                        default="dispersion.png",
                        help="output file naming")
    parser.add_argument("-dt", "--deltat", dest="dt", type=float, default=1.0, help="output period in days")
    parser.add_argument("-A", "--animate", dest="animate", action='store_true', default=False,
                        help="Generate particle animations (default: False).")
    args = parser.parse_args()

    # filedir = '/var/scratch/experiments/NNvsGeostatistics/data/2'
    filedir = args.filedir
    animate = args.animate
    outname = args.outname
    anim_dt = args.dt
    write_nc = False
    Writer = writers['imagemagick_file']
    ani_writer_h5 = Writer()
    ani_writer_nc = Writer()
    psize = 10.0
    ls = psize * 0.05
    alpha = 0.3
    zorder = 1

    # ==== Load flow-field data ==== #
    f_grid_h5 = h5py.File(os.path.join(filedir, 'grid.h5'), "r")
    fT_h5 = f_grid_h5['times']
    fT_h5_attrs = fT_h5.attrs
    fX_h5 = f_grid_h5['longitude']
    fX_h5_attrs = fX_h5.attrs
    fY_h5 = f_grid_h5['latitude']
    fY_h5_attrs = fY_h5.attrs
    extends = (float(fX_h5_attrs['min']), float(fX_h5_attrs['max']), float(fY_h5_attrs['min']), float(fY_h5_attrs['max']))

    f_u_h5 = h5py.File(os.path.join(filedir, 'hydrodynamic_U.h5'), "r")
    fU_h5 = f_u_h5['uo']  # [()]
    fU_h5_attrs = fU_h5.attrs
    max_u_value = np.maximum(np.abs(fU_h5_attrs['min']), np.abs(fU_h5_attrs['max']))
    fu_ext = (-max_u_value, +max_u_value)
    f_v_h5 = h5py.File(os.path.join(filedir, 'hydrodynamic_V.h5'), "r")
    fV_h5 = f_v_h5['vo']  # [()]
    fV_h5_attrs = fV_h5.attrs
    max_v_value = np.maximum(np.abs(fV_h5_attrs['min']), np.abs(fV_h5_attrs['max']))
    fv_ext = (-max_v_value, +max_v_value)

    # f_velmag_ext_h5 = (np.finfo(fU_h5.dtype).eps, np.sqrt(max_v_value**2 + max_u_value**2))
    f_velmag_ext_h5 = (np.finfo(fU_h5.dtype).max, np.finfo(fU_h5.dtype).min)
    for ti in range(fT_h5.shape[0]):
        speed_c = fU_h5[ti] ** 2 + fV_h5[ti] ** 2
        speed_c = np.where(speed_c > 0, np.sqrt(speed_c), 0)
        speed_c_invalid = np.isclose(speed_c, 0) & np.isclose(speed_c, -0)
        if not speed_c_invalid.all():
            f_velmag_ext_h5 = (np.minimum(f_velmag_ext_h5[0], np.min(speed_c[~speed_c_invalid])), np.maximum(f_velmag_ext_h5[1], np.max(speed_c[~speed_c_invalid])))
        else:
            f_velmag_ext_h5 = (f_velmag_ext_h5[0], np.maximum(f_velmag_ext_h5[1], np.max(speed_c)))
    print(f_velmag_ext_h5)

    total_items = fT_h5.shape[0]
    tN = fT_h5.shape[0]

    # u_h5_dir = os.path.join(filedir, "U_h5")
    # if not os.path.exists(u_h5_dir):
    #     os.mkdir(u_h5_dir)
    # v_h5_dir = os.path.join(filedir, "V_h5")
    # if not os.path.exists(v_h5_dir):
    #     os.mkdir(v_h5_dir)
    # velmag_h5_dir = os.path.join(filedir, "VelMag_h5")
    # if not os.path.exists(velmag_h5_dir):
    #     os.mkdir(velmag_h5_dir)

    speed_h5_ti = fU_h5[0] ** 2 + fU_h5[0] ** 2
    speed_h5_ti = np.where(speed_h5_ti > 0, np.sqrt(speed_h5_ti), 0)

    fig_h5_single_u, ax_h5_single_u = plt.subplots(1, 1)
    ax_h5_single_u.set_facecolor('grey')
    cs_h5s_u = plt.imshow(fU_h5[0], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['U']['colour_scale'], interpolation='bilinear', vmin=fu_ext[0], vmax=fu_ext[1])
    cbar_h5s_u = fig_h5_single_u.add_axes([0.0, 0.9, 0.05, 0.07])
    plt.colorbar(cs_h5s_u, cax=cbar_h5s_u)

    fig_h5_single_v, ax_h5_single_v = plt.subplots(1, 1)
    ax_h5_single_v.set_facecolor('grey')
    cs_h5s_v = plt.imshow(fV_h5[0], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['V']['colour_scale'], interpolation='bilinear', vmin=fv_ext[0], vmax=fv_ext[1])  # , alpha=0.95
    cbar_h5s_v = fig_h5_single_v.add_axes([0.0, 0.9, 0.05, 0.07])
    plt.colorbar(cs_h5s_v, cax=cbar_h5s_v)

    fig_h5_single_velmag, ax_h5_single_velmag = plt.subplots(1, 1)
    ax_h5_single_velmag.set_facecolor('black')
    cs_h5s_velmag = plt.imshow(speed_h5_ti, extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['VelMag']['colour_scale'], interpolation='bilinear', vmin=f_velmag_ext_h5[0], vmax=f_velmag_ext_h5[1])
    ax_cbar_h5s_velmag = fig_h5_single_velmag.add_axes([0.0, 0.9, 0.05, 0.07])
    cbar_h5s_velmag = plt.colorbar(cs_h5s_velmag, cax=ax_cbar_h5s_velmag)

    # di = 0  # depth index - fixed to zero cause data are 2D

    Pn = 2048
    # indices = None  # np.random.randint(0, 100, Pn, dtype=int)

    # ============================================== #
    # ====  Particle Trails - Animation - HDF5  ==== #
    # ============================================== #
    if animate:
        f_pts = h5py.File(os.path.join(filedir, 'particles.h5'), "r")
        pT_h5 = f_pts['p_t']  # [()]
        pT_0_h5 = np.array(pT_h5[:, 0])
        pT_1_h5 = np.array(pT_h5[:, 1])
        sec_per_day = 86400.0

        N = pT_h5.shape[0]
        tN = pT_h5.shape[1]
        print("N: {}; Pn: {}; tN: {}".format(N, Pn, tN))
        # if indices is None:
        #     indices = np.random.randint(0, N - 1, Pn, dtype=int)

        time_since_release_h5 = pT_h5
        # time_since_release_h5 = (pT_h5.transpose() - pT_h5[:, 0])  # substract the initial time from each timeseries
        # sim_dt_h5 = time_since_release_h5[0, 1] - time_since_release_h5[0, 0]
        sim_dt_h5 = np.nanmax(pT_1_h5) - np.nanmax(pT_0_h5)
        print("dt = {}".format(sim_dt_h5))
        pX_h5 = f_pts['p_x']  # [()][indices, :]  # only plot the first 32 particles
        pY_h5 = f_pts['p_y']  # [()][indices, :]  # only plot the first 32 particles
        pZ_h5 = f_pts['p_z']  # [()][indices, :]  # only plot the first 32 particles
        sim_dt_d_h5 = sim_dt_h5 / sec_per_day
        print("Starting distance calculations ...")
        dtracker = DistanceTracker(pX_h5, pY_h5, pZ_h5, sim_dt_d_h5, 7.0)
        dtracker.compute_all()
        print("\nDistance calculation finished.")
        distances = dtracker.results
        distances_neg_invalid = np.isclose(distances, 0) & np.isclose(distances,-0) & (distances < 0)
        ds_min = np.maximum(distances[~distances_neg_invalid].min(), GLOBAL_MIN_DIST)
        ds_max = np.maximum(np.round(distances[~distances_neg_invalid].max(), decimals=4), GLOBAL_MIN_DIST)
        distances[distances_neg_invalid] = np.finfo(distances.dtype).eps
        print("({}, {})".format(ds_min, ds_max))


        speed_h5_0 = fU_h5[0] ** 2 + fV_h5[0] ** 2
        speed_h5_0 = np.where(speed_h5_0 > 0, np.sqrt(speed_h5_0), 0)
        speed_0_invalid = np.isclose(speed_h5_0, 0) & np.isclose(speed_h5_0,-0)
        speed_0_min = np.finfo(speed_h5_0.dtype).eps if speed_0_invalid.all() else np.min(speed_h5_0[~speed_0_invalid])
        speed_h5_0[speed_0_invalid] = speed_0_min

        indices = np.random.randint(0, N - 1, Pn, dtype=int)
        # sortindices = np.argsort(distances[indices, 0])
        # sortindices = np.arange(0, indices.shape[0])
        px_t_0 = np.array(pX_h5[:, 0])
        px_t_0 = np.take_along_axis(px_t_0, indices, axis=0)
        # px_t_0 = np.take_along_axis(px_t_0, indices[sortindices], axis=0)
        py_t_0 = np.array(pY_h5[:, 0])
        py_t_0 = np.take_along_axis(py_t_0, indices, axis=0)
        # py_t_0 = np.take_along_axis(py_t_0, indices[sortindices], axis=0)
        pd_t_0 = np.array(distances[:, 0])
        pd_t_0 = np.take_along_axis(pd_t_0, indices, axis=0)
        # pd_t_0 = np.take_along_axis(pd_t_0, indices[sortindices], axis=0)

        base_px = np.array([extends[0], extends[0], extends[1], extends[1]])
        base_py = np.array([extends[2], extends[3], extends[2], extends[3]])
        base_pd = np.array([0.01, 0.01, 0.01, 0.01])
        # base_px = np.array([0, ])
        # base_py = np.array([0, ])
        # base_pd = np.array([0.01, ])

        px_t_0 = np.hstack((base_px, px_t_0))
        py_t_0 = np.hstack((base_py, py_t_0))
        pd_t_0 = np.hstack((base_pd, pd_t_0))

        # speed_map = cm.get_cmap('twilight', 256)
        # fig_anim_h5 = plt.figure()
        # ax_anim_h5 = plt.axes()
        fig_anim_h5, ax_anim_h5 = plt.subplots(1, 1, figsize=(10, 8))  #, constrained_layout=True
        # ax_scat = fig_anim_h5.add_axes([0, 0, 1, 1], frameon=False)
        # ax_scat = ax_anim_h5
        # ax_anim_h5.grid(True, linestyle='-', color='0.75')
        ax_anim_h5.set_xlim([extends[0], extends[1]])
        ax_anim_h5.set_ylim([extends[2], extends[3]])

        def register_patch(patch):
            ax_anim_h5.add_patch(patch)

        # ideally, we want the FTLE map of the flow as background, if we can get that (-> ask Laura)
        # cs_h5a_velmag = ax_anim_h5.pcolormesh(fX_h5, fY_h5, speed_h5_0, cmap=colour_scales['blue_quadrant_palette_blackbase']['VelMag']['colour_scale'],
        #                                       norm=colors.Normalize(vmin=f_velmag_ext_h5[0], vmax=f_velmag_ext_h5[1]), shading='gouraud', animated=True, zorder=0)  # , zorder=1

        # (ClipLogNorm_forward, ClipLogNorm_invert),
        # cs_h5a_velmag = ax_anim_h5.imshow(speed_h5_0, extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['VelMag']['colour_scale'], interpolation='bilinear', norm=colors.Normalize(vmin=f_velmag_ext_h5[0], vmax=f_velmag_ext_h5[1]), animated=True, zorder=0 , origin='lower') # , origin='lower', aspect='auto'
        # cs_h5a_velmag = ax_anim_h5.imshow(speed_h5_0, extent=extends, cmap=colour_scales['black_to_grey_palette_blackbase']['Any']['colour_scale'], interpolation='bilinear', norm=colors.LogNorm(vmin=f_velmag_ext_h5[0], vmax=f_velmag_ext_h5[1]), animated=True, zorder=0, origin='lower')  # , origin='lower', aspect='auto'
        cs_h5a_velmag = ax_anim_h5.imshow(speed_h5_0, extent=extends, cmap=colour_scales['black_to_grey_palette_blackbase']['Any'][ 'colour_scale'], interpolation='bilinear', norm=colors.LogNorm(vmin=f_velmag_ext_h5[0], vmax=f_velmag_ext_h5[1]), animated=True, zorder=0 , origin='lower', aspect='auto')  # , origin='lower', aspect='auto'
        # ==== comment: REALLY try to apply a LogNormalize scale here ==== #
        # cs_h5a_pts_coll = plt.scatter(px_t_0, py_t_0, s=psize, linewidths=ls, c=pd_t_0, cmap=colour_scales['white_to_blue_pallete_alpha']['Any']['colour_scale'], norm=colors.Normalize(vmin=ds_min, vmax=ds_max), alpha=0.5, zorder=zorder)  # , offset_position='data', alpha=alpha, transform=ax_anim_h5.transData
        # cs_h5a_pts = TransparentCircles(px_t_0, py_t_0, register_patch_func=register_patch, s=psize, linewidths=ls, values=pd_t_0, cmap=colour_scales['white_to_blue_pallete_opaque']['Any']['colour_scale'], norm=colors.Normalize(vmin=ds_min, vmax=ds_max), alphas=alpha, zorder=2, transform=ax_anim_h5.transData)  #
        # ==== the one below is the fasted circle-plotter ==== #
        cs_h5a_pts = TransparentEllipses(px_t_0, py_t_0, s=math.sqrt(psize), linewidths=ls, values=pd_t_0, cmap=colour_scales['white_to_blue_pallete_opaque']['Any']['colour_scale'], norm=colors.LogNorm(vmin=ds_min, vmax=ds_max), zorder=2, transform=ax_anim_h5.transData, alphas=alpha, edgecolors='White')  #
        cs_h5a_pts_coll = cs_h5a_pts.get_collection()
        ax_anim_h5.add_collection(cs_h5a_pts_coll)

        # cs_h5a_pts = ax_anim_h5.scatter(px_t_0, py_t_0, s=psize, linewidths=ls, c=pd_t_0, cmap='Blues', alpha=alpha, zorder=zorder)  # , norm=colors.Normalize(vmin=ds_min, vmax=ds_max)
        # cs_h5a_pts = ax_scat.plot(px_t_0, py_t_0, 'o', color=(0.9, 0.9, 1.0), alpha=0.8)
        # ax_anim_h5.set_xlim([extends[0], extends[1]])
        # ax_anim_h5.set_ylim([extends[2], extends[3]])

        cbar_h5a_velmag_bounds = []
        logval = round(math.log10(f_velmag_ext_h5[0])) - 1
        bval = 10**(logval)
        while bval < f_velmag_ext_h5[1]:
            logval += 1
            bval = 10.0 ** (logval)
            cbar_h5a_velmag_bounds.append(bval)
        # ax_cbar_h5a_velmag = fig_anim_h5.add_axes([0.0, 0.9, 0.04, 0.085])
        ax_cbar_h5a_velmag = fig_anim_h5.add_axes([0.1, 0.055, 0.8, 0.02])
        cbar_h5a_velmag = plt.colorbar(cs_h5a_velmag, cax=ax_cbar_h5a_velmag, ticks=cbar_h5a_velmag_bounds, orientation='horizontal')
        ax_cbar_h5a_velmag.set_xlabel("velocity magnitude [m/s]")

        cbar_h5a_pts_bounds = []
        logval = round(math.log10(ds_min)) - 1
        bval = 10**(logval)
        while bval < ds_max:
            logval += 1
            bval = 10.0 ** (logval)
            cbar_h5a_pts_bounds.append(bval)
        # ax_cbar_h5a_pts = fig_anim_h5.add_axes([0.10, 0.9, 0.04, 0.085])
        ax_cbar_h5a_pts = fig_anim_h5.add_axes([0.915, 0.1, 0.02, 0.8])
        cbar_h5a_pts = plt.colorbar(cs_h5a_pts_coll, cax=ax_cbar_h5a_pts, ticks=cbar_h5a_pts_bounds, orientation='vertical')
        # ax_cbar_h5a_pts.set_xlabel("travel distance [arc-deg.]")
        ax_cbar_h5a_pts.set_ylabel("travel distance [arc-deg.]")

        ax_anim_h5.set_title("Simulation - NetCDF data - t = %5.1f d" % (time_since_release_h5[0, 0] / sec_per_day))
        plt.savefig(os.path.join(filedir, "dispersion_ref.png"), dpi=300)

        # lines_h5 = []
        # for i in range(0, Pn):
        #     lines_h5.append(DecayLine(14, 8, [0.0, 0.625, 0.0], zorder=2 + i))

        # for i in range(0, Pn):
        #     lines_h5[i].add_point(pX_h5[indices[i], 0], pY_h5[indices[i], 0])

        # for l in lines_h5:
        #     ax_anim_h5.add_collection(l.get_LineCollection())
        # frames_anim = int(tN * 0.5)
        # dt_anim = sim_dt_h5 * 2.0  # /4.0
        dt_anim = anim_dt * sec_per_day
        frames_anim = int(tN * (sim_dt_d_h5 / anim_dt))
        total_items = frames_anim

        def init_h5_animation():
            # cs_nca_u.set_array(fU_nc[0, 0])
            # cs_nca_v.set_array(fV_nc[0, 0])
            cs_h5a_velmag.set_array(speed_h5_0)
            # sortindices = np.unravel_index(np.argsort(distances[indices, 0]), indices.shape)
            # sortindices = np.arange(0, indices.shape[0])
            # print("indices: {}".format(indices[sortindices]))
            px_t_0 = np.array(pX_h5[:, 0])
            px_t_0 = np.take_along_axis(px_t_0, indices, axis=0)
            # px_t_0 = np.take_along_axis(px_t_0, indices[sortindices], axis=0)
            py_t_0 = np.array(pY_h5[:, 0])
            py_t_0 = np.take_along_axis(py_t_0, indices, axis=0)
            # py_t_0 = np.take_along_axis(py_t_0, indices[sortindices], axis=0)
            pd_t_0 = np.array(distances[:, 0])
            pd_t_0 = np.take_along_axis(pd_t_0, indices, axis=0)
            # pd_t_0 = np.take_along_axis(pd_t_0, indices[sortindices], axis=0)

            px_t_0 = np.hstack((base_px, px_t_0))
            py_t_0 = np.hstack((base_py, py_t_0))
            pd_t_0 = np.hstack((base_pd, pd_t_0))

            cs_h5a_pts.set_array(pd_t_0)
            # cs_h5a_pts_coll.set_array(pd_t_0)  # sets the values (e.g. lifetime) interpreted by the colour map
            # offsets = np.array([py_t, px_t])
            # offsets = np.array([px_t, py_t])
            # offsets = np.hstack((px_t[:, np.newaxis], py_t[:, np.newaxis]))
            # offsets = np.c_[px_t-extends[0], py_t-extends[2]]
            # offsets = np.c_[px_t_0, py_t_0].reshape(-1, 1, 2)
            offsets = np.column_stack((px_t_0, py_t_0))  # .reshape(-1, 1, 2)
            # print("offsets shape: {}".format(offsets.shape))
            # cs_h5a_pts_coll.set_offsets(offsets)  # sets the new particle positions
            cs_h5a_pts.set_offsets(offsets)
            cs_h5a_pts_coll = cs_h5a_pts.get_collection()
            # cs_h5a_pts = ax_anim_h5.scatter(px_t, py_t, s=psize, linewidths=ls, c=pd_t, cmap='Blues',
            #                                 norm=colors.Normalize(vmin=ds_min, vmax=ds_max), alpha=alpha,
            #                                 zorder=zorder)

            results = []
            results.append(cs_h5a_velmag)
            results.append(cs_h5a_pts_coll)
            # results += cs_h5a_pts.get_artists()
            # results.append(cs_h5a_pts)
            # for l in lines_h5:
            #     results.append(l.get_LineCollection())
            ax_anim_h5.set_title("Simulation - NetCDF data - t = %5.1f d" % (time_since_release_h5[0, 0] / sec_per_day))
            return results


        def update_flow_only_h5(frames, *args):
            # if (frames % 10) == 0:
            #     print("Plotting frame {} of {} ...".format(frames, tN))
            dt = args[0]
            tx = np.float64(frames) * dt
            # tx = math.fmod(tx, fT_h5[-1])
            tx = np.fmod(tx, fT_h5[-1])
            ti0 = time_index_value(tx, fT_h5)
            tt = time_partion_value(tx, fT_h5)
            ti1 = 0
            if ti0 < (len(fT_h5) - 1):
                ti1 = ti0 + 1
            else:
                ti1 = 0
            # print("Processing tx={}, tt={} ti0={} and ti1 = {}".format(tx, tt, ti0, ti1))
            speed_1 = fU_h5[ti0] ** 2 + fV_h5[ti0] ** 2
            speed_1 = np.where(speed_1 > 0, np.sqrt(speed_1), 0)
            speed_1_invalid = np.isclose(speed_1, 0) & np.isclose(speed_1,-0)
            speed_1_min = np.finfo(speed_1.dtype).eps if speed_0_invalid.all() else np.min(speed_1[~speed_1_invalid])
            speed_1[speed_1_invalid] = speed_1_min
            speed_2 = fU_h5[ti1] ** 2 + fV_h5[ti1] ** 2
            speed_2 = np.where(speed_2 > 0, np.sqrt(speed_2), 0)
            speed_2_invalid = np.isclose(speed_2, 0) & np.isclose(speed_2,-0)
            speed_2_min = np.finfo(speed_2.dtype).eps if speed_0_invalid.all() else np.min(speed_2[~speed_2_invalid])
            speed_2[speed_2_invalid] = speed_2_min
            # fu_show = tt*fU_nc[ti0] + (1.0-tt)*fU_nc[ti1]
            # fv_show = tt*fV_nc[ti0] + (1.0-tt)*fV_nc[ti1]
            fs_show = (1.0 - tt) * speed_1 + tt * speed_2
            # cs_nca_u.set_array(fu_show)
            # cs_nca_v.set_array(fv_show)
            cs_h5a_velmag.set_array(fs_show)

            # == add new lines == #
            # if frames > 0:
            #     for pindex in range(0, Pn):
            #         lines_h5[pindex].add_point(pX_h5[indices[pindex], frames], pY_h5[indices[pindex], frames])
            # == collect results == #
            # sortindices = np.unravel_index(np.argsort(distances[indices, ti1]), indices.shape)
            # sortindices = np.arange(0, indices.shape[0])
            px_t_0 = np.array(pX_h5[:, ti0])
            px_t_0 = np.take_along_axis(px_t_0, indices, axis=0)
            # px_t_0 = np.take_along_axis(px_t_0, indices[sortindices], axis=0)
            py_t_0 = np.array(pY_h5[:, ti0])
            py_t_0 = np.take_along_axis(py_t_0, indices, axis=0)
            # py_t_0 = np.take_along_axis(py_t_0, indices[sortindices], axis=0)
            pd_t_0 = np.array(distances[:, ti0])
            pd_t_0 = np.take_along_axis(pd_t_0, indices, axis=0)
            # pd_t_0 = np.take_along_axis(pd_t_0, indices[sortindices], axis=0)
            px_t_1 = np.array(pX_h5[:, ti1])
            px_t_1 = np.take_along_axis(px_t_1, indices, axis=0)
            # px_t_1 = np.take_along_axis(px_t_1, indices[sortindices], axis=0)
            py_t_1 = np.array(pY_h5[:, ti1])
            py_t_1 = np.take_along_axis(py_t_1, indices, axis=0)
            # py_t_1 = np.take_along_axis(py_t_1, indices[sortindices], axis=0)
            pd_t_1 = np.array(distances[:, ti1])
            pd_t_1 = np.take_along_axis(pd_t_1, indices, axis=0)
            # pd_t_1 = np.take_along_axis(pd_t_1, indices[sortindices], axis=0)
            px_show = (1.0 - tt) * px_t_0 + tt * px_t_1
            py_show = (1.0 - tt) * py_t_0 + tt * py_t_1
            pd_show = (1.0 - tt) * pd_t_0 + tt * pd_t_1

            px_show = np.hstack((base_px, px_show))
            py_show = np.hstack((base_py, py_show))
            pd_show = np.hstack((base_pd, pd_show))

            cs_h5a_pts.set_array(pd_show)
            # cs_h5a_pts_coll.set_array(pd_show)  # sets the values (e.g. lifetime) interpreted by the colour map
            # offsets = np.array([py_show, px_show])
            # offsets = np.array([px_show, py_show])
            # offsets = np.hstack((px_show[:, np.newaxis], py_show[:, np.newaxis]))
            # offsets = np.c_[px_show-extends[0], py_show-extends[2]]
            # offsets = np.c_[px_show, py_show].reshape(-1, 1, 2)
            offsets = np.column_stack((px_show, py_show))  # .reshape(-1, 1, 2)
            # cs_h5a_pts_coll.set_offsets(offsets)  # sets the new particle positions
            cs_h5a_pts.set_offsets(offsets)  # sets the new particle positions
            cs_h5a_pts_coll = cs_h5a_pts.get_collection()
            # cs_h5a_pts = ax_anim_h5.scatter(px_show, py_show, s=psize, linewidths=ls, c=pd_show, cmap='Blues',
            #                                 norm=colors.Normalize(vmin=ds_min, vmax=ds_max), alpha=alpha,
            #                                 zorder=zorder)

            # == collected particles == #
            results = []
            results.append(cs_h5a_velmag)
            results.append(cs_h5a_pts_coll)
            # results += cs_h5a_pts.get_artists()
            # results.append(cs_h5a_pts)
            # for l in lines_h5:
            #     results.append(l.get_LineCollection())
            ax_anim_h5.set_title("Simulation - h5 data - t = %5.1f d" % (tx / sec_per_day))

            del px_t_0
            del px_t_1
            del py_t_0
            del py_t_1
            del pd_t_0
            del pd_t_1

            current_item = frames
            workdone = current_item / total_items
            print("\rProgress - plotting particle dispersion: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
            return results

        ani_h5_sim = FuncAnimation(fig_anim_h5, update_flow_only_h5, init_func=init_h5_animation, frames=frames_anim, interval=1,
                               fargs=[dt_anim, ])  # , save_count=1, cache_frame_data=False) #, blit=True), blit=True
        ani_h5_dir = os.path.join(filedir, "dispersion_h5")
        if not os.path.exists(ani_h5_dir):
            os.mkdir(ani_h5_dir)
        ani_h5_sim.save(os.path.join(ani_h5_dir, outname), writer=ani_writer_h5, dpi=180)

        f_pts.close()
        del f_pts
    print("\nFinished.")
    f_grid_h5.close()
    del f_grid_h5
    f_v_h5.close()
    del f_v_h5
    f_u_h5.close()
    del f_u_h5
