# Script generates a particle image (pi) for neural network training from particle sets
import os
import h5py
import math
import numpy as np
from datetime import timedelta
from argparse import ArgumentParser
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib import cm
from matplotlib.colors import ListedColormap

GLOBAL_MIN_DIST = 0.00005

# convert output images via: convert <subfolder>/<pi|u|u>-*.png -trim -channel Gray <subfolder>/<pi|u|u>-%d.tif

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

if __name__ == "__main__":
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-d", "--filedir", dest="filedir", type=str,
                        default="/var/scratch/experiments/NNvsGeostatistics/data",
                        help="head directory containing all input data and also are the store target for output files")
    parser.add_argument("-dt", "--deltat", dest="dt", type=float, default=1.0, help="output period in days")
    args = parser.parse_args()

    # Example for filedir: '/var/scratch/experiments/NNvsGeostatistics/data/2'
    filedir = args.filedir
    anim_dt = args.dt
    Writer = writers['imagemagick_file']
    ani_writer_h5_sim = Writer()
    ani_writer_h5_u = Writer()
    ani_writer_h5_v = Writer()
    psize = 10.0 # to be modified to shrink or enlargen individual particles
    ls = psize * 0.05 # better leave fixed - line size relative to circle size
    alpha = 0.05 # to be modified for higher- or lower alpha saturation
    zorder = 1 # fixed / const
    Pn = 65365  # to be modfied to sample more or less particles

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
    fU_h5 = f_u_h5['uo']
    fU_h5_attrs = fU_h5.attrs
    max_u_value = np.maximum(np.abs(fU_h5_attrs['min']), np.abs(fU_h5_attrs['max']))
    fu_ext = (-max_u_value, +max_u_value)
    f_v_h5 = h5py.File(os.path.join(filedir, 'hydrodynamic_V.h5'), "r")
    fV_h5 = f_v_h5['vo']
    fV_h5_attrs = fV_h5.attrs
    max_v_value = np.maximum(np.abs(fV_h5_attrs['min']), np.abs(fV_h5_attrs['max']))
    fv_ext = (-max_v_value, +max_v_value)

    total_items = fT_h5.shape[0]
    tN = fT_h5.shape[0]

    f_pts = h5py.File(os.path.join(filedir, 'particles.h5'), "r")
    pT_h5 = f_pts['p_t']  # [()]
    pT_0_h5 = np.array(pT_h5[:, 0])
    pT_1_h5 = np.array(pT_h5[:, 1])
    sec_per_day = 86400.0

    N = pT_h5.shape[0]
    tN = pT_h5.shape[1]
    print("N: {}; Pn: {}; tN: {}".format(N, Pn, tN))

    time_since_release_h5 = pT_h5
    sim_dt_h5 = np.nanmax(pT_1_h5) - np.nanmax(pT_0_h5)
    print("dt = {}".format(sim_dt_h5))
    pX_h5 = f_pts['p_x']
    pY_h5 = f_pts['p_y']
    pZ_h5 = f_pts['p_z']
    sim_dt_d_h5 = sim_dt_h5 / sec_per_day

    const_blend_value = np.ones(N, dtype=np.float32)  # * 0.1
    indices = np.random.randint(0, N - 1, Pn, dtype=int)
    px_t_0 = np.array(pX_h5[:, 0])
    px_t_0 = np.take_along_axis(px_t_0, indices, axis=0)
    py_t_0 = np.array(pY_h5[:, 0])
    py_t_0 = np.take_along_axis(py_t_0, indices, axis=0)
    pd_t_0 = np.array(const_blend_value[:])
    pd_t_0 = np.take_along_axis(pd_t_0, indices, axis=0)

    fig_anim_h5, ax_anim_h5 = plt.subplots(1, 1, figsize=(10, 8), frameon=False)
    ax_anim_h5.set_facecolor('black')
    fig_anim_h5.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax_anim_h5.axis('off')
    ax_anim_h5.set_xlim([extends[0], extends[1]])
    ax_anim_h5.set_ylim([extends[2], extends[3]])

    cs_nca_bg = ax_anim_h5.imshow(np.zeros(fU_h5[0].shape), extent=extends, cmap=ListedColormap(np.array([[0, 0, 0, 1.0], ] * 256, dtype=np.float32), name='black_solid'), interpolation='bilinear')  # black background
    cs_h5a_pts_coll = ax_anim_h5.scatter(px_t_0, py_t_0, s=psize, linewidths=ls, c=pd_t_0,
                                  cmap='gray',
                                  norm=colors.Normalize(vmin=0., vmax=1.),
                                  transform=ax_anim_h5.transData, zorder=2, alpha=alpha)
    ax_anim_h5.add_collection(cs_h5a_pts_coll)
    ax_anim_h5.axis('off')

    dt_anim = anim_dt * sec_per_day
    frames_anim = int(tN * (sim_dt_d_h5 / anim_dt))
    total_items = frames_anim


    def init_h5_animation():
        px_t_0 = np.array(pX_h5[:, 0])
        px_t_0 = np.take_along_axis(px_t_0, indices, axis=0)
        py_t_0 = np.array(pY_h5[:, 0])
        py_t_0 = np.take_along_axis(py_t_0, indices, axis=0)
        pd_t_0 = np.array(const_blend_value[:])
        pd_t_0 = np.take_along_axis(pd_t_0, indices, axis=0)
        cs_h5a_pts_coll.set_array(pd_t_0)  # sets the values (e.g. lifetime) interpreted by the colour map
        offsets = np.column_stack((px_t_0, py_t_0))
        cs_h5a_pts_coll.set_offsets(offsets)  # sets the new particle positions

        results = []
        results.append(cs_h5a_pts_coll)
        ax_anim_h5.axis('off')
        return results


    def update_flow_only_h5(frames, *args):
        dt = args[0]
        tx = np.float64(frames) * dt
        tx = np.fmod(tx, fT_h5[-1])
        ti0 = time_index_value(tx, fT_h5)
        tt = time_partion_value(tx, fT_h5)
        ti1 = 0
        if ti0 < (len(fT_h5) - 1):
            ti1 = ti0 + 1
        else:
            ti1 = 0

        px_t_0 = np.array(pX_h5[:, ti0])
        px_t_0 = np.take_along_axis(px_t_0, indices, axis=0)
        py_t_0 = np.array(pY_h5[:, ti0])
        py_t_0 = np.take_along_axis(py_t_0, indices, axis=0)
        pd_t_0 = np.array(const_blend_value[:])
        pd_t_0 = np.take_along_axis(pd_t_0, indices, axis=0)
        px_t_1 = np.array(pX_h5[:, ti1])
        px_t_1 = np.take_along_axis(px_t_1, indices, axis=0)
        py_t_1 = np.array(pY_h5[:, ti1])
        py_t_1 = np.take_along_axis(py_t_1, indices, axis=0)
        px_show = (1.0 - tt) * px_t_0 + tt * px_t_1
        py_show = (1.0 - tt) * py_t_0 + tt * py_t_1
        pd_show = pd_t_0
        cs_h5a_pts_coll.set_array(pd_show)  # sets the values (e.g. lifetime) interpreted by the colour map
        offsets = np.column_stack((px_show, py_show))
        cs_h5a_pts_coll.set_offsets(offsets)  # sets the new particle positions
        # == collected particles == #
        results = []
        results.append(cs_h5a_pts_coll)
        ax_anim_h5.axis('off')

        del px_t_0
        del px_t_1
        del py_t_0
        del py_t_1
        del pd_t_0

        current_item = frames
        workdone = current_item / total_items
        print("\rProgress - plotting particle image: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50),
                                                                                     workdone * 100), end="", flush=True)
        return results


    ani_h5_sim = FuncAnimation(fig_anim_h5, update_flow_only_h5, init_func=init_h5_animation, frames=frames_anim,
                               interval=1, fargs=[dt_anim, ])
    ani_h5_dir = os.path.join(filedir, "particle_image")
    if not os.path.exists(ani_h5_dir):
        os.mkdir(ani_h5_dir)
    ani_h5_sim.save(os.path.join(ani_h5_dir, "pi.png"), writer=ani_writer_h5_sim, dpi=180)




    fig_anim_u, ax_anim_u = plt.subplots(1, 1, figsize=(10, 8), frameon=False)
    ax_anim_u.set_facecolor('black')
    fig_anim_u.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax_anim_u.axis('off')
    ax_anim_u.set_xlim([extends[0], extends[1]])
    ax_anim_u.set_ylim([extends[2], extends[3]])

    cs_h5a_u = ax_anim_u.imshow(fU_h5[0], extent=extends,
                                 cmap='gray',
                                 norm=colors.Normalize(vmin=fu_ext[0], vmax=fu_ext[1]),
                                 animated=True, zorder=0, origin='lower', interpolation='bilinear', aspect='auto')
    ax_anim_u.axis('off')

    def init_h5_u_animation():
        cs_h5a_u.set_array(fU_h5[0])
        results = []
        results.append(cs_h5a_u)
        ax_anim_u.axis('off')
        return results


    def update_flow_only_h5_u(frames, *args):
        # if (frames % 10) == 0:
        #     print("Plotting frame {} of {} ...".format(frames, tN))
        dt = args[0]
        tx = np.float64(frames) * dt
        tx = np.fmod(tx, fT_h5[-1])
        ti0 = time_index_value(tx, fT_h5)
        tt = time_partion_value(tx, fT_h5)
        ti1 = 0
        if ti0 < (len(fT_h5) - 1):
            ti1 = ti0 + 1
        else:
            ti1 = 0
        # print("Processing tx={}, tt={} ti0={} and ti1 = {}".format(tx, tt, ti0, ti1))

        fu_show = (1.0-tt)*fU_h5[ti0] + tt*fU_h5[ti1]
        cs_h5a_u.set_array(fu_show)
        results = []
        results.append(cs_h5a_u)
        ax_anim_u.axis('off')

        current_item = frames
        workdone = current_item / total_items
        print("\rProgress - plotting u-velocity: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50),
                                                                                     workdone * 100), end="", flush=True)
        return results


    ani_h5_u = FuncAnimation(fig_anim_u, update_flow_only_h5_u, init_func=init_h5_u_animation, frames=frames_anim,
                               interval=1, fargs=[dt_anim, ])
    ani_h5_dir_u = os.path.join(filedir, "Udata")
    if not os.path.exists(ani_h5_dir_u):
        os.mkdir(ani_h5_dir_u)
    ani_h5_u.save(os.path.join(ani_h5_dir_u, "u.png"), writer=ani_writer_h5_u, dpi=180)




    fig_anim_v, ax_anim_v = plt.subplots(1, 1, figsize=(10, 8), frameon=False)
    ax_anim_v.set_facecolor('black')
    fig_anim_v.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax_anim_v.axis('off')
    ax_anim_v.set_xlim([extends[0], extends[1]])
    ax_anim_v.set_ylim([extends[2], extends[3]])

    cs_h5a_v = ax_anim_v.imshow(fV_h5[0], extent=extends,
                                 cmap='gray',
                                 norm=colors.Normalize(vmin=fv_ext[0], vmax=fv_ext[1]),
                                 animated=True, zorder=0, origin='lower', interpolation='bilinear', aspect='auto')
    ax_anim_v.axis('off')

    def init_h5_v_animation():
        cs_h5a_v.set_array(fV_h5[0])
        results = []
        results.append(cs_h5a_v)
        ax_anim_v.axis('off')
        return results


    def update_flow_only_h5_v(frames, *args):
        # if (frames % 10) == 0:
        #     print("Plotting frame {} of {} ...".format(frames, tN))
        dt = args[0]
        tx = np.float64(frames) * dt
        tx = np.fmod(tx, fT_h5[-1])
        ti0 = time_index_value(tx, fT_h5)
        tt = time_partion_value(tx, fT_h5)
        ti1 = 0
        if ti0 < (len(fT_h5) - 1):
            ti1 = ti0 + 1
        else:
            ti1 = 0
        # print("Processing tx={}, tt={} ti0={} and ti1 = {}".format(tx, tt, ti0, ti1))

        fv_show = (1.0-tt)*fV_h5[ti0] + tt*fV_h5[ti1]
        cs_h5a_v.set_array(fv_show)

        results = []
        results.append(cs_h5a_v)
        ax_anim_v.axis('off')

        current_item = frames
        workdone = current_item / total_items
        print("\rProgress - plotting v-velocity: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50),
                                                                                     workdone * 100), end="", flush=True)
        return results


    ani_h5_v = FuncAnimation(fig_anim_v, update_flow_only_h5_v, init_func=init_h5_v_animation, frames=frames_anim,
                               interval=1, fargs=[dt_anim, ])
    ani_h5_dir_v = os.path.join(filedir, "Vdata")
    if not os.path.exists(ani_h5_dir_v):
        os.mkdir(ani_h5_dir_v)
    ani_h5_v.save(os.path.join(ani_h5_dir_v, "v.png"), writer=ani_writer_h5_v, dpi=180)




    f_pts.close()
    del f_pts
    print("\nFinished.")
    f_grid_h5.close()
    del f_grid_h5
    f_v_h5.close()
    del f_v_h5
    f_u_h5.close()
    del f_u_h5

