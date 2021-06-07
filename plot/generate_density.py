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
# from multivariate_colour_palettes import colour_scales

# convert output images via: convert <subfolder>/<pi|u|u>-*.png -trim -channel Gray <subfolder>/<pi|u|u>-%d.tif

#convert <subfolder>/<pi|u|v>-*.tif <subfolder>/<pi|u|u>-%d.tif

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

if __name__ == "__main__":
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-d", "--filedir", dest="filedir", type=str,
                        default="/var/scratch/experiments/NNvsGeostatistics/data",
                        help="head directory containing all input data and also are the store target for output files")
    parser.add_argument("-dt", "--deltat", dest="dt", type=float, default=1.0, help="output period in days")
    args = parser.parse_args()

    # filedir = '/var/scratch/experiments/NNvsGeostatistics/data/2'
    filedir = args.filedir
    anim_dt = args.dt
    Writer = writers['imagemagick_file']
    ani_writer_h5_sim = Writer()
    ani_writer_h5_u = Writer()
    ani_writer_h5_v = Writer()
    psize = 10.0
    ls = psize * 0.05
    alpha = 0.05
    zorder = 1
    # Pn = 2048
    Pn = 65365

    # ==== Load flow-field data ==== #
    f_grid_h5 = h5py.File(os.path.join(filedir, 'grid.h5'), "r")
    fT_h5 = f_grid_h5['times']
    fT_h5_attrs = fT_h5.attrs
    fX_h5 = f_grid_h5['longitude']
    fX_h5_attrs = fX_h5.attrs
    fY_h5 = f_grid_h5['latitude']
    fY_h5_attrs = fY_h5.attrs
    extends = (float(fX_h5_attrs['min']), float(fX_h5_attrs['max']), float(fY_h5_attrs['min']), float(fY_h5_attrs['max']))
    fT_0_h5 = fT_h5[0]
    fT_1_h5 = fT_h5[1]

    f_density = h5py.File(os.path.join(filedir, 'density.h5'), "r")
    fD_h5 = f_density['density']
    fD_h5_attrs = fD_h5.attrs
    fD_ext = (fD_h5_attrs['min'], fD_h5_attrs['max'])

    total_items = fT_h5.shape[0]
    tN = fT_h5.shape[0]
    sec_per_day = 86400.0

    tN = fT_h5.shape[0]
    print("tN: {}".format(tN))

    time_since_release_h5 = fT_h5
    sim_dt_h5 = fT_1_h5 - fT_0_h5
    print("dt = {}".format(sim_dt_h5))
    sim_dt_d_h5 = sim_dt_h5 / sec_per_day

    fig_anim_h5, ax_anim_h5 = plt.subplots(1, 1, figsize=(10, 8), frameon=False)  #, constrained_layout=True
    ax_anim_h5.set_facecolor('black')
    fig_anim_h5.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax_anim_h5.axis('off')
    ax_anim_h5.set_xlim([extends[0], extends[1]])
    ax_anim_h5.set_ylim([extends[2], extends[3]])

    cs_h5a_D = ax_anim_h5.imshow(fD_h5[0], extent=extends, interpolation='bilinear', zorder=0, origin='lower', aspect='equal',
                                      cmap='gray',
                                      norm=colors.Normalize(vmin=fD_ext[0], vmax=fD_ext[1]), animated=True)

    ax_anim_h5.axis('off')

    plt.savefig(os.path.join(filedir, "density_ref.png"), dpi=300)

    dt_anim = anim_dt * sec_per_day
    frames_anim = int(tN * (sim_dt_d_h5 / anim_dt))
    total_items = frames_anim

    def init_h5_D_animation():
        cs_h5a_D.set_array(fD_h5[0])

        results = []
        results.append(cs_h5a_D)
        ax_anim_h5.axis('off')
        return results


    def update_flow_only_h5_D(frames, *args):
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

        fD_show = (1.0-tt)*fD_h5[ti0] + tt*fD_h5[ti1]
        cs_h5a_D.set_array(fD_show)

        # == collected particles == #
        results = []
        results.append(cs_h5a_D)
        ax_anim_h5.axis('off')

        current_item = frames
        workdone = current_item / total_items
        print("\rProgress - plotting u-velocity: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50),
                                                                                     workdone * 100), end="", flush=True)
        return results


    ani_h5_u = FuncAnimation(fig_anim_h5, update_flow_only_h5_D, init_func=init_h5_D_animation, frames=frames_anim,
                             interval=1,
                             fargs=[dt_anim, ])
    ani_h5_dir_u = os.path.join(filedir, "density_image")
    if not os.path.exists(ani_h5_dir_u):
        os.mkdir(ani_h5_dir_u)
    ani_h5_u.save(os.path.join(ani_h5_dir_u, "D.png"), writer=ani_writer_h5_u, dpi=180)



    f_density.close()
    del f_density
    print("\nFinished.")
    f_grid_h5.close()
    del f_grid_h5

