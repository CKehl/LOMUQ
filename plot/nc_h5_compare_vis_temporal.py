import os
import h5py
import math
from scipy.io import netcdf
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib import colors
from matplotlib.colors import colorConverter
from matplotlib.animation import FuncAnimation, writers
from multivariate_colour_palettes import colour_scales
from DecayLine import DecayLine
from datetime import timedelta
from argparse import ArgumentParser


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


if __name__ == '__main__':
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-d", "--filedir", dest="filedir", type=str,
                        default="/var/scratch/experiments/NNvsGeostatistics/data",
                        help="head directory containing all input data and also are the store target for output files")
    parser.add_argument("-A", "--animate", dest="animate", action='store_true', default=False,
                        help="Generate particle animations (default: False).")
    args = parser.parse_args()

    # filedir = '/var/scratch/experiments/NNvsGeostatistics/data/2'
    filedir = args.filedir
    animate = args.animate
    write_nc = False
    Writer = writers['imagemagick_file']
    ani_writer_h5 = Writer()
    ani_writer_nc = Writer()

    # fT_nc = None
    # fX_nc = None
    # fY_nc = None
    # fU_nc = None
    # fV_nc = None
    if write_nc:
        # ==== Load flow-field data ==== #
        f_u_nc = xr.open_dataset(os.path.join(filedir, 'doublegyreU.nc'), decode_cf=True, engine='netcdf4')
        fT_nc = f_u_nc['time_counter']
        fX_nc = f_u_nc['x']
        fY_nc = f_u_nc['y']
        fU_nc = f_u_nc['vozocrtx']
        max_u_value = max(abs(float(fU_nc.min())), abs(float(fU_nc.max())))
        fu_ext = (-max_u_value, +max_u_value)
        # f_u.close()
        # del f_u

        f_v_nc = xr.open_dataset(os.path.join(filedir, 'doublegyreV.nc'), decode_cf=True, engine='netcdf4')
        fV_nc = f_v_nc['vomecrty']
        extends = (float(fX_nc.min()), float(fX_nc.max()), float(fY_nc.min()), float(fY_nc.max()))
        max_v_value = max(abs(float(fV_nc.min())), abs(float(fV_nc.max())))
        fv_ext = (-max_v_value, +max_v_value)
        # f_v.close()
        # del f_v

        f_velmag_ext_nc = (0, np.sqrt(max_v_value**2 + max_u_value**2))
        di = 0  # depth index - fixed to zero cause data are 2D
        total_items = fT_nc.shape[0]

        speed_nc_ti = fU_nc[0, di] ** 2 + fV_nc[0, di] ** 2
        speed_nc_ti = np.where(speed_nc_ti > 0, np.sqrt(speed_nc_ti), 0)

        u_nc_dir = os.path.join(filedir, "U_nc")
        if not os.path.exists(u_nc_dir):
                os.mkdir(u_nc_dir)
        v_nc_dir = os.path.join(filedir, "V_nc")
        if not os.path.exists(v_nc_dir):
                os.mkdir(v_nc_dir)
        velmag_nc_dir = os.path.join(filedir, "VelMag_nc")
        if not os.path.exists(velmag_nc_dir):
                os.mkdir(velmag_nc_dir)

        fig_nc_single_u, ax_nc_single_u = plt.subplots(1, 1)
        ax_nc_single_u.set_facecolor('grey')
        cs_ncs_u = plt.imshow(fU_nc[0, di], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['U']['colour_scale'], interpolation='bilinear', vmin=fu_ext[0], vmax=fu_ext[1])
        cbar_ncs_u = fig_nc_single_u.add_axes([0.0, 0.9, 0.05, 0.07])
        plt.colorbar(cs_ncs_u, cax=cbar_ncs_u)

        fig_nc_single_v, ax_nc_single_v = plt.subplots(1, 1)
        ax_nc_single_v.set_facecolor('grey')
        cs_ncs_v = plt.imshow(fV_nc[0, di], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['V']['colour_scale'], interpolation='bilinear', vmin=fv_ext[0], vmax=fv_ext[1])
        cbar_ncs_v = fig_nc_single_v.add_axes([0.0, 0.9, 0.05, 0.07])
        plt.colorbar(cs_ncs_v, cax=cbar_ncs_v)

        fig_nc_single_velmag, ax_nc_single_velmag = plt.subplots(1, 1)
        ax_nc_single_velmag.set_facecolor('black')
        cs_ncs_velmag = plt.imshow(speed_nc_ti, extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['VelMag']['colour_scale'], interpolation='bilinear', vmin=f_velmag_ext_nc[0], vmax=f_velmag_ext_nc[1])
        cbar_ncs_velmag = fig_nc_single_velmag.add_axes([0.0, 0.9, 0.05, 0.07])
        plt.colorbar(cs_ncs_velmag, cax=cbar_ncs_velmag)

        print("\nPlotting NetCDF U/V data ...")
        for ti in range(fT_nc.shape[0]):
            speed_nc_ti = fU_nc[ti, di] ** 2 + fV_nc[ti, di] ** 2
            speed_nc_ti = np.where(speed_nc_ti > 0, np.sqrt(speed_nc_ti), 0)

            cs_ncs_u.set_data(fU_nc[ti, di])
            fig_nc_single_u.canvas.flush_events()  # .draw()
            u_fig_fname = "u_nc_%s.png" % (str(ti))
            plt.savefig(os.path.join(u_nc_dir, u_fig_fname), dpi=180)

            cs_ncs_v.set_data(fV_nc[ti, di])
            fig_nc_single_v.canvas.flush_events()  # .draw()
            v_fig_fname = "v_nc_%s.png" % (str(ti))
            plt.savefig(os.path.join(v_nc_dir, v_fig_fname), dpi=180)

            cs_ncs_velmag.set_data(speed_nc_ti)
            fig_nc_single_velmag.canvas.flush_events()  # .draw()
            velmag_fig_fname = "velmag_nc_%s.png" % (str(ti))
            plt.savefig(os.path.join(velmag_nc_dir, velmag_fig_fname), dpi=180)

            current_item = ti
            workdone = current_item / total_items
            print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
        print("\nFinished plotting NetCDF U/V plots.")

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
    # f_u_h5.close()
    # del f_u_h5
    f_v_h5 = h5py.File(os.path.join(filedir, 'hydrodynamic_V.h5'), "r")
    fV_h5 = f_v_h5['vo']  # [()]
    fV_h5_attrs = fV_h5.attrs
    max_v_value = np.maximum(np.abs(fV_h5_attrs['min']), np.abs(fV_h5_attrs['max']))
    fv_ext = (-max_v_value, +max_v_value)
    # f_v_h5.close()
    # del f_v_h5

    f_velmag_ext_h5 = (0, np.sqrt(max_v_value**2 + max_u_value**2))

    total_items = fT_h5.shape[0]
    tN = fT_h5.shape[0]

    u_h5_dir = os.path.join(filedir, "U_h5")
    if not os.path.exists(u_h5_dir):
        os.mkdir(u_h5_dir)
    v_h5_dir = os.path.join(filedir, "V_h5")
    if not os.path.exists(v_h5_dir):
        os.mkdir(v_h5_dir)
    velmag_h5_dir = os.path.join(filedir, "VelMag_h5")
    if not os.path.exists(velmag_h5_dir):
        os.mkdir(velmag_h5_dir)

    speed_h5_ti = fU_h5[0] ** 2 + fV_h5[0] ** 2
    speed_h5_ti = np.where(speed_h5_ti > 0, np.sqrt(speed_h5_ti), 0)

    fig_h5_single_u, ax_h5_single_u = plt.subplots(1, 1)
    ax_h5_single_u.set_facecolor('grey')
    cs_h5s_u = plt.imshow(fU_h5[0], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['U']['colour_scale'], interpolation='bilinear', vmin=fu_ext[0], vmax=fu_ext[1], origin='lower')
    cbar_h5s_u = fig_h5_single_u.add_axes([0.0, 0.9, 0.05, 0.07])
    plt.colorbar(cs_h5s_u, cax=cbar_h5s_u)

    fig_h5_single_v, ax_h5_single_v = plt.subplots(1, 1)
    ax_h5_single_v.set_facecolor('grey')
    cs_h5s_v = plt.imshow(fV_h5[0], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['V']['colour_scale'], interpolation='bilinear', vmin=fv_ext[0], vmax=fv_ext[1], origin='lower')  # , alpha=0.95
    cbar_h5s_v = fig_h5_single_v.add_axes([0.0, 0.9, 0.05, 0.07])
    plt.colorbar(cs_h5s_v, cax=cbar_h5s_v)

    fig_h5_single_velmag, ax_h5_single_velmag = plt.subplots(1, 1)
    ax_h5_single_velmag.set_facecolor('black')
    cs_h5s_velmag = plt.imshow(speed_h5_ti, extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['VelMag']['colour_scale'], interpolation='bilinear', vmin=f_velmag_ext_h5[0], vmax=f_velmag_ext_h5[1], origin='lower')
    cbar_h5s_velmag = fig_h5_single_velmag.add_axes([0.0, 0.9, 0.05, 0.07])
    plt.colorbar(cs_h5s_velmag, cax=cbar_h5s_velmag)

    print("\nPlotting HDF5 U/V data ...")
    # for ti in range(fT_nc.shape[0]):
    #     speed_h5_ti = fU_h5[ti] ** 2 + fU_h5[ti] ** 2
    #     speed_h5_ti = np.where(speed_h5_ti > 0, np.sqrt(speed_h5_ti), 0)
    #
    #     cs_h5s_u.set_data(fU_h5[ti])
    #     fig_h5_single_u.canvas.draw()
    #     u_fig_fname = "u_h5_%s.png" % (str(ti))
    #     plt.savefig(os.path.join(u_h5_dir, u_fig_fname), dpi=180)
    #
    #     cs_h5s_v.set_data(fV_h5[ti])
    #     fig_h5_single_v.canvas.draw()
    #     v_fig_fname = "v_h5_%s.png" % (str(ti))
    #     plt.savefig(os.path.join(v_h5_dir, v_fig_fname), dpi=180)
    #
    #     cs_h5s_velmag.set_data(speed_h5_ti)
    #     fig_h5_single_velmag.canvas.draw()
    #     velmag_fig_fname = "velmag_h5_%s.png" % (str(ti))
    #     plt.savefig(os.path.join(velmag_h5_dir, velmag_fig_fname), dpi=180)
    #
    #     current_item = ti
    #     workdone = current_item / total_items
    #     print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)



    def init_U_h5_animation():
        cs_h5s_u.set_data(fU_h5[0])
        results = [cs_h5s_u, ]
        return results

    def update_U_h5_ani(frames, *args):
        ti = frames
        cs_h5s_u.set_data(fU_h5[ti])
        results = [cs_h5s_u, ]
        current_item = ti
        workdone = current_item / total_items
        print("\rProgress - plot U data: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
        return results

    ani_h5_u = FuncAnimation(fig_h5_single_u, update_U_h5_ani, init_func=init_U_h5_animation, frames=tN, interval=1, fargs=[])
    ani_h5_u.save(os.path.join(u_h5_dir, 'u_h5.png'), writer=ani_writer_h5, dpi=180)

    def init_V_h5_animation():
        cs_h5s_v.set_data(fV_h5[0])
        results = [cs_h5s_v, ]
        return results

    def update_V_h5_ani(frames, *args):
        ti = frames
        cs_h5s_v.set_data(fV_h5[ti])
        results = [cs_h5s_v, ]
        current_item = ti
        workdone = current_item / total_items
        print("\rProgress - plot V data: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
        return results

    ani_h5_v = FuncAnimation(fig_h5_single_v, update_V_h5_ani, init_func=init_V_h5_animation, frames=tN, interval=1, fargs=[])
    ani_h5_v.save(os.path.join(v_h5_dir, 'v_h5.png'), writer=ani_writer_h5, dpi=180)

    def init_VelMag_h5_animation():
        speed_h5_ti = fU_h5[0] ** 2 + fU_h5[0] ** 2
        speed_h5_ti = np.where(speed_h5_ti > 0, np.sqrt(speed_h5_ti), 0)
        cs_h5s_velmag.set_data(speed_h5_ti)
        results = [cs_h5s_velmag, ]
        return results

    def update_VelMag_h5_ani(frames, *args):
        ti = frames
        speed_h5_ti = fU_h5[ti] ** 2 + fU_h5[ti] ** 2
        speed_h5_ti = np.where(speed_h5_ti > 0, np.sqrt(speed_h5_ti), 0)
        cs_h5s_velmag.set_data(speed_h5_ti)
        results = [cs_h5s_velmag, ]
        current_item = ti
        workdone = current_item / total_items
        print("\rProgress - plot velocity magnitude data: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
        return results

    ani_h5_velmag = FuncAnimation(fig_h5_single_velmag, update_VelMag_h5_ani, init_func=init_VelMag_h5_animation, frames=tN, interval=1, fargs=[])
    ani_h5_velmag.save(os.path.join(velmag_h5_dir, 'velmag_h5.png'), writer=ani_writer_h5, dpi=180)

    print("\nFinished plotting HDF5 U/V plots.")


    Pn = 96
    indices = None  # np.random.randint(0, 100, Pn, dtype=int)
    # ============================================== #
    # ==== Particle Trails - Animation - NetCDF ==== #
    # ============================================== #
    if animate and write_nc:
        # data_xarray = xr.open_dataset('/var/scratch/experiments/NNvsGeostatistics/data/benchmark_perlin_noMPI_n4096_p_fwd.nc')
        # data_xarray = xr.open_dataset('/var/scratch/experiments/NNvsGeostatistics/data/benchmark_bickleyjet_noMPI_p_n4096_fwd.nc')
        data_xarray = xr.open_dataset(os.path.join(filedir, 'benchmark_doublegyre_noMPI_p_n4096_fwd.nc'))
        np.set_printoptions(linewidth=160)
        ns_per_sec = np.timedelta64(1, 's')  # nanoseconds in an sec
        sec_per_day = 86400.0
        time_array = data_xarray['time'].data / ns_per_sec

        N = data_xarray['lon'].shape[0]
        tN = data_xarray['lon'].shape[1]
        indices = np.random.randint(0, N - 1, Pn, dtype=int)

        time_since_release_nc = (
                    time_array.transpose() - time_array[:, 0])  # substract the initial time from each timeseri
        sim_dt_nc = time_since_release_nc[1, 0] - time_since_release_nc[0, 0]
        print("dt = {}".format(sim_dt_nc))
        pX_nc = data_xarray['lon']  # [indices, :]  # only plot the first 32 particles
        pY_nc = data_xarray['lat']  # [indices, :]  # only plot the first 32 particles

        speed_nc_0 = fU_nc[0, di] ** 2 + fV_nc[0, di] ** 2
        speed_nc_0 = np.where(speed_nc_0 > 0, np.sqrt(speed_nc_0), 0)

        # speed_map = cm.get_cmap('twilight', 256)
        fig_anim_nc = plt.figure()
        ax_anim_nc = plt.axes()
        cs_nca_velmag = ax_anim_nc.pcolormesh(fX_nc, fY_nc, speed_nc_0,
                                              cmap=colour_scales['blue_quadrant_palette_blackbase']['VelMag'][
                                                  'colour_scale'], vmin=f_velmag_ext_nc[0], vmax=f_velmag_ext_nc[1],
                                              shading='gouraud', zorder=1)
        # cs_nca_bg = plt.imshow(np.zeros(fU_nc[0, 0].shape), extent=extends, cmap=ListedColormap(np.array([[0, 0, 0, 1.0], ] * 256, dtype=np.float32), name='black_solid'), interpolation='bilinear')  # black background
        # cs_nca_u = plt.imshow(fU_nc[0, 0], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['U']['colour_scale'], interpolation='bilinear', vmin=fu_ext[0], vmax=fu_ext[1])  # , alpha=0.95
        # cs_nca_v = plt.imshow(fV_nc[0, 0], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['V']['colour_scale'], interpolation='bilinear', vmin=fv_ext[0], vmax=fv_ext[1])  # , alpha=0.95
        # cs_nca_velmag = plt.imshow(speed_nc[0, 0], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['VelMag']['colour_scale'], interpolation='bilinear', alpha=0.3)
        # cbar_nca_velmag = fig_anim_nc.add_axes([0, 0.015, 0.1, 0.1])
        cbar_nca_velmag = fig_anim_nc.add_axes([0.0, 0.9, 0.05, 0.07])
        plt.colorbar(cs_nca_velmag, cax=cbar_nca_velmag)
        ax_anim_nc.set_title("Simulation - NetCDF data - t = %5.1f d" % (time_since_release_nc[0, 0] / sec_per_day))
        lines_nc = []
        for i in range(0, Pn):
            # lines.append(DecayLine(14, 8, [1.0, 0.625, 0.], zorder=2 + i))
            lines_nc.append(DecayLine(14, 8, [0.0, 0.625, 0.0], zorder=2 + i))

        for i in range(0, Pn):
            lines_nc[i].add_point(pX_nc[indices[i], 0], pY_nc[indices[i], 0])

        for l in lines_nc:
            ax_anim_nc.add_collection(l.get_LineCollection())


        def init_nc_animation():
            # cs_nca_u.set_array(fU_nc[0, 0])
            # cs_nca_v.set_array(fV_nc[0, 0])
            cs_nca_velmag.set_array(speed_nc_0)
            # results = [cs_nca_u, cs_nca_v, ]
            results = []
            results.append(cs_nca_velmag)
            for l in lines_nc:
                results.append(l.get_LineCollection())
            ax_anim_nc.set_title("Simulation t = %10.4f s" % (time_since_release_nc[0, 0]))
            return results


        def update_flow_only_nc(frames, *args):
            if (frames % 10) == 0:
                print("Plotting frame {} of {} ...".format(frames, tN))
            dt = args[0]
            tx = float(frames) * dt
            tx = math.fmod(tx, fT_nc[-1])
            ti0 = time_index_value(tx, fT_nc)
            tt = time_partion_value(tx, fT_nc)
            ti1 = 0
            if ti0 < (len(fT_nc) - 1):
                ti1 = ti0 + 1
            else:
                ti1 = 0
            speed_1 = fU_nc[ti0, di] ** 2 + fV_nc[ti0, di] ** 2
            speed_1 = np.where(speed_1 > 0, np.sqrt(speed_1), 0)
            speed_2 = fU_nc[ti1, di] ** 2 + fV_nc[ti1, di] ** 2
            speed_2 = np.where(speed_2 > 0, np.sqrt(speed_2), 0)
            # fu_show = tt*fU_nc[ti0][0] + (1.0-tt)*fU_nc[ti1][0]
            # fv_show = tt*fV_nc[ti0][0] + (1.0-tt)*fV_nc[ti1][0]
            fs_show = (1.0 - tt) * speed_1 + tt * speed_2
            # cs_nca_u.set_array(fu_show)
            # cs_nca_v.set_array(fv_show)
            cs_nca_velmag.set_array(fs_show)
            # == add new lines == #
            if frames > 0:
                for pindex in range(0, Pn):
                    # pt = sim[pindex].point(frames)
                    # pi_lon = data_xarray['lon'].data[indices[i], frames]
                    # pi_lat = data_xarray['lat'].data[indices[i], frames]
                    lines_nc[pindex].add_point(pX_nc[indices[pindex], frames], pY_nc[indices[pindex], frames])
            # == collect results == #
            # results = [cs_nca_u, cs_nca_v, ]
            results = []
            results.append(cs_nca_velmag)
            for l in lines_nc:
                results.append(l.get_LineCollection())
            ax_anim_nc.set_title("Simulation - NetCDF data - t = %5.1f d" % (tx / sec_per_day))
            return results


        ##Writer = writers['ffmpeg']
        ##ani_writer = Writer(fps=25, bitrate=24000)
        # ani_writer = Writer()

        ##plt.show()
        ##ani = FuncAnimation(fig, update_flow_only, init_func=init, frames=sim_steps, interval=25, fargs=[dt,], blit=True)
        ##ani.save("flow_w_particles.mp4", writer=ani_writer)

        # sim_dt
        ani = FuncAnimation(fig_anim_nc, update_flow_only_nc, init_func=init_nc_animation, frames=tN, interval=1,
                            fargs=[sim_dt_nc, ])  # , save_count=1, cache_frame_data=False) #, blit=True)
        # ani.save("/var/scratch/experiments/NNvsGeostatistics/data/sim_imgs_nc/benchmark_perlin.png", writer=ani_writer, dpi=180)
        # ani.save("/var/scratch/experiments/NNvsGeostatistics/data/sim_imgs_nc/benchmark_bickleyjet.png", writer=ani_writer, dpi=180)
        ani_nc_dir = os.path.join(filedir, "sim_imgs_nc")
        if not os.path.exists(ani_nc_dir):
            os.mkdir(ani_nc_dir)
        ani.save(os.path.join(ani_nc_dir, 'benchmark_doublegyre.png'), writer=ani_writer_nc, dpi=180)

    # ============================================== #
    # ====  Particle Trails - Animation - HDF5  ==== #
    # ============================================== #
    if animate:
        # f_grid = h5py.File(os.path.join(filedir, 'grid.h5'), "r")
        # fT_h5 = f_grid["times"]  # [()]
        # print("fT_h5 = {}".format(fT_h5))
        # f_grid.close()
        # del f_grid
        f_pts = h5py.File(os.path.join(filedir, 'particles.h5'), "r")
        pT_h5 = f_pts['p_t']  # [()]
        sec_per_day = 86400.0

        N = pT_h5.shape[0]
        tN = pT_h5.shape[1]
        if indices is None:
            indices = np.random.randint(0, N - 1, Pn, dtype=int)

        time_since_release_h5 = pT_h5
        # time_since_release_h5 = (pT_h5.transpose() - pT_h5[:, 0])  # substract the initial time from each timeseries
        sim_dt_h5 = time_since_release_h5[0, 1] - time_since_release_h5[0, 0]
        print("dt = {}".format(sim_dt_h5))
        pX_h5 = f_pts['p_x']  # [()][indices, :]  # only plot the first 32 particles
        pY_h5 = f_pts['p_y']  # [()][indices, :]  # only plot the first 32 particles
        # f_pts.close()
        # del f_pts

        speed_h5_0 = fU_h5[0] ** 2 + fV_h5[0] ** 2
        speed_h5_0 = np.where(speed_h5_0 > 0, np.sqrt(speed_h5_0), 0)

        # speed_map = cm.get_cmap('twilight', 256)
        fig_anim_h5 = plt.figure()
        ax_anim_h5 = plt.axes()
        # cs_h5a_velmag = ax_anim_h5.pcolormesh(fX_h5, fY_h5, speed_nc[0, 0], cmap=colour_scales['blue_quadrant_palette_blackbase']['VelMag']['colour_scale'], vmin=fu_ext[0], vmax=fu_ext[1], shading='gouraud', zorder=1)
        # cs_nca_bg = plt.imshow(np.zeros(fU_h5[0].shape), extent=extends, cmap=ListedColormap(np.array([[0, 0, 0, 1.0], ] * 256, dtype=np.float32), name='black_solid'), interpolation='bilinear')  # black background
        # cs_nca_u = plt.imshow(fU_h5[0], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['U']['colour_scale'], interpolation='bilinear', vmin=fu_ext[0], vmax=fu_ext[1])  # , alpha=0.95
        # cs_nca_v = plt.imshow(fV_h5[0], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['V']['colour_scale'], interpolation='bilinear', vmin=fv_ext[0], vmax=fv_ext[1])  # , alpha=0.95
        cs_h5a_velmag = plt.imshow(speed_h5_0, extent=extends,
                                   cmap=colour_scales['blue_quadrant_palette_blackbase']['VelMag']['colour_scale'],
                                   interpolation='bilinear', norm=colors.Normalize(vmin=f_velmag_ext_h5[0],
                                                                                   vmax=f_velmag_ext_h5[1]), origin='lower')
        # cbar_nca_velmag = fig_anim_nc.add_axes([0, 0.015, 0.1, 0.1])
        cbar_h5a_velmag = fig_anim_h5.add_axes([0.0, 0.9, 0.05, 0.07])
        plt.colorbar(cs_h5a_velmag, cax=cbar_h5a_velmag)
        ax_anim_h5.set_title("Simulation - NetCDF data - t = %5.1f d" % (time_since_release_h5[0, 0] / sec_per_day))
        lines_h5 = []
        for i in range(0, Pn):
            # lines.append(DecayLine(14, 8, [1.0, 0.625, 0.], zorder=2 + i))
            lines_h5.append(DecayLine(14, 8, [0.0, 0.625, 0.0], zorder=2 + i))

        for i in range(0, Pn):
            lines_h5[i].add_point(pX_h5[indices[i], 0], pY_h5[indices[i], 0])

        for l in lines_h5:
            ax_anim_h5.add_collection(l.get_LineCollection())


        def init_h5_animation():
            # cs_nca_u.set_array(fU_nc[0, 0])
            # cs_nca_v.set_array(fV_nc[0, 0])
            cs_h5a_velmag.set_array(speed_h5_0)
            # results = [cs_nca_u, cs_nca_v, ]
            results = []
            results.append(cs_h5a_velmag)
            for l in lines_h5:
                results.append(l.get_LineCollection())
            ax_anim_h5.set_title("Simulation - NetCDF data - t = %5.1f d" % (time_since_release_h5[0, 0] / sec_per_day))
            return results


        def update_flow_only_h5(frames, *args):
            if (frames % 10) == 0:
                print("Plotting frame {} of {} ...".format(frames, tN))
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
            speed_1 = fU_h5[ti0] ** 2 + fV_h5[ti0] ** 2
            speed_1 = np.where(speed_1 > 0, np.sqrt(speed_1), 0)
            speed_2 = fU_h5[ti1] ** 2 + fV_h5[ti1] ** 2
            speed_2 = np.where(speed_2 > 0, np.sqrt(speed_2), 0)
            # fu_show = tt*fU_nc[ti0] + (1.0-tt)*fU_nc[ti1]
            # fv_show = tt*fV_nc[ti0] + (1.0-tt)*fV_nc[ti1]
            fs_show = (1.0 - tt) * speed_1 + tt * speed_2
            # cs_nca_u.set_array(fu_show)
            # cs_nca_v.set_array(fv_show)
            cs_h5a_velmag.set_array(fs_show)
            # == add new lines == #
            if frames > 0:
                for pindex in range(0, Pn):
                    lines_h5[pindex].add_point(pX_h5[indices[pindex], frames], pY_h5[indices[pindex], frames])
            # == collect results == #
            # results = [cs_nca_u, cs_nca_v, ]
            results = []
            results.append(cs_h5a_velmag)
            for l in lines_h5:
                results.append(l.get_LineCollection())
            ax_anim_h5.set_title("Simulation - h5 data - t = %5.1f d" % (tx / sec_per_day))
            return results


        # Writer = writers['imagemagick_file']
        # ani_writer_h5 = Writer()
        ani_h5_sim = FuncAnimation(fig_anim_h5, update_flow_only_h5, init_func=init_h5_animation, frames=tN, interval=1,
                               fargs=[sim_dt_h5, ])  # , save_count=1, cache_frame_data=False) #, blit=True)
        # ani.save("/var/scratch/experiments/NNvsGeostatistics/data/sim_imgs_nc/benchmark_perlin.png", writer=ani_writer, dpi=180)
        # ani.save("/var/scratch/experiments/NNvsGeostatistics/data/sim_imgs_nc/benchmark_bickleyjet.png", writer=ani_writer, dpi=180)
        ani_h5_dir = os.path.join(filedir, "sim_imgs_h5")
        if not os.path.exists(ani_h5_dir):
            os.mkdir(ani_h5_dir)
        ani_h5_sim.save(os.path.join(ani_h5_dir, 'benchmark_doublegyre.png'), writer=ani_writer_h5, dpi=180)

    # ================================= #
    # ==== Density - Image - HDF5  ==== #
    # ================================= #
    f_density = h5py.File(os.path.join(filedir, 'density.h5'), "r")
    fD_h5 = f_density['density']
    fD_h5_attrs = fD_h5.attrs
    fD_ext = (fD_h5_attrs['min'] + np.finfo(fD_h5.dtype).eps, fD_h5_attrs['max'])
    # f_density.close()
    # del f_density

    total_items = fD_h5.shape[0]
    tN = fD_h5.shape[0]

    D_h5_dir = os.path.join(filedir, "D_h5")
    if not os.path.exists(D_h5_dir):
        os.mkdir(D_h5_dir)

    fig_h5_d, ax_h5_d = plt.subplots(1, 1)
    ax_h5_d.set_facecolor('midnightblue')
    # cs_h5s_d = plt.imshow(fD_h5[0], extent=extends, cmap='cividis', interpolation='bilinear', vmin=fD_ext[0], vmax=fD_ext[1])
    cs_h5s_d = plt.imshow(fD_h5[0], extent=extends, cmap='cividis', interpolation='bilinear', norm=colors.Normalize(vmin=fD_ext[0], vmax=fD_ext[1]), origin='lower')
    cbar_h5s_d = fig_h5_d.add_axes([0.0, 0.9, 0.05, 0.07])
    plt.colorbar(cs_h5s_d, cax=cbar_h5s_d)
    # plt.savefig(os.path.join(filedir, 'density_h5.png'), dpi=180)

    def init_D_h5_animation():
        cs_h5s_d.set_array(fD_h5[0])
        results = []
        results.append(cs_h5s_d)
        ax_h5_d.set_title("Simulation - HDF5 - Density - Iteration: %d" % (0))
        return results

    def update_D_h5_ani(frames, *args):
        ti = frames
        cs_h5s_d.set_array(fD_h5[ti])
        results = []
        results.append(cs_h5s_d)
        ax_h5_d.set_title("Simulation - HDF5 - Density - Iteration: %d" % (ti))
        current_item = ti
        workdone = current_item / total_items
        print("\rProgress - plot density data: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
        return results

    ani_h5_d = FuncAnimation(fig_h5_d, update_D_h5_ani, init_func=init_D_h5_animation, frames=tN, interval=1, fargs=[])
    ani_h5_d.save(os.path.join(D_h5_dir, 'density_h5.png'), writer=ani_writer_h5, dpi=180)

    # ================================== #
    # ==== Lifetime - Image - HDF5  ==== #
    # ================================== #
    f_lifetime = h5py.File(os.path.join(filedir, 'lifetime.h5'), "r")
    fL_h5 = f_lifetime['lifetime']  # [()]
    fL_h5_attrs = fL_h5.attrs
    # fL_ext = (fL_h5.min(), fL_h5.max())
    fL_ext = (1.0, fD_h5_attrs['max'])
    # f_lifetime.close()
    # del f_lifetime

    total_items = fL_h5.shape[0]
    tN = fL_h5.shape[0]

    L_h5_dir = os.path.join(filedir, "L_h5")
    if not os.path.exists(L_h5_dir):
        os.mkdir(L_h5_dir)

    fig_h5_L, ax_h5_L = plt.subplots(1, 1)
    ax_h5_L.set_facecolor('black')
    cs_h5s_L = plt.imshow(fL_h5[0], extent=extends, cmap='gray', interpolation='bilinear', norm=colors.Normalize(vmin=fL_ext[0], vmax=fL_ext[1]), origin='lower')
    cbar_h5s_L = fig_h5_L.add_axes([0.0, 0.9, 0.05, 0.07])
    plt.colorbar(cs_h5s_L, cax=cbar_h5s_L)
    # plt.savefig(os.path.join(filedir, 'lifetime_h5.png'), dpi=180)

    def init_L_h5_animation():
        cs_h5s_L.set_array(fL_h5[0])
        results = []
        results.append(cs_h5s_L)
        ax_h5_L.set_title("Simulation - HDF5 - Lifetime - Iteration: %d" % (0))
        return results

    def update_L_h5_ani(frames, *args):
        ti = frames
        cs_h5s_L.set_array(fL_h5[ti])
        results = []
        results.append(cs_h5s_L)
        ax_h5_L.set_title("Simulation - HDF5 - Lifetime - Iteration: %d" % (ti))
        current_item = ti
        workdone = current_item / total_items
        print("\rProgress - plot lifetime data: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
        return results

    ani_h5_L = FuncAnimation(fig_h5_L, update_L_h5_ani, init_func=init_L_h5_animation, frames=tN, interval=1, fargs=[])
    ani_h5_L.save(os.path.join(L_h5_dir, 'lifetime_h5.png'), writer=ani_writer_h5, dpi=180)

    # ========================================== #
    # ==== Relative Density - Image - HDF5  ==== #
    # ========================================== #
    f_rel_density = h5py.File(os.path.join(filedir, 'rel_density.h5'), "r")
    frD_h5 = f_rel_density['rel_density']  # [()]
    frD_h5_attrs = frD_h5.attrs
    frD_ext = (frD_h5_attrs['min'] + np.finfo(frD_h5.dtype).eps, frD_h5_attrs['max'])
    # frD_ext = (0.00001, frD_h5.max())
    # f_rel_density.close()
    # del f_rel_density

    total_items = frD_h5.shape[0]
    tN = frD_h5.shape[0]

    rD_h5_dir = os.path.join(filedir, "rD_h5")
    if not os.path.exists(rD_h5_dir):
        os.mkdir(rD_h5_dir)

    fig_h5_rd, ax_h5_rd = plt.subplots(1, 1)
    ax_h5_rd.set_facecolor('midnightblue')
    cs_h5s_rd = plt.imshow(frD_h5[0], extent=extends, cmap='cividis', interpolation='bilinear', norm=colors.LogNorm(vmin=frD_ext[0], vmax=frD_ext[1]), origin='lower')
    cbar_h5s_rd = fig_h5_rd.add_axes([0.0, 0.9, 0.05, 0.07])
    plt.colorbar(cs_h5s_rd, cax=cbar_h5s_rd)
    # plt.savefig(os.path.join(filedir, 'rel_density_h5.png'), dpi=180)

    def init_rD_h5_animation():
        cs_h5s_rd.set_array(frD_h5[0])
        results = []
        results.append(cs_h5s_rd)
        ax_h5_rd.set_title("Simulation - HDF5 - relative Density - Iteration: %d" % (0))
        return results

    def update_rD_h5_ani(frames, *args):
        ti = frames
        cs_h5s_rd.set_array(frD_h5[ti])
        results = []
        results.append(cs_h5s_rd)
        ax_h5_rd.set_title("Simulation - HDF5 - relative Density - Iteration: %d" % (ti))
        current_item = ti
        workdone = current_item / total_items
        print("\rProgress - plot relative Density data: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
        return results

    ani_h5_rD = FuncAnimation(fig_h5_rd, update_rD_h5_ani, init_func=init_rD_h5_animation, frames=tN, interval=1, fargs=[])
    ani_h5_rD.save(os.path.join(rD_h5_dir, 'rD_h5.png'), writer=ani_writer_h5, dpi=180)

    # ======================================== #
    # ==== Particle Count - Image - HDF5  ==== #
    # ======================================== #
    f_pcount = h5py.File(os.path.join(filedir, 'particlecount.h5'), "r")
    fPC_h5 = f_pcount['pcount']  # [()]
    fPC_h5_attrs = fPC_h5.attrs
    max_pcount_value = np.maximum(np.abs(fPC_h5_attrs['min']), np.abs(fPC_h5_attrs['max']))
    fPC_ext = (1, fPC_h5_attrs['max'])
    # f_pcount.close()
    # del f_pcount

    total_items = fPC_h5.shape[0]
    tN = fPC_h5.shape[0]

    PC_h5_dir = os.path.join(filedir, "pcount_h5")
    if not os.path.exists(PC_h5_dir):
        os.mkdir(PC_h5_dir)

    fig_h5_pc, ax_h5_pc = plt.subplots(1, 1)
    ax_h5_pc.set_facecolor('white')
    cs_h5s_pc = plt.imshow(fPC_h5[0], extent=extends, cmap='Reds', interpolation='bilinear', norm=colors.LogNorm(vmin=fPC_ext[0], vmax=fPC_ext[1]), origin='lower')
    cbar_h5s_pc = fig_h5_pc.add_axes([0.0, 0.9, 0.05, 0.07])
    plt.colorbar(cs_h5s_pc, cax=cbar_h5s_pc)
    # plt.savefig(os.path.join(filedir, 'pcount_h5.png'), dpi=180)

    def init_PC_h5_animation():
        cs_h5s_pc.set_array(fPC_h5[0])
        results = []
        results.append(cs_h5s_pc)
        ax_h5_rd.set_title("Simulation - HDF5 - particle count - Iteration: %d" % (0))
        return results

    def update_PC_h5_ani(frames, *args):
        ti = frames
        cs_h5s_pc.set_array(fPC_h5[ti])
        results = []
        results.append(cs_h5s_pc)
        ax_h5_rd.set_title("Simulation - HDF5 - particle count - Iteration: %d" % (ti))
        current_item = ti
        workdone = current_item / total_items
        print("\rProgress - plot particle count data: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
        return results

    ani_h5_PC = FuncAnimation(fig_h5_pc, update_PC_h5_ani, init_func=init_PC_h5_animation, frames=tN, interval=1, fargs=[])
    ani_h5_PC.save(os.path.join(PC_h5_dir, 'pcount_h5.png'), writer=ani_writer_h5, dpi=180)

    print("\nImage generation concluded.")
