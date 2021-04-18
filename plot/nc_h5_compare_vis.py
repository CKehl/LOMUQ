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

if __name__ =='__main__':
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-d", "--filedir", dest="filedir", type=str, default="/var/scratch/experiments/NNvsGeostatistics/data", help="head directory containing all input data and also are the store target for output files")
    parser.add_argument("-A", "--animate", dest="animate", action='store_true', default=False, help="Generate particle animations (default: False).")
    args = parser.parse_args()

    # filedir = '/var/scratch/experiments/NNvsGeostatistics/data/2'
    filedir = args.filedir
    animate = args.animate

    # ==== Load flow-field data ==== #
    # f_u = netcdf.netcdf_file('/var/scratch/experiments/SoA/MEDUSA/perlinU.nc', 'r')
    # f_u = xr.open_dataset('/var/scratch/experiments/NNvsGeostatistics/data/perlinU.nc', decode_cf=True, engine='netcdf4')
    # f_u = xr.open_dataset('/var/scratch/experiments/NNvsGeostatistics/data/bickleyjetU.nc', decode_cf=True, engine='netcdf4')
    f_u = xr.open_dataset(os.path.join(filedir, 'doublegyreU.nc'), decode_cf=True, engine='netcdf4')
    fT_nc = f_u.variables['time_counter'].data
    fX_nc = f_u.variables['x'].data
    fY_nc = f_u.variables['y'].data
    fU_nc = f_u.variables['vozocrtx'].data
    # fU = np.transpose(fU, [0,1,3,2])
    max_u_value = np.maximum(np.abs(fU_nc.min()), np.abs(fU_nc.max()))
    fu_ext = (-max_u_value, +max_u_value)
    f_u.close()
    del f_u
    # f_v = netcdf.netcdf_file('/var/scratch/experiments/SoA/MEDUSA/perlinV.nc', 'r')
    # f_v = xr.open_dataset('/var/scratch/experiments/NNvsGeostatistics/data/perlinV.nc', decode_cf=True, engine='netcdf4')
    # f_v = xr.open_dataset('/var/scratch/experiments/NNvsGeostatistics/data/bickleyjetV.nc', decode_cf=True, engine='netcdf4')
    f_v = xr.open_dataset(os.path.join(filedir, 'doublegyreV.nc'), decode_cf=True, engine='netcdf4')
    fV_nc = f_v.variables['vomecrty'].data
    # fV = np.transpose(fV, [0,1,3,2])
    extends = (fX_nc.min(), fX_nc.max(), fY_nc.min(), fY_nc.max())
    max_v_value = np.maximum(np.abs(fV_nc.min()), np.abs(fV_nc.max()))
    fv_ext = (-max_v_value, +max_v_value)
    f_v.close()
    del f_v
    speed_nc = fU_nc ** 2 + fV_nc ** 2
    speed_nc = np.where(speed_nc > 0, np.sqrt(speed_nc), 0)
    f_velmag_ext_nc = (speed_nc.min(), speed_nc.max())

    # plt.figure(facecolor='black')
    # plt.figure(facecolor='white')
    fig_nc_single_u, ax_nc_single_u = plt.subplots(1, 1)  #, facecolor='grey')
    ax_nc_single_u.set_facecolor('grey')
    # cs_ncs_s = ax_nc_single.pcolormesh(fX_nc, fY_nc, speed[0], cmap=cm_compose_speed, shading='gouraud', zorder=1)
    # cs0 = ax.pcolormesh(fX, fY, speed[0], cmap=cm_compose_speed, shading='gouraud' , zorder=1)  #
    # cs_ncs_u = ax_nc_single_u.pcolormesh(fX_nc, fY_nc, fU_nc[0, 0], cmap=colour_scales['blue_quadrant_palette_greybase']['U']['colour_scale'], shading='gouraud' , zorder=2, alpha=0.5)
    # cs_ncs_u = plt.imshow(fU_nc[0, 0], extent=extends, cmap=colour_scales['blue_quadrant_palette_greybase']['U']['colour_scale'], interpolation='bilinear', vmin=fu_ext[0], vmax=fu_ext[1])  # , alpha=0.95
    cs_ncs_u = plt.imshow(fU_nc[0, 0], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['U']['colour_scale'], interpolation='bilinear', vmin=fu_ext[0], vmax=fu_ext[1])  # , alpha=0.95
    # cs2 = ax.pcolormesh(fX, fY, v[0], cmap=cm_compose_v, shading='gouraud' , zorder=3, alpha=0.5)  #
    # cs0 = ax.pcolormesh(fX, fY, speed[0], cmap=cm_compose_speed, shading='gouraud' , zorder=4)  #
    # cs_ncs_u.cmap.set_over('w')
    # cs_ncs_u.cmap.set_under('k')
    # cbar_ncs_u = fig_nc_single_u.add_axes([0, 0, 0.1, 0.1])
    # plt.colorbar(cs_ncs_u, cax=cbar_ncs_u)
    cbar_ncs_u = fig_nc_single_u.add_axes([0.0, 0.9, 0.05, 0.07])
    plt.colorbar(cs_ncs_u, cax=cbar_ncs_u)
    plt.savefig(os.path.join(filedir, 'u_nc.png'), dpi=180)  # , facecolor=combi_fig.get_facecolor(), transparent=True)
    fig_nc_single_v, ax_nc_single_v = plt.subplots(1, 1)  #, facecolor='grey')
    ax_nc_single_v.set_facecolor('grey')
    # cs_ncs_v = ax_nc_single_v.pcolormesh(fX_nc, fY_nc, fV_nc[0, 0], cmap=colour_scales['blue_quadrant_palette_greybase']['V']['colour_scale'], shading='gouraud' , zorder=2, alpha=0.5)
    # cs_ncs_v = plt.imshow(fV_nc[0, 0], extent=extends, cmap=colour_scales['blue_quadrant_palette_greybase']['V']['colour_scale'], interpolation='bilinear', vmin=fv_ext[0], vmax=fv_ext[1])  # , alpha=0.95
    cs_ncs_v = plt.imshow(fV_nc[0, 0], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['V']['colour_scale'], interpolation='bilinear', vmin=fv_ext[0], vmax=fv_ext[1])  # , alpha=0.95
    # cs_ncs_v.cmap.set_over('w')
    # cs_ncs_v.cmap.set_under('k')
    # cbar_ncs_v = fig_nc_single_v.add_axes([0, 0, 0.1, 0.1])
    # plt.colorbar(cs_ncs_v, cax=cbar_ncs_v)
    cbar_ncs_v = fig_nc_single_v.add_axes([0.0, 0.9, 0.05, 0.07])
    plt.colorbar(cs_ncs_v, cax=cbar_ncs_v)
    plt.savefig(os.path.join(filedir, 'v_nc.png'), dpi=180)  # , facecolor=combi_fig.get_facecolor(), transparent=True)
    fig_nc_single_velmag, ax_nc_single_velmag = plt.subplots(1, 1)  #, facecolor='grey')
    ax_nc_single_velmag.set_facecolor('grey')
    # cs_ncs_velmag = ax_nc_single_v.pcolormesh(fX_nc, fY_nc, speed_nc[0, 0][0, 0], cmap=colour_scales['blue_quadrant_palette_greybase']['VelMag']['colour_scale'], shading='gouraud' , zorder=2, alpha=0.5)
    # cs_ncs_velmag = plt.imshow(speed_nc[0, 0], extent=extends, cmap=colour_scales['blue_quadrant_palette_greybase']['VelMag']['colour_scale'], interpolation='bilinear', vmin=fv_ext[0], vmax=fv_ext[1])  # , alpha=0.95
    cs_ncs_velmag = plt.imshow(speed_nc[0, 0], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['VelMag']['colour_scale'], interpolation='bilinear', vmin=f_velmag_ext_nc[0], vmax=f_velmag_ext_nc[1])  # , alpha=0.95
    # cs_ncs_velmag.cmap.set_over('w')
    # cs_ncs_velmag.cmap.set_under('k')
    # cbar_ncs_velmag = fig_nc_single_velmag.add_axes([0, 0, 0.1, 0.1])
    # plt.colorbar(cs_ncs_velmag, cax=cbar_ncs_velmag)
    cbar_ncs_velmag = fig_nc_single_velmag.add_axes([0.0, 0.9, 0.05, 0.07])
    plt.colorbar(cs_ncs_velmag, cax=cbar_ncs_velmag)
    plt.savefig(os.path.join(filedir, 'velmag_nc.png'), dpi=180)  # , facecolor=combi_fig.get_facecolor(), transparent=True)
    #cbar_ax0 = ax.add_axes([0.0, 0.9, 0.05, 0.07])
    #plt.colorbar(cs0, cax=cbar_ax0)
    #cbar_ax1 = ax.add_axes([0.075, 0.9, 0.05, 0.07])
    #plt.colorbar(cs1, cax=cbar_ax1)
    #cbar_ax2 = ax.add_axes([0.15, 0.9, 0.05, 0.07])
    #plt.colorbar(cs2, cax=cbar_ax2)

    f_u = h5py.File(os.path.join(filedir, 'hydrodynamic_U.h5'), "r")
    fU_h5 = f_u['uo'][()]
    max_u_value = np.maximum(np.abs(fU_h5.min()), np.abs(fU_h5.max()))
    fu_ext = (-max_u_value, +max_u_value)
    f_u.close()
    del f_u
    f_v = h5py.File(os.path.join(filedir, 'hydrodynamic_V.h5'), "r")
    fV_h5 = f_v['vo'][()]
    max_v_value = np.maximum(np.abs(fV_h5.min()), np.abs(fV_h5.max()))
    fv_ext = (-max_v_value, +max_v_value)
    f_v.close()
    del f_v
    speed_h5 = fU_h5 ** 2 + fV_h5 ** 2
    speed_h5 = np.where(speed_h5 > 0, np.sqrt(speed_h5), 0)
    f_velmag_ext_h5 = (speed_h5.min(), speed_h5.max())

    fig_h5_single_u, ax_h5_single_u = plt.subplots(1, 1)
    ax_h5_single_u.set_facecolor('grey')
    # cs_h5s_u = plt.imshow(fU_h5[0], extent=extends, cmap=colour_scales['blue_quadrant_palette_greybase']['U']['colour_scale'], interpolation='bilinear', vmin=fu_ext[0], vmax=fu_ext[1])  # , alpha=0.95
    cs_h5s_u = plt.imshow(fU_h5[0], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['U']['colour_scale'], interpolation='bilinear', vmin=fu_ext[0], vmax=fu_ext[1])  # , alpha=0.95
    cbar_h5s_u = fig_h5_single_u.add_axes([0.0, 0.9, 0.05, 0.07])
    plt.colorbar(cs_h5s_u, cax=cbar_h5s_u)
    plt.savefig(os.path.join(filedir, 'u_h5.png'), dpi=180)
    fig_h5_single_v, ax_h5_single_v = plt.subplots(1, 1)
    ax_h5_single_v.set_facecolor('grey')
    # cs_h5s_v = plt.imshow(fV_h5[0], extent=extends, cmap=colour_scales['blue_quadrant_palette_greybase']['V']['colour_scale'], interpolation='bilinear', vmin=fv_ext[0], vmax=fv_ext[1])  # , alpha=0.95
    cs_h5s_v = plt.imshow(fV_h5[0], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['V']['colour_scale'], interpolation='bilinear', vmin=fv_ext[0], vmax=fv_ext[1])  # , alpha=0.95
    cbar_h5s_v = fig_h5_single_v.add_axes([0.0, 0.9, 0.05, 0.07])
    plt.colorbar(cs_h5s_v, cax=cbar_h5s_v)
    plt.savefig(os.path.join(filedir, 'v_h5.png'), dpi=180)
    fig_h5_single_velmag, ax_h5_single_velmag = plt.subplots(1, 1)
    ax_h5_single_velmag.set_facecolor('grey')
    # cs_h5s_velmag = plt.imshow(speed_h5[0], extent=extends, cmap=colour_scales['blue_quadrant_palette_greybase']['VelMag']['colour_scale'], interpolation='bilinear', vmin=fv_ext[0], vmax=fv_ext[1])  # , alpha=0.95
    cs_h5s_velmag = plt.imshow(speed_h5[0], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['VelMag']['colour_scale'], interpolation='bilinear', vmin=f_velmag_ext_h5[0], vmax=f_velmag_ext_h5[1])  # , alpha=0.95
    cbar_h5s_velmag = fig_h5_single_velmag.add_axes([0.0, 0.9, 0.05, 0.07])
    plt.colorbar(cs_h5s_velmag, cax=cbar_h5s_velmag)
    plt.savefig(os.path.join(filedir, 'velmag_h5.png'), dpi=180)

    Pn = 96
    indices = np.random.randint(0, 100, Pn, dtype=int)
    # ============================================== #
    # ==== Particle Trails - Animation - NetCDF ==== #
    # ============================================== #
    if animate:
        # data_xarray = xr.open_dataset('/var/scratch/experiments/NNvsGeostatistics/data/benchmark_perlin_noMPI_n4096_p_fwd.nc')
        # data_xarray = xr.open_dataset('/var/scratch/experiments/NNvsGeostatistics/data/benchmark_bickleyjet_noMPI_p_n4096_fwd.nc')
        data_xarray = xr.open_dataset(os.path.join(filedir, 'benchmark_doublegyre_noMPI_p_n4096_fwd.nc'))
        np.set_printoptions(linewidth=160)
        ns_per_sec = np.timedelta64(1, 's')  # nanoseconds in an sec
        sec_per_day = 86400.0
        time_array = data_xarray['time'].data / ns_per_sec

        N = data_xarray['lon'].data.shape[0]
        tN = data_xarray['lon'].data.shape[1]
        indices = np.random.randint(0, N-1, Pn, dtype=int)

        time_since_release_nc = (time_array.transpose() - time_array[:, 0])  # substract the initial time from each timeseri
        sim_dt_nc = time_since_release_nc[1,0] - time_since_release_nc[0,0]
        print("dt = {}".format(sim_dt_nc))
        pX_nc = data_xarray['lon'].data[indices, :]  # only plot the first 32 particles
        pY_nc = data_xarray['lat'].data[indices, :]  # only plot the first 32 particles

        # speed_map = cm.get_cmap('twilight', 256)
        fig_anim_nc = plt.figure()
        ax_anim_nc = plt.axes()
        cs_nca_velmag = ax_anim_nc.pcolormesh(fX_nc, fY_nc, speed_nc[0, 0], cmap=colour_scales['blue_quadrant_palette_blackbase']['VelMag']['colour_scale'], vmin=f_velmag_ext_nc[0], vmax=f_velmag_ext_nc[1], shading='gouraud', zorder=1)
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
            lines_nc[i].add_point(pX_nc[i, 0], pY_nc[i, 0])

        for l in lines_nc:
            ax_anim_nc.add_collection(l.get_LineCollection())

        def init_nc_animation():
            # cs_nca_u.set_array(fU_nc[0, 0])
            # cs_nca_v.set_array(fV_nc[0, 0])
            cs_nca_velmag.set_array(speed_nc[0, 0])
            # results = [cs_nca_u, cs_nca_v, ]
            results = []
            results.append(cs_nca_velmag)
            for l in lines_nc:
                results.append(l.get_LineCollection())
            ax_anim_nc.set_title("Simulation t = %10.4f s"% (time_since_release_nc[0, 0]))
            return results

        def update_flow_only_nc(frames, *args):
            if (frames % 10) == 0:
                print("Plotting frame {} of {} ...".format(frames, tN))
            dt = args[0]
            tx = float(frames)*dt
            tx = math.fmod(tx, fT_nc[-1])
            ti0 = time_index_value(tx, fT_nc)
            tt = time_partion_value(tx, fT_nc)
            ti1 = 0
            if ti0 < (len(fT_nc)-1):
                ti1 = ti0+1
            else:
                ti1 = 0
            speed_1 = fU_nc[ti0] ** 2 + fV_nc[ti0] ** 2
            speed_1 = np.where(speed_1 > 0, np.sqrt(speed_1), 0)
            speed_2 = fU_nc[ti1] ** 2 + fV_nc[ti1] ** 2
            speed_2 = np.where(speed_2 > 0, np.sqrt(speed_2), 0)
            # fu_show = tt*fU_nc[ti0][0] + (1.0-tt)*fU_nc[ti1][0]
            # fv_show = tt*fV_nc[ti0][0] + (1.0-tt)*fV_nc[ti1][0]
            fs_show = tt*speed_1[0] + (1.0-tt)*speed_2[0]
            # cs_nca_u.set_array(fu_show)
            # cs_nca_v.set_array(fv_show)
            cs_nca_velmag.set_array(fs_show)
            # == add new lines == #
            if frames>0:
                for pindex in range(0, Pn):
                    # pt = sim[pindex].point(frames)
                    #pi_lon = data_xarray['lon'].data[indices[i], frames]
                    #pi_lat = data_xarray['lat'].data[indices[i], frames]
                    lines_nc[pindex].add_point(pX_nc[pindex, frames], pY_nc[pindex, frames])
            # == collect results == #
            # results = [cs_nca_u, cs_nca_v, ]
            results = []
            results.append(cs_nca_velmag)
            for l in lines_nc:
                results.append(l.get_LineCollection())
            ax_anim_nc.set_title("Simulation - NetCDF data - t = %5.1f d"% (tx / sec_per_day))
            return results

        ##Writer = writers['ffmpeg']
        ##ani_writer = Writer(fps=25, bitrate=24000)
        Writer = writers['imagemagick_file']
        ani_writer = Writer()

        ##plt.show()
        ##ani = FuncAnimation(fig, update_flow_only, init_func=init, frames=sim_steps, interval=25, fargs=[dt,], blit=True)
        ##ani.save("flow_w_particles.mp4", writer=ani_writer)

        #sim_dt
        ani = FuncAnimation(fig_anim_nc, update_flow_only_nc, init_func=init_nc_animation, frames=tN, interval=1, fargs=[sim_dt_nc,]) #, save_count=1, cache_frame_data=False) #, blit=True)
        # ani.save("/var/scratch/experiments/NNvsGeostatistics/data/sim_imgs_nc/benchmark_perlin.png", writer=ani_writer, dpi=180)
        # ani.save("/var/scratch/experiments/NNvsGeostatistics/data/sim_imgs_nc/benchmark_bickleyjet.png", writer=ani_writer, dpi=180)
        ani_nc_dir = os.path.join(filedir, "sim_imgs_nc")
        if not os.path.exists(ani_nc_dir):
            os.mkdir(ani_nc_dir)
        ani.save(os.path.join(ani_nc_dir, 'benchmark_doublegyre.png'), writer=ani_writer, dpi=180)


    # ============================================== #
    # ====  Particle Trails - Animation - HDF5  ==== #
    # ============================================== #
    if animate:
        f_grid = h5py.File(os.path.join(filedir, 'grid.h5'), "r")
        fT_h5 = f_grid["times"][()]
        # print("fT_h5 = {}".format(fT_h5))
        f_pts = h5py.File(os.path.join(filedir, 'particles.h5'), "r")
        pT_h5 = f_pts['p_t'][()]
        sec_per_day = 86400.0

        N = pT_h5.shape[0]
        tN = pT_h5.data.shape[1]

        time_since_release_h5 = pT_h5
        # time_since_release_h5 = (pT_h5.transpose() - pT_h5[:, 0])  # substract the initial time from each timeseries
        pT_h5 = pT_h5[indices, :]
        sim_dt_h5 = time_since_release_h5[0,1] - time_since_release_h5[0,0]
        print("dt = {}".format(sim_dt_h5))
        pX_h5 = f_pts['p_x'][()][indices, :]  # only plot the first 32 particles
        pY_h5 = f_pts['p_y'][()][indices, :]  # only plot the first 32 particles
        f_pts.close()
        del f_pts

        # speed_map = cm.get_cmap('twilight', 256)
        fig_anim_h5 = plt.figure()
        ax_anim_h5 = plt.axes()
        # cs_h5a_velmag = ax_anim_h5.pcolormesh(fX_h5, fY_h5, speed_nc[0, 0], cmap=colour_scales['blue_quadrant_palette_blackbase']['VelMag']['colour_scale'], vmin=fu_ext[0], vmax=fu_ext[1], shading='gouraud', zorder=1)
        # cs_nca_bg = plt.imshow(np.zeros(fU_h5[0].shape), extent=extends, cmap=ListedColormap(np.array([[0, 0, 0, 1.0], ] * 256, dtype=np.float32), name='black_solid'), interpolation='bilinear')  # black background
        # cs_nca_u = plt.imshow(fU_h5[0], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['U']['colour_scale'], interpolation='bilinear', vmin=fu_ext[0], vmax=fu_ext[1])  # , alpha=0.95
        # cs_nca_v = plt.imshow(fV_h5[0], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['V']['colour_scale'], interpolation='bilinear', vmin=fv_ext[0], vmax=fv_ext[1])  # , alpha=0.95
        cs_h5a_velmag = plt.imshow(speed_h5[0], extent=extends, cmap=colour_scales['blue_quadrant_palette_blackbase']['VelMag']['colour_scale'], interpolation='bilinear', norm=colors.Normalize(vmin=f_velmag_ext_h5[0], vmax=f_velmag_ext_h5[1]))  # , alpha=0.3, shading='gouraud', zorder=1
        # cbar_nca_velmag = fig_anim_nc.add_axes([0, 0.015, 0.1, 0.1])
        cbar_h5a_velmag = fig_anim_h5.add_axes([0.0, 0.9, 0.05, 0.07])
        plt.colorbar(cs_h5a_velmag, cax=cbar_h5a_velmag)
        ax_anim_h5.set_title("Simulation - NetCDF data - t = %5.1f d" % (time_since_release_h5[0, 0] / sec_per_day))
        lines_h5 = []
        for i in range(0, Pn):
            # lines.append(DecayLine(14, 8, [1.0, 0.625, 0.], zorder=2 + i))
            lines_h5.append(DecayLine(14, 8, [0.0, 0.625, 0.0], zorder=2 + i))

        for i in range(0, Pn):
            lines_h5[i].add_point(pX_h5[i, 0], pY_h5[i, 0])

        for l in lines_h5:
            ax_anim_h5.add_collection(l.get_LineCollection())

        def init_h5_animation():
            # cs_nca_u.set_array(fU_nc[0, 0])
            # cs_nca_v.set_array(fV_nc[0, 0])
            cs_h5a_velmag.set_array(speed_h5[0])
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
            tx = np.float64(frames)*dt
            # tx = math.fmod(tx, fT_h5[-1])
            tx = np.fmod(tx, fT_h5[-1])
            ti0 = time_index_value(tx, fT_h5)
            tt = time_partion_value(tx, fT_h5)
            ti1 = 0
            if ti0 < (len(fT_h5)-1):
                ti1 = ti0+1
            else:
                ti1 = 0
            speed_1 = fU_h5[ti0] ** 2 + fV_h5[ti0] ** 2
            speed_1 = np.where(speed_1 > 0, np.sqrt(speed_1), 0)
            speed_2 = fU_h5[ti1] ** 2 + fV_h5[ti1] ** 2
            speed_2 = np.where(speed_2 > 0, np.sqrt(speed_2), 0)
            # fu_show = tt*fU_nc[ti0] + (1.0-tt)*fU_nc[ti1]
            # fv_show = tt*fV_nc[ti0] + (1.0-tt)*fV_nc[ti1]
            fs_show = tt*speed_1 + (1.0-tt)*speed_2
            # cs_nca_u.set_array(fu_show)
            # cs_nca_v.set_array(fv_show)
            cs_h5a_velmag.set_array(fs_show)
            # == add new lines == #
            if frames>0:
                for pindex in range(0, Pn):
                    lines_h5[pindex].add_point(pX_h5[pindex, frames], pY_h5[pindex, frames])
            # == collect results == #
            # results = [cs_nca_u, cs_nca_v, ]
            results = []
            results.append(cs_h5a_velmag)
            for l in lines_h5:
                results.append(l.get_LineCollection())
            ax_anim_h5.set_title("Simulation - h5 data - t = %5.1f d"% (tx / sec_per_day))
            return results

        Writer = writers['imagemagick_file']
        ani_writer_h5 = Writer()
        ani_h5 = FuncAnimation(fig_anim_h5, update_flow_only_h5, init_func=init_h5_animation, frames=tN, interval=1, fargs=[sim_dt_h5,]) #, save_count=1, cache_frame_data=False) #, blit=True)
        # ani.save("/var/scratch/experiments/NNvsGeostatistics/data/sim_imgs_nc/benchmark_perlin.png", writer=ani_writer, dpi=180)
        # ani.save("/var/scratch/experiments/NNvsGeostatistics/data/sim_imgs_nc/benchmark_bickleyjet.png", writer=ani_writer, dpi=180)
        ani_h5_dir = os.path.join(filedir, "sim_imgs_h5")
        if not os.path.exists(ani_h5_dir):
            os.mkdir(ani_h5_dir)
        ani_h5.save(os.path.join(ani_h5_dir, 'benchmark_doublegyre.png'), writer=ani_writer_h5, dpi=180)


    # ============================================== #
    # ==== Lifetime and Density - Image - HDF5  ==== #
    # ============================================== #
    f_density = h5py.File(os.path.join(filedir, 'density.h5'), "r")
    fD_h5 = f_density['density'][()]
    fD_ext = (fD_h5.min()+np.finfo(fD_h5.dtype).eps, fD_h5.max())
    # fD_ext = (0., fD_h5.max())
    f_density.close()
    del f_density

    fig_h5_single_d, ax_h5_single_d = plt.subplots(1, 1)
    ax_h5_single_d.set_facecolor('midnightblue')
    # cs_h5s_d = plt.imshow(fD_h5[0], extent=extends, cmap='cividis', interpolation='bilinear', vmin=fD_ext[0], vmax=fD_ext[1])  # , alpha=0.95
    cs_h5s_d = plt.imshow(fD_h5[-1], extent=extends, cmap='cividis', interpolation='bilinear', vmin=fD_ext[0], vmax=fD_ext[1])  # , alpha=0.95
    cbar_h5s_d = fig_h5_single_d.add_axes([0.0, 0.9, 0.05, 0.07])
    plt.colorbar(cs_h5s_d, cax=cbar_h5s_d)
    plt.savefig(os.path.join(filedir, 'density_h5.png'), dpi=180)

    f_lifetime = h5py.File(os.path.join(filedir, 'lifetime.h5'), "r")
    fL_h5 = f_lifetime['lifetime'][()]
    # fL_ext = (fL_h5.min(), fL_h5.max())
    fL_ext = (1.0, fL_h5.max())
    f_lifetime.close()
    del f_lifetime

    fig_h5_single_L, ax_h5_single_L = plt.subplots(1, 1)
    ax_h5_single_L.set_facecolor('black')
    # cs_h5s_L = plt.imshow(fL_h5[0], extent=extends, cmap='cividis', interpolation='bilinear', vmin=fL_ext[0], vmax=fL_ext[1])  # , alpha=0.95
    # cs_h5s_L = plt.imshow(fL_h5[-1], extent=extends, cmap='gray', interpolation='bilinear', vmin=fL_ext[0], vmax=fL_ext[1])  # , alpha=0.95
    cs_h5s_L = plt.imshow(fL_h5[-1], extent=extends, cmap='gray', interpolation='bilinear', norm=colors.LogNorm(vmin=fL_ext[0], vmax=fL_ext[1]))
    cbar_h5s_L = fig_h5_single_L.add_axes([0.0, 0.9, 0.05, 0.07])
    plt.colorbar(cs_h5s_L, cax=cbar_h5s_L)
    plt.savefig(os.path.join(filedir, 'lifetime_h5.png'), dpi=180)


    # ============================================================= #
    # ==== Particle Count and relative Density - Image - HDF5  ==== #
    # ============================================================= #
    f_rel_density = h5py.File(os.path.join(filedir, 'rel_density.h5'), "r")
    frD_h5 = f_rel_density['rel_density'][()]
    frD_ext = (frD_h5.min()+np.finfo(frD_h5.dtype).eps, frD_h5.max())
    # frD_ext = (0.00001, frD_h5.max())
    f_rel_density.close()
    del f_rel_density

    fig_h5_single_rd, ax_h5_single_rd = plt.subplots(1, 1)
    ax_h5_single_rd.set_facecolor('midnightblue')
    # cs_h5s_rd = plt.imshow(frD_h5[0], extent=extends, cmap='cividis', interpolation='bilinear', vmin=frD_ext[0], vmax=frD_ext[1])  # , alpha=0.95
    # cs_h5s_rd = plt.imshow(frD_h5[-1], extent=extends, cmap='cividis', interpolation='bilinear', vmin=frD_ext[0], vmax=frD_ext[1])  # , alpha=0.95
    cs_h5s_rd = plt.imshow(frD_h5[-1], extent=extends, cmap='cividis', interpolation='bilinear', norm=colors.LogNorm(vmin=frD_ext[0], vmax=frD_ext[1]))
    cbar_h5s_rd = fig_h5_single_rd.add_axes([0.0, 0.9, 0.05, 0.07])
    plt.colorbar(cs_h5s_rd, cax=cbar_h5s_rd)
    plt.savefig(os.path.join(filedir, 'rel_density_h5.png'), dpi=180)

    f_pcount = h5py.File(os.path.join(filedir, 'particlecount.h5'), "r")
    fPC_h5 = f_pcount['pcount'][()]
    max_pcount_value = np.maximum(np.abs(fPC_h5.min()), np.abs(fPC_h5.max()))
    fPC_ext = (1, fPC_h5.max())
    f_pcount.close()
    del f_pcount

    fig_h5_single_pc, ax_h5_single_pc = plt.subplots(1, 1)
    ax_h5_single_pc.set_facecolor('white')
    # cs_h5s_pc = plt.imshow(fPC_h5[0], extent=extends, cmap='cividis', interpolation='bilinear', vmin=fPC_ext[0], vmax=fPC_ext[1])  # , alpha=0.95
    # cs_h5s_pc = plt.imshow(fPC_h5[-1], extent=extends, cmap='Reds', interpolation='bilinear', vmin=fPC_ext[0], vmax=fPC_ext[1])  # , alpha=0.95
    cs_h5s_pc = plt.imshow(fPC_h5[-1], extent=extends, cmap='Reds', interpolation='bilinear', norm=colors.LogNorm(vmin=fPC_ext[0], vmax=fPC_ext[1]))
    cbar_h5s_pc = fig_h5_single_pc.add_axes([0.0, 0.9, 0.05, 0.07])
    plt.colorbar(cs_h5s_pc, cax=cbar_h5s_pc)
    plt.savefig(os.path.join(filedir, 'pcount_h5.png'), dpi=180)

