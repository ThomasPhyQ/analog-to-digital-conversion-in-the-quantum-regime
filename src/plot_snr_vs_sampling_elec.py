import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pic

##################################################################################
### Plotting script for plotting the maximum SNR under sampling and electronic ###
### filtering contraint with varying sampling rate and electronic bandwidth ######
##################################################################################

def plot_transmission_noise(filename,param, xmin = None, xmax = None, ymin=None, ymax = None, logscale = False):
    with open(filename,"rb") as f:
        dict_transmission = pic.load(f)

        transmission_tab = dict_transmission["transmisison"]
        noise_tab = dict_transmission["noise"]
        adc_rate_tab = dict_transmission["sampling_rate"]
        elec_bw_tab = dict_transmission["elec_bw"]
        symbol_rate  = dict_transmission["symbol_rate"]
        rho  = dict_transmission["rho"]
        snr_tab = dict_transmission["snr"]
        mode_matching_loss = -10*np.log10(transmission_tab)
        snr_loss = 10*np.log10(2/snr_tab)

        signal_bandwidth = symbol_rate*(1+rho)/2


        plt.rcParams.update({
            "text.usetex":True,
            "font.family":"Latin Modern",
            "font.size"  : 10,
        })

        golden_ratio = (1+np.sqrt(5))/2

        width_prop = 1/3*4/5
        width_A4 = 210
        width_mm = width_prop*width_A4
        width_inch = width_mm/25.4

        fig = plt.figure(figsize=(width_inch,width_inch))

        bandwidth_axis = np.vectorize(lambda x: f"{x:.1f}")(elec_bw_tab*1e-6)
        sampling_axis = np.vectorize(lambda x: f"{x:.1f}")(adc_rate_tab*1e-6)

        transmission_levels = 20

        transmission_levels_linear = np.linspace(0.5,1,transmission_levels)

        cs = plt.contourf(elec_bw_tab/signal_bandwidth, np.flip(adc_rate_tab)/signal_bandwidth, np.flip(transmission_tab, axis=0), levels = transmission_levels_linear,cmap = "viridis", extend='both')
        cs2 = plt.contour(cs, levels = cs.levels[::1], colors='k', linewidths=1)
        plt.xlabel(r"Normalized electronic bandwidth")
        plt.ylabel(r"Normalized sampling rate")
        if logscale:
            plt.xscale('log')
            plt.yscale('log')

        if xmin!= None and xmax != None:
            plt.xlim(xmin,xmax)
        if ymin != None and ymax!=None:
            plt.ylim(ymin,ymax)
        else:
            ymin = xmin
            ymax = xmax
            if ymin != None and ymax!=None:
                plt.ylim(ymin,ymax)

        cbar = fig.colorbar(cs)
        cbar.ax.set_ylabel('Mode-matching coefficient')
        cbar.add_lines(cs2)
        plt.gca().set_aspect('equal')
        plt.savefig("data/plot/SNR/modematching_vs_sampling_vs_elec_" +param+"_contour.pdf", dpi=1000,bbox_inches="tight")
        plt.clf()

        transmission_levels_db = np.linspace(mode_matching_loss.min(),mode_matching_loss.max(),transmission_levels)

        cs = plt.contourf(elec_bw_tab/signal_bandwidth, np.flip(adc_rate_tab)/signal_bandwidth, np.flip(mode_matching_loss, axis=0), levels = transmission_levels_db,cmap = "viridis_r", extend='both')
        cs2 = plt.contour(cs, levels = cs.levels[::1], colors='k', linewidths=1)
        plt.xlabel(r"Normalized electronic bandwidth")
        plt.ylabel(r"Normalized sampling rate")
        if logscale:
            plt.xscale('log')
            plt.yscale('log')

        if xmin!= None and xmax != None:
            plt.xlim(xmin,xmax)
        if ymin != None and ymax!=None:
            plt.ylim(ymin,ymax)
        else:
            ymin = xmin
            ymax = xmax
            if ymin != None and ymax!=None:
                plt.ylim(ymin,ymax)

        cbar = fig.colorbar(cs)
        cbar.ax.set_ylabel('Mode-matching loss (dB)')
        cbar.add_lines(cs2)
        plt.gca().set_aspect('equal')
        plt.savefig("data/plot/SNR/loss_vs_sampling_vs_elec_"+param+"_contour.pdf", dpi=1000,bbox_inches="tight")
        plt.clf()

        noise_levels = 20

        noise_levels_db = np.linspace(0,1,noise_levels)
        noise_levels_lin = np.linspace(noise_tab.min(),noise_tab.max(),noise_levels)

        cs = plt.contourf(elec_bw_tab/signal_bandwidth, np.flip(adc_rate_tab)/signal_bandwidth, np.flip(noise_tab, axis=0), levels = noise_levels_lin,cmap = "viridis_r", extend='both')
        cs2 = plt.contour(cs, levels = cs.levels[::1], colors='k', linewidths=1)
        plt.xlabel(r"Normalized electronic bandwidth")
        plt.ylabel(r"Normalized sampling rate")
        if logscale:
            plt.xscale('log')
            plt.yscale('log')

        if xmin!= None and xmax != None:
            plt.xlim(xmin,xmax)
        if ymin != None and ymax!=None:
            plt.ylim(ymin,ymax)
        else:
            ymin = xmin
            ymax = xmax
            if ymin != None and ymax!=None:
                plt.ylim(ymin,ymax)

        cbar = fig.colorbar(cs)
        cbar.ax.set_ylabel('Effective electronic noise variance')
        cbar.add_lines(cs2)
        plt.gca().set_aspect('equal')
        plt.savefig("data/plot/SNR/noise_lin_vs_sampling_vs_elec_"+param+"_contour.pdf", dpi=1000,bbox_inches="tight")
        plt.clf()

        snr_levels = 20
        snr_levels_db = np.linspace(snr_loss.min(),snr_loss.max(),snr_levels)

        cs = plt.contourf(elec_bw_tab/signal_bandwidth, np.flip(adc_rate_tab)/signal_bandwidth, np.flip(snr_loss, axis=0), levels = snr_levels_db,cmap = "viridis_r", extend='both')
        cs2 = plt.contour(cs, levels = cs.levels[::1], colors='k', linewidths=1)
        plt.xlabel(r"Normalized electronic bandwidth")
        plt.ylabel(r"Normalized sampling rate")
        if logscale:
            plt.xscale('log')
            plt.yscale('log')

        if xmin!= None and xmax != None:
            plt.xlim(xmin,xmax)
        if ymin != None and ymax!=None:
            plt.ylim(ymin,ymax)
        else:
            ymin = xmin
            ymax = xmax
            if ymin != None and ymax!=None:
                plt.ylim(ymin,ymax)

        cbar = fig.colorbar(cs)
        cbar.ax.set_ylabel(r'$3\ \mathrm{dB}-\mathrm{SNR}_{\mathrm{dB}}$')
        cbar.add_lines(cs2)
        plt.gca().set_aspect('equal')
        plt.savefig("data/plot/SNR/snr_db_vs_sampling_vs_elec_"+param+"_contour.pdf", dpi=1000,bbox_inches="tight")
        #plt.show()
        plt.clf()


plot_transmission_noise("SNR_vs_sampling_bw_butter_2_clear_10_sn_3_log.pkl", "butter_2_clear_10_sn_3_log", logscale = True)

