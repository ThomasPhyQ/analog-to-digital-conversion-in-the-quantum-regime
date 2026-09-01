import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pic

#############################################################################################
### Plotting script for plotting the maximum mode-matching coefficient under sampling and ###
### electronic filtering constraint with varying sampling rate and electronic bandwidth #####
#############################################################################################

def plot_transmission(filename,param, xmin = None, xmax = None, ymin=None, ymax = None, logscale = False):
    with open(filename,"rb") as f:
        dict_transmission = pic.load(f)

        transmission_tab = dict_transmission["transmisison"]
        adc_rate_tab = dict_transmission["sampling_rate"]
        elec_bw_tab = dict_transmission["elec_bw"]
        symbol_rate  = dict_transmission["symbol_rate"]
        rho  = dict_transmission["rho"]
        signal_bandwidth = (1+rho)/2*symbol_rate


        plt.rcParams.update({
            "text.usetex":True,
            "font.family":"rmfamily",
            "font.size"  : 10,
        })

        golden_ratio = (1+np.sqrt(5))/2

        width_prop =1/2
        width_A4 = 210
        width_mm = width_prop*width_A4
        width_inch = width_mm/25.4

        fig = plt.figure(figsize=(width_inch,width_inch))

        bandwidth_axis = np.vectorize(lambda x: f"{x:.1f}")(elec_bw_tab/signal_bandwidth)
        sampling_axis = np.vectorize(lambda x: f"{x:.1f}")(adc_rate_tab/signal_bandwidth)

        n_levels = 20
        levels_linear = np.linspace(0.5,1,n_levels)
        levels_db = np.linspace(0,10,n_levels)

        cs = plt.contourf(elec_bw_tab/signal_bandwidth, np.flip(adc_rate_tab/signal_bandwidth), np.flip(transmission_tab, axis=0), levels = levels_linear,cmap = "viridis", extend='both')
        cs2 = plt.contour(cs, levels = cs.levels[::1], colors='k', linewidths=1)
        plt.xlabel("Electronic bandwidth (MHz)")
        plt.ylabel("Sampling rate (MHz)")
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
        plt.savefig("data/plot/modematching_vs_sampling_vs_elec_" +param+"_contour.pdf", dpi=1000,bbox_inches="tight")
        plt.clf()

        cs = plt.contourf(elec_bw_tab/signal_bandwidth, np.flip(adc_rate_tab/signal_bandwidth), np.flip(-10*np.log10(transmission_tab), axis=0), levels = levels_db,cmap = "viridis_r", extend='both')
        cs2 = plt.contour(cs, levels = cs.levels[::1], colors='k', linewidths=1)
        plt.xlabel("Normalized electronic bandwidth")
        plt.ylabel("Normalized sampling rate")
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
        plt.savefig("data/plot/loss_vs_sampling_vs_elec_"+param+"_contour.pdf", dpi=1000,bbox_inches="tight")
        plt.clf()

plot_transmission("transmission_vs_sampling_bw_butter_2_log.pkl", "butter_2_log", logscale = True)

