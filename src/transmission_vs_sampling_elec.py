import os
os.environ["OMP_NUM_THREADS"] = "40"
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pic
from commpy.filters import rrcosfilter
from sampling import *

###############################################################################################
### Simulation script for computing the maximum mode-matching coefficient under sampling ######
### and electronic filtering constraint with varying sampling rate and electronic bandwidth ###
###############################################################################################

rng = np.random.default_rng(6426527)

symbol_rate = 125e6
rho = 0.1

n_points = 1000
n_points_adc = n_points
n_points_bw = n_points

adc_max_log = 1
adc_min_log = -1

bw_min_log = adc_min_log
bw_max_log = adc_max_log

adc_rate_tab = np.logspace(adc_min_log,adc_max_log,n_points_adc) * symbol_rate
elec_bw_tab = np.logspace(bw_min_log,bw_max_log,n_points_bw) * symbol_rate

transmission_tab = np.zeros((n_points_adc,n_points_bw))
butterworth_order = 2

for rate in range(n_points_adc):
    for bw in range(n_points_bw):

        elec_bw = elec_bw_tab[bw]
        adc_rate = adc_rate_tab[rate]
        time_rate = 10*np.max([symbol_rate,elec_bw,adc_rate])
        freq_resolution = np.min([symbol_rate,elec_bw,adc_rate])/10
        n_sample = int(time_rate/freq_resolution)
        if n_sample%2 == 0:
            n_sample = n_sample+1
        n_adc = int(adc_rate*n_sample/time_rate)

        if n_adc == 0:
            continue

        symbol_mode = rrcosfilter(n_sample, rho, 1/symbol_rate, time_rate)[1]
        symbol_mode = symbol_mode/np.linalg.norm(symbol_mode)
        elec_filt = butterworth_filter(n_sample,time_rate,elec_bw,butterworth_order)

        basis_matrix = np.zeros((n_sample,n_adc))
        for sample in range(n_adc):
            basis_matrix[:,sample]=shift_time(elec_filt,time_rate,sample/adc_rate-n_adc//2/adc_rate)

        projection_coeff = np.real(np.linalg.lstsq(basis_matrix,symbol_mode)[0])
        projection_mode = basis_matrix @ projection_coeff

        normalized_projection_mode = projection_mode/np.linalg.norm(projection_mode)

        transmission_tab[rate,bw] = np.abs(inner_product(normalized_projection_mode,symbol_mode))**2

dict_save = {"transmisison" : transmission_tab, "sampling_rate":adc_rate_tab, "elec_bw":elec_bw_tab, "symbol_rate":symbol_rate, "rho":rho}

with open("transmission_vs_sampling_bw_butter_2_log.pkl","wb") as f:
    pic.dump(dict_save,f)
