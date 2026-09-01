import os
os.environ["OMP_NUM_THREADS"] = "40"
os.environ["OPENBLAS_NUM_THREADS"] = "40"
import numpy as np
import scipy as sp
import pickle as pic
from commpy.filters import rrcosfilter
from sampling import *

#####################################################################################
### Simulation script for computing the maximum SNR under sampling and electronic ###
### filtering constraint with varying sampling rate and electronic bandwidth ########
#####################################################################################

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
noise_tab = np.zeros((n_points_adc,n_points_bw))
SNR_tab = np.zeros((n_points_adc,n_points_bw))
butterworth_order = 2

shot_noise_level = 1e-3
clearance_dB = 10 #in dB
clearance = 10**(clearance_dB/10) #in dB
noise_level = shot_noise_level/(10**(clearance_dB/10)-1)

def norm_projection_mode(projection_coeff, basis_matrix):
    y = basis_matrix@projection_coeff
    return y

def transmission(norm_projection_mode, symbol_mode):
    return np.abs(np.real(np.dot(norm_projection_mode,symbol_mode)))**2

def noise(clearance, norm_projection_coeff,shot_noise_level):
    return 1/(4*shot_noise_level)*clearance/(clearance-1) + 1/(clearance-1)*np.linalg.norm(norm_projection_coeff)**2

def SNR(transmission,noise):
    return transmission/noise

for rate in reversed(range(n_points_adc)):
    for bw in range(n_points_bw):

        elec_bw = elec_bw_tab[bw]
        adc_rate = adc_rate_tab[rate]
        time_rate = 5*np.max([symbol_rate,elec_bw,adc_rate])
        freq_resolution = np.min([symbol_rate,elec_bw,adc_rate])/5
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

        overlap_mode = symbol_mode@basis_matrix
        overlap_matrix = np.outer(overlap_mode,overlap_mode)


        # Normalization of the mode
        def cons_f(projection_coeff):
            return np.linalg.norm(basis_matrix@projection_coeff)**2-1/(2*shot_noise_level)*clearance/(clearance-1)

        # Numerical optimization of the constraint
        def cons_J(projection_coeff):
            return 2*np.transpose(projection_coeff)@np.transpose(basis_matrix)@basis_matrix

        def cons_H(projection_coeff,v):
            return 2*v[0]*np.transpose(basis_matrix)@basis_matrix

        # -SNR function
        def objective_function(projection_coeff):
            mode = norm_projection_mode(projection_coeff,basis_matrix)
            eta = transmission(mode,symbol_mode)
            w = noise(clearance,projection_coeff,shot_noise_level)
            return -SNR(eta,w)

        # Numerical optimization for scipy.optimize
        def objective_J(projection_coeff):
            norm = np.linalg.norm(projection_coeff)**2
            outer = np.outer(projection_coeff,projection_coeff)
            factor = 1/(clearance-1)
            denom = 1/(4*shot_noise_level)*clearance*factor + factor*norm
            return -2*overlap_matrix@projection_coeff/denom + 2*factor*outer@overlap_matrix@projection_coeff/(denom**2)

        def objective_H(projection_coeff):
            norm=np.linalg.norm(projection_coeff)**2
            overlap = np.dot(overlap_mode,projection_coeff)
            outer = np.outer(projection_coeff,projection_coeff)
            factor = 1/(clearance-1)
            denom = 1/(4*shot_noise_level)*clearance*factor + factor*norm
            outer_mix = np.outer(overlap_mode,projection_coeff)
            identity = np.identity(n_adc)
            return -2*overlap_matrix/denom + 2*overlap**2*factor/(denom**2)*identity -8*factor**2*overlap**2*outer/(denom**3) + 4*factor*overlap_matrix@outer/(denom**2) + 4*factor*outer@overlap_matrix/(denom**2)

        non_linear_constraint = sp.optimize.NonlinearConstraint(cons_f,0,0,jac=cons_J,hess=cons_H)

        x_0 = np.ones(n_adc)/np.linalg.norm(basis_matrix@np.ones(n_adc))*np.sqrt(2*shot_noise_level*(clearance-1)/clearance)
        res = sp.optimize.minimize(objective_function,x_0,method="trust-constr", constraints=[non_linear_constraint], jac=objective_J, hess=objective_H)

        normalized_projection_coeff = res.x

        normalized_projection_mode = basis_matrix@normalized_projection_coeff

        transmission_tab[rate,bw] = 2*shot_noise_level*(clearance-1)/clearance*np.abs(np.dot(normalized_projection_coeff,overlap_mode))**2
        noise_tab[rate,bw] = noise_level*np.linalg.norm(normalized_projection_coeff)**2
        SNR_tab[rate,bw] = -res.fun

dict_save = {"snr" : SNR_tab,"transmisison" : transmission_tab, "noise" : noise_tab, "sampling_rate":adc_rate_tab, "elec_bw":elec_bw_tab, "symbol_rate":symbol_rate, "rho":rho, "clearance_dB": clearance_dB, "shot_noise_level" : shot_noise_level}

with open("SNR_vs_sampling_bw_butter_2_clear_10_sn_3_log.pkl","wb") as f:
    pic.dump(dict_save,f)

