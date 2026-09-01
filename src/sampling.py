import numpy as np
import scipy as sp
from commpy.filters import rrcosfilter

def make_time_vector(n_sample,sampling_rate):
    if n_sample%2==0:
        raise Exception("All number of samples must be odd...")
    T = n_sample/sampling_rate
    return np.linspace(-T/2,T/2,n_sample)

def make_freq_vector(n_sample,sampling_rate):
    if n_sample%2==0:
        raise Exception("All number of samples must be odd...")
    return np.linspace(-sampling_rate/2,sampling_rate/2,n_sample)

def exp_filter(n_sample,sampling_rate,tau):
    t_vec = make_time_vector(n_sample,sampling_rate)
    filt = np.zeros(n_sample)
    filt[0:int(n_sample/2)] = np.exp(t_vec[0:int(n_sample/2)]/tau)
    return filt/np.linalg.norm(filt)

def gauss_filter(n_sample,sampling_rate,tau):
    t_vec = make_time_vector(n_sample,sampling_rate)
    filt = np.exp(-t_vec**2/2/tau**2)
    return filt#/np.linalg.norm(filt)

def window_filter(n_sample,sampling_rate,tau):
    t_vec = make_time_vector(n_sample,sampling_rate)
    filt = np.zeros(n_sample)
    filt[n_sample//2-int(tau*sampling_rate):n_sample//2] = np.ones(int(tau*sampling_rate))
    return filt/np.linalg.norm(filt)

def window_freq_filter(n_sample,sampling_rate,cutoff):
    f_vec = make_freq_vector(n_sample,sampling_rate)
    filt = np.zeros(n_sample)
    bw_bin = int(cutoff*n_sample/sampling_rate)
    if bw_bin%2==1:
        bw_bin=bw_bin+1
    filt[n_sample//2-bw_bin:n_sample//2+bw_bin] = np.ones(2*bw_bin)
    out_filt = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(filt)))
    return out_filt/np.linalg.norm(out_filt)

def shift_sample(signal,n_0):
    return np.roll(signal, n_0)

def shift_time(signal, sampling_rate, tau):
    n_bin = np.rint(tau*sampling_rate)
    return shift_sample(signal,n_bin)

def butterworth_filter(n_sample,sampling_rate,cutoff,order,normalized=True):
    normalized_cutoff = cutoff/sampling_rate*2
    numerator, denominator = sp.signal.butter(order,normalized_cutoff,btype='lowpass',output='ba')
    impulse = np.zeros(n_sample)
    impulse[n_sample//2+1] = 1
    out_filter = sp.signal.lfilter(numerator, denominator, impulse)
    if normalized:
        return out_filter/np.linalg.norm(out_filter)
    else :
        return out_filter

def inner_product(x,y):
    n = np.shape(x)[0]
    if np.shape(y)[0]==n:
        return np.dot(x,y)
    print("Non valid shape for inner product")
    return 0

def realistic_rrc(n_sample,roll_off,symbol_period,sampling_rate,dac_bit,rng):
    symbol_mode = rrcosfilter(n_sample, roll_off, symbol_period, sampling_rate)[1]
    symbol_mode = symbol_mode/np.linalg.norm(symbol_mode)
    PE = np.max(symbol_mode) - np.min(symbol_mode)
    q = PE/(2**dac_bit)
    sigma = np.sqrt(q**2/12)
    return symbol_mode + sigma*rng.normal(n_sample)/np.sqrt(sampling_rate)

def normalized_spectrum(x):
    return np.abs(np.fft.fftshift(np.fft.fft(x))/np.fft.fft(x)[0])**2
