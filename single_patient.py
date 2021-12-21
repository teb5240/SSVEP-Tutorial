# Import Packages
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import signal
import os

import scipy.io

import nitime.algorithms as tsa
import nitime.utils as utils

import Signal

data_path = './data/Patients'
os.chdir(data_path)

fs = 250

patient = scipy.io.loadmat('S001.mat')

os.chdir('..')

subjects = scipy.io.loadmat('Subjects_Information.mat')

data = patient['data']

eeg = data[5, :, 1, :, 1]

plt.figure()
plt.plot(eeg)
plt.show()

for i in range(0, 10):

    eeg_Sig = Signal.Signal(fs, eeg[:, i])
    eeg_Sig.dft('Subject 1 EEG')
    plt.show()
    
    # # Low Pass Filter Signal
    # Omega_pass = 40  # Pass Band of Low Pass Filter in [Hz]
    # Omega_stop = 50  # Stop Band of Low Pass Filter in [Hz]
    # eeg_Sig.lowpass_kaiser(Omega_pass, Omega_stop)
    # eeg_Sig.dft('Subject 1 EEG')
    
    # Band Pass Filter Signal
    Omega_pass = np.array([7.25, 90])  # Pass Band of Low Pass Filter in [Hz]
    Omega_stop = np.array([4, 100])  # Stop Band of Low Pass Filter in [Hz]
    eeg_Sig.bandpass_chebII(Omega_pass, Omega_stop)
    eeg_Sig.dft('Subject 1 EEG')
    
    # eeg_Sig.detrend()
    
    # detrended_eeg = eeg_Sig.samples
    
    filtered_eeg = eeg_Sig.samples 
    
    # eeg_Sig.detrend()
    
    plt.figure()
    plt.plot(filtered_eeg)
    plt.show()
    
    # plt.figure()
    # plt.plot(detrended_eeg)
    # plt.show()
    
    def dB(x, out=None):
        if out is None:
            return 10 * np.log10(x)
        else:
            np.log10(x, out)
            np.multiply(out, 10, out)
            
    f, Phi_multi, nu = tsa.multi_taper_psd(filtered_eeg[300:],  adaptive=False, jackknife=False)
    Phi_multi_norm = Phi_multi/sum(Phi_multi)
    Phi_multi_dB = dB(Phi_multi_norm)
    
    f_cont = np.linspace(0, 125, len(f))
    
    plt.figure()
    plt.plot(f_cont, Phi_multi_dB, 'ro')
    # plt.ylim([-120, 0])
    plt.xlim([9, 20])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density [dB]')
    plt.title('Multitaper Power Spectral Density Estimate [signal package]')
    
    if i == 0:
        
        spectra_dB = np.array([Phi_multi_dB])
        
    else:
        
        Phi_multi_dB = np.array([Phi_multi_dB])
        spectra_dB = np.append(spectra_dB, Phi_multi_dB, axis = 0)

    
    print(Phi_multi_dB)
    
    
print('pause')

average_spectra = np.average(spectra_dB, axis = 0)

plt.figure()
plt.plot(average_spectra, 'o')
plt.xlim([8, 25])
plt.show()
