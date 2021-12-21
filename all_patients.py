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

# Import Built Packages
import Signal

data_path = './data/Patients'
os.chdir(data_path)

patients = os.listdir()

fs = 250

patient_data = {}

# Isolate all the data for 11.25 Hz (Index 1)
for index, patient in enumerate(patients):
    
    data_temp = scipy.io.loadmat(patient)
    eeg_data = data_temp['data']
    patient_data[patient.replace('.mat', '')] = eeg_data
    
    if index == 0:
        
        Oz_patient = eeg_data[5, :, 1, :, 1]
        Oz = np.array(Oz_patient)
        
    else:
        
        Oz_patient = eeg_data[5, :, 1, :, 1]
        Oz = np.append(Oz, Oz_patient, axis = 1)

os.chdir('..')
subjects = scipy.io.loadmat('Subjects_Information.mat')

Oz_average = np.average(Oz, axis = 1)

Oz_average_Sig = Signal.Signal(fs, Oz_average)
Oz_average_Sig.dft('Averaged Oz Frequency Domain')
plt.show()

# Band Pass Filter Signal
Omega_pass = np.array([7.25, 90])  # Pass Band of Low Pass Filter in [Hz]
Omega_stop = np.array([4, 100])  # Stop Band of Low Pass Filter in [Hz]
Oz_average_Sig.bandpass_chebII(Omega_pass, Omega_stop)
fft_output = Oz_average_Sig.dft('Averaged Frequency Domain')

plt.figure()
plt.plot(Oz_average_Sig.samples)
plt.show()