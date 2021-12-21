import scipy
import scipy.signal as signal
from matplotlib import pyplot as plt
import numpy as np
import cmath
import time
import dtsp

class Signal:

    def __init__(self, fs, data, sig_name = ''):
        # fs = sampling frequency
        # data = data in an array

        print(" ---------------- INITIALIZING ---------------- ")

        self.fs                  = fs # sampling frequency
        self.samples             = data # the actual data
        self.data_length_sec     = self.samples.shape[0] / fs # length of data in sec
        self.data_length_samples = self.samples.shape[0] # length of data in samples
        self.color_1             = "steelblue"
        self.color_2             = "orange"
        self.color_3			 = "indianred"
        self.sig_name 			 = sig_name

    def plot_signal(self, ylabel = '', title = ''):

        time_vector = np.linspace(0, self.data_length_sec, self.data_length_samples)

        plt.figure()
        plt.plot(time_vector, self.samples)
        plt.xlabel('Time [sec]')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def dft(self, title):
        # Plot DFT of Samples through FFT Approximation

        print(" ---------------- DFT ---------------- ")

        # Look at Variables
        print("Length of input samples: {}".format(len(self.samples)))
        print("Samples: {}".format(self.samples))

        samples_dtft = np.fft.fft(self.samples) # fft of input samples
        N = len(samples_dtft) # Number of samples of the DTFT for the DFT

        print(N)

        fig, (ax1) = plt.subplots(1, 1)

        # Split Data (Positive Length)
        if N % 2 == 0:
            time_negative         = np.arange(int(-N/2), -1)
            time_positive         = np.arange(0, int(N/2) - 1)
            samples_dtft_positive = samples_dtft[0:int(N/2)-1]
            samples_dtft_negative = samples_dtft[int(N/2):N-1]
        else:
            time_negative         = np.arange(int(-(N/2) + .5), -1)
            time_positive         = np.arange(0, int((N/2) - .5))
            samples_dtft_positive = samples_dtft[0 : int((N/2) - .5)]
            samples_dtft_negative = samples_dtft[int((N/2) + .5): N-1]

        # Plot Magnitude
        factor = 1
        fig.suptitle(title, fontsize=16)
        ax1.plot(2*time_positive[::factor]/N, 20*np.log10(np.abs(samples_dtft_positive[::factor])),  linewidth=0.5,  color = self.color_1) # magnitude
        ax1.plot(2*time_negative[::factor]/N, 20*np.log10(np.abs(samples_dtft_negative[::factor])), linewidth=0.5,  color = self.color_1) # magnitude
        ax1.set_xlabel('Frequency (\u03C9)/\u03C0')
        ax1.set_ylabel('Magnitude of X[k] (dB)')
        ax1.set_xlim([-1, 1])
        ax1.set_title('DFT using FFT Algorithm')
        # plt.show()

        # Look at Variables
        # print("Length of input samples: {}".format(len(self.samples)))
        # print("Samples: {}".format(samples_dtft[0:int(N/2)]))
        # print("Length of dft: {}".format(N))
        # print("Length of time_positive: {}".format(len(time_positive)))
        # print("First Sample: {}".format(samples_dtft[0]))
        # print("Last Sample: {}".format(samples_dtft[-1]))
        # print("Last Sample: {}".format(samples_dtft[N-1]))
        
        return samples_dtft

    def bandpass_butter(self, Omega_p, Omega_s):

        # Passband Omega_p (Hz)
        # Stopband Omega_s (Hz)

        print(" ---------------- LOWPASS FILTER CHEB II ---------------- ")

        # Initialize IIR Parameters ------------------------------

        omega_p     = (2 * Omega_p) / self.fs # normalized band
        omega_s     = (2 * Omega_s) / self.fs # normalized band
        max_gain_p  = 0 # dB
        min_atten_p = 1 # dB
        ripple_p    = min_atten_p - max_gain_p # dB
        max_atten_s = 50 # dB

        # Generate Filter Parameters
        ord, wn = signal.buttord(omega_p, omega_s, min_atten_p - max_gain_p, max_atten_s)

        # Outputs polynomial coefficients, using the bilinear transform automatically
        b, a = signal.butter(ord, wn, btype = 'bandpass', output='ba')

        # Plot Frequency Response
        w, h = signal.freqz(b, a)
        fig = plt.figure()
        plt.plot(w, 20 * np.log10(abs(h)), color = self.color_1)
        # print(20 * np.log10(abs(h)))
        plt.title('Butter BandPass Fit to Constraints')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude [dB]')
        plt.grid(which='both', axis='both')
        plt.show()

        # Group Delay
        # w, gd = signal.group_delay((b, a))
        # fig = plt.figure()
        # plt.title('Digital filter group delay')
        # plt.plot(w, gd)
        # plt.ylabel('Group delay [samples]')
        # plt.xlabel('Frequency [rad/sample]')
        # plt.show()

        sos = signal.butter(ord, wn, btype = 'bandpass', output='sos')
        self.samples = signal.sosfilt(sos, self.samples)

        print("Chebyshev II Parameters: \na = {} \nb = {} \nord ={} ".format(a,b,ord))
        print("omega_p: {}".format(omega_p))
        print("omega_s: {}".format(omega_s))

    def bandpass_chebII(self, Omega_p, Omega_s):

        # Passband Omega_p (Hz)
        # Stopband Omega_s (Hz)

        print(" ---------------- BANDPASS FILTER CHEB II ---------------- ")

        # Initialize IIR Parameters ------------------------------

        omega_p     = (2 * Omega_p) / self.fs # normalized band
        omega_s     = (2 * Omega_s) / self.fs # normalized band
        max_gain_p  = 0 # dB
        min_atten_p = 1 # dB
        ripple_p    = min_atten_p - max_gain_p # dB
        max_atten_s = 50 # dB

        # Generate Filter Parameters
        ord, ws = signal.cheb2ord(omega_p, omega_s, min_atten_p - max_gain_p, max_atten_s)

        # Outputs polynomial coefficients, using the bilinear transform automatically
        b, a = signal.cheby2(ord, max_atten_s, ws, btype='band',  output='ba')

        # Plot Frequency Response
        w, h = signal.freqz(b, a)
        fig = plt.figure()
        plt.plot(w, 20 * np.log10(abs(h)), color = self.color_1)
        # print(20 * np.log10(abs(h)))
        plt.title('Chebyshev II Fit to Constraints')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude [dB]')
        plt.grid(which='both', axis='both')
        plt.show()

        # Group Delay
        # w, gd = signal.group_delay((b, a))
        # fig = plt.figure()
        # plt.title('Digital filter group delay')
        # plt.plot(w, gd)
        # plt.ylabel('Group delay [samples]')
        # plt.xlabel('Frequency [rad/sample]')
        # plt.show()

        sos = signal.cheby2(ord, max_atten_s, ws, btype='band', output='sos')
        self.samples = signal.filtfilt(b, a, self.samples)

        print("Chebyshev II Parameters: \na = {} \nb = {} \nord ={} ".format(a,b,ord))
        print("omega_p: {}".format(omega_p))
        print("omega_s: {}".format(omega_s))

    def lowpass_chebII(self, Omega_p, Omega_s):

        # Passband Omega_p (Hz)
        # Stopband Omega_s (Hz)

        print(" ---------------- LOWPASS FILTER CHEB II ---------------- ")

        # Initialize IIR Parameters ------------------------------

        omega_p     = (2 * Omega_p) / self.fs # normalized band
        omega_s     = (2 * Omega_s) / self.fs # normalized band
        max_gain_p  = 0 # dB
        min_atten_p = 1 # dB
        ripple_p    = min_atten_p - max_gain_p # dB
        max_atten_s = 50 # dB

        # Generate Filter Parameters
        ord, ws = signal.cheb2ord(omega_p, omega_s, min_atten_p - max_gain_p, max_atten_s)

        # Outputs polynomial coefficients, using the bilinear transform automatically
        b, a = signal.cheby2(ord, max_atten_s, ws, output='ba')

        # Plot Frequency Response
        # w, h = signal.freqz(b, a)
        # fig = plt.figure()
        # plt.plot(w, 20 * np.log10(abs(h)), color = self.color_1)
        # # print(20 * np.log10(abs(h)))
        # plt.title('Chebyshev II Fit to Constraints')
        # plt.xlabel('Frequency [radians / second]')
        # plt.ylabel('Amplitude [dB]')
        # plt.grid(which='both', axis='both')
        # plt.show()

        # Group Delay
        # w, gd = signal.group_delay((b, a))
        # fig = plt.figure()
        # plt.title('Digital filter group delay')
        # plt.plot(w, gd)
        # plt.ylabel('Group delay [samples]')
        # plt.xlabel('Frequency [rad/sample]')
        # plt.show()

        sos = signal.cheby2(ord, max_atten_s, ws, output='sos')
        self.samples = signal.sosfilt(sos, self.samples)

        print("Chebyshev II Parameters: \na = {} \nb = {} \nord ={} ".format(a,b,ord))
        print("omega_p: {}".format(omega_p))
        print("omega_s: {}".format(omega_s))

    def lowpass_parks(self, Omega_p, Omega_s):

        print(" ---------------- LOWPASS FILTER PARKS-MCCLELLAND ---------------- ")

        k       = (10**(-1/20) + 1) / 2
        dpass   = (1/k) - 1
        dstop   = 1/(100 * k * np.sqrt(10))

        omega_p = (2 * Omega_p) / self.fs # normalized band
        omega_s = (2 * Omega_s) / self.fs # normalized band

        print("dpass: {}".format(dpass))
        print("dstop: {}".format(dstop))
        print("omega_p: {}".format(omega_p))
        print("omega_s: {}".format(omega_s))

        numtaps, bands, amps, weights = dtsp.remezord([omega_p/2.0, omega_s/2.0], [1, 0], [dpass,dstop], Hz=1)
        bands *= 2.0    # above function outputs frequencies normalized from 0.0 to 0.5

        print("numtaps: {}".format(numtaps))

        b = signal.remez(numtaps, bands, amps, weights, Hz=2.0)

        print("Parks-McClellan Parameters: \nb = {} \nk = {}".format(b, k))
        print("Length b: {}".format(b.shape))

        # Plot Frequency Response
        fig = plt.figure()
        w, h = signal.freqz(k*b)
        plt.plot(w, 20 * np.log10(abs(h)), color = self.color_1)
        # print(20 * np.log10(abs(k*h)))
        plt.title('Parks-McClellan Fit to Constraints')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Magnitude [dB]')
        plt.grid(which='both', axis='both')
        plt.show()

        # Group Delay
        w, gd = signal.group_delay((b, [1]))
        fig = plt.figure()
        plt.title('Digital filter group delay')
        plt.plot(w, gd)
        plt.ylabel('Group delay [samples]')
        plt.xlabel('Frequency [rad/sample]')
        plt.show()

        # self.samples = signal.filtfilt(k * b, [1], self.samples)
        self.samples = signal.lfilter(k * b, [1], self.samples)

    def lowpass_kaiser(self, Omega_p, Omega_s):

        print(" ---------------- LOWPASS FILTER KAISER WINDOW ---------------- ")

        omega_p = (2 * Omega_p) / self.fs # normalized pass band with units of (pi radians/sample)
        omega_s = (2 * Omega_s) / self.fs # normalized stop band with units of (pi radians/sample)

        delta = .005 # upper bound for the deviation of the magnitude of the freq response
        ripple = -20.0*np.log10(delta) # upper bound for the deviation of the magnitude of the freq response from that of desired in (dB)
        width = omega_s - omega_p # width of transition region with units of (pi radians/sample)
        numtaps, beta = signal.kaiserord(ripple, width)

        # From FIR Tools
        wc = (omega_s + omega_p) /2

        group_delay = numtaps/2
        print("Group Delay: {}".format(group_delay))
        print("Filter Order: {}".format(numtaps))

        b = signal.firwin(numtaps, wc, window = ('kaiser', beta))
        # print("Kaiser Parameters: \nb = {} \nk = {}".format(b, k))

        # Plot Time Domain Response
        # fig = plt.figure()

        # Plot Coefficients
        # plt.plot(b)

        # # Plot Frequency Response
        # w, h = signal.freqz(b)
        # fig = plt.figure()
        # plt.plot(w/np.pi, 20 * np.log10(abs(h)))
        # # plt.plot(w, 20 * np.log10(abs(k*h))) # Took out this k because I don't know what it is [FIXME]
        # # print(20 * np.log10(abs(k*h)))
        # plt.title('Kaiser Fit to Constraints')
        # plt.xlabel('Frequency [radians / second]')
        # plt.ylabel('Amplitude [dB]')
        # plt.grid(which='both', axis='both')
        # plt.show()

        # # Group Delay [FIXME: Does not output correct plot for group delay]
        # w, gd = signal.group_delay((b, [1]))
        # fig = plt.figure()
        # plt.title('Digital filter group delay')
        # plt.plot(w, gd)
        # plt.ylabel('Group delay [samples]')
        # plt.xlabel('Frequency [rad/sample]')
        # plt.show()

        self.samples = signal.lfilter(b, [1], self.samples)
        self.group_delay = group_delay

        return group_delay

    def decimate(self, factor):

        print(" ---------------- DECIMATE ---------------- ")

        # Downsample
        self.samples             = self.samples[::factor] # downsample
        self.fs                  = self.fs / factor # update fs
        self.data_length_sec     = self.samples.shape[0] / self.fs # update total length of data in sec
        self.data_length_samples = self.samples.shape[0] # update total number of samples

        # Look at Variables
        # print("Downsample by 8 Samples: {}".format(self.samples))
        print("Length of input samples: {}".format(self.data_length_samples))

