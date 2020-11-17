# Load NeuroKit and other useful packages
import matplotlib
import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SAMPLING_RATE = 300.0

# Read in data for training and testing
x_train = pd.read_csv("data/X_train.csv", index_col=0, header=0, low_memory=False)
y_train = pd.read_csv("data/y_train.csv", index_col=0, header=0)
x_test = pd.read_csv("data/X_test.csv", index_col=0, header=0, low_memory=False)

#get lengths of signals for each sample

lengths = []
width = x_train.shape[1]

for row in range(x_train.shape[0]):
    temp_width = width
    for item in x_train.loc[row][::-1]:
        if not pd.isna(item) and (isinstance(item, float) or isinstance(item, int)):
            temp_width -= 1
            break

        temp_width -= 1

    lengths.append(temp_width)


# Retrieve ECG data from data folder (sampling rate= 1000 Hz)
ecg_signal = x_train.loc[4328][:lengths[4328]+1]
ecg_signal = nk.ecg_clean(ecg_signal,SAMPLING_RATE)


# Find peaks
peaks, info = nk.ecg_peaks(ecg_signal, sampling_rate=SAMPLING_RATE)

# Visualize R-peaks in ECG signal
plot = nk.events_plot(info['ECG_R_Peaks'], ecg_signal)
print(info)

# R amplitude
print(info['ECG_R_Peaks'])
R_amplitudes = ecg_signal[info['ECG_R_Peaks']]

data = [[np.mean(R_amplitudes),
         np.median(R_amplitudes),
         np.percentile(R_amplitudes, q=5),
         np.percentile(R_amplitudes, q=95),
         np.std(R_amplitudes)]]

names = ['R_ampl_mean', 'R_ampl_median', 'R_ampl_perc5', 'R_ampl_perc95', 'R_ampl_sd']
features = pd.DataFrame(data, columns=names)

print(features)

# Delineate the ECG signal and visualizing all peaks of ECG complexes
# “ECG_P_Peaks”, “ECG_Q_Peaks”, “ECG_S_Peaks”, “ECG_T_Peaks”, “ECG_P_Onsets”, “ECG_T_Offsets”

_, waves_peak = nk.ecg_delineate(ecg_signal, info, sampling_rate=SAMPLING_RATE, show=True, show_type='peaks')

# Visualize the T-peaks, P-peaks, Q-peaks and S-peaks
plot = nk.events_plot([waves_peak['ECG_T_Peaks'],
                       waves_peak['ECG_P_Peaks'],
                       waves_peak['ECG_Q_Peaks'],
                       waves_peak['ECG_S_Peaks']], ecg_signal)



print(waves_peak)

# Q amplitude

Q_peaks =[peak for peak in waves_peak['ECG_Q_Peaks'] if str(peak) != 'nan']
print(Q_peaks)

Q_amplitudes = ecg_signal[Q_peaks]
print(Q_amplitudes)

data = [[np.mean(Q_amplitudes),
         np.median(Q_amplitudes),
         np.percentile(Q_amplitudes, q=5),
         np.percentile(Q_amplitudes, q=95),
         np.std(Q_amplitudes)]]

names = ['Q_ampl_mean', 'Q_ampl_median', 'Q_ampl_perc5', 'Q_ampl_perc95', 'Q_ampl_sd']
Q_ampl_frame = pd.DataFrame(data, columns=names)

features = pd.concat([features, Q_ampl_frame], axis=1)

print(features)


print(Q_peaks)
print(Q_peaks[:len(Q_peaks)-1])
Q_peaks_diff = [(Q_peaks[index+1] - Q_peaks[index])/SAMPLING_RATE for index, item in enumerate(Q_peaks[:len(Q_peaks)-1])]

print(Q_peaks_diff)

#QQ interval
print(np.mean(Q_peaks_diff))
print(np.median(Q_peaks_diff))
print(np.percentile(Q_peaks_diff, q=5))
print(np.percentile(Q_peaks_diff, q=95))
print(np.std(Q_peaks_diff))

data = [[np.mean(Q_peaks_diff),
         np.median(Q_peaks_diff),
         np.percentile(Q_peaks_diff, q=5),
         np.percentile(Q_peaks_diff, q=95),
         np.std(Q_peaks_diff)]]

names = ['Q_diff_mean', 'Q_diff_median', 'Q_diff_perc5', 'Q_diff_perc95', 'Q_diff_sd']
Q_diff_frame = pd.DataFrame(data, columns=names)

features = pd.concat([features, Q_diff_frame], axis=1)

print(features)


plt.show()


# Extract clean EDA and SCR features
# explanation of features:
# https://neurokit2.readthedocs.io/en/latest/functions.html?highlight=hrv%20time#neurokit2.hrv.hrv_time

hrv_time = nk.hrv_time(peaks, sampling_rate=SAMPLING_RATE, show=True)

features = pd.concat([features, hrv_time], axis=1)

print("feature vec:")
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(features)