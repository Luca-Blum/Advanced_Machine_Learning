import neurokit2 as nk
import numpy as np
import pandas as pd
import pywt
from biosppy.signals import ecg
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

SAMPLING_RATE = 300.0


def create_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    # get lengths of signals for each sample
    lengths = []
    width = dataframe.shape[1]

    for row in dataframe.index.tolist():
        temp_width = width
        for item in dataframe.loc[row][::-1]:
            if not pd.isna(item) and isinstance(item, float):
                temp_width -= 1
                break

            temp_width -= 1

        lengths.append(temp_width)

    """
    README
    
    For the following features we measured: [mean, median, 5 % percentile, 95 % percentile, standard deviation]
    R-peak location were retrieved by nk.ecg_peaks
    Q-peak and S-location were retrieved by nk.ecg_delineate
    
    ?_ampl_*        ?-Peak amplitude
    ?_nr_peaks      number of ?-Peaks
    ?_diff_*        Interval between ?-Peaks
    QRS_diff_*      QRS duration
    len_*           length of signal
    Qual_*          quality of signal measured with nk.ecg_quality
    sign_*          signal
    
    Also the output from nk.hrv_time which contains different measurements for the heart rate variation (HRV*) was added
    
    Additionally one 'typical' heartbeat was greated (all length 180):
    
    MN_*            mean signal
    MD_*            median signal
    P5_*            5 % percentile signal
    P95_*           95 % percentile signal
    SD_*            standard deviation of signal
    """

    names = ['R_ampl_mean', 'R_ampl_median', 'R_ampl_perc5', 'R_ampl_perc95', 'R_ampl_sd', 'R_nr_peaks',
             'len_mean', 'len_median', 'len_perc5', 'len_perc95', 'len_sd',
             'sign_mean', 'sign_median', 'sign_perc5', 'sign_perc95', 'sign_sd',
             'Qual_mean', 'Qual_median', 'Qual_perc5', 'Qual_perc95', 'Qual_sd',
             'Q_ampl_mean', 'Q_ampl_median', 'Q_ampl_perc5', 'Q_ampl_perc95', 'Q_ampl_sd', 'Q_nr_peaks',
             'Q_diff_mean', 'Q_diff_median', 'Q_diff_perc5', 'Q_diff_perc95', 'Q_diff_sd',
             'S_ampl_mean', 'S_ampl_median', 'S_ampl_perc5', 'S_ampl_perc95', 'S_ampl_sd', 'S_nr_peaks',
             'S_diff_mean', 'S_diff_median', 'S_diff_perc5', 'S_diff_perc95', 'S_diff_sd',
             'P_ampl_mean', 'P_ampl_median', 'P_ampl_perc5', 'P_ampl_perc95', 'P_ampl_sd', 'P_nr_peaks',
             'T_ampl_mean', 'T_ampl_median', 'T_ampl_perc5', 'T_ampl_perc95', 'T_ampl_sd', 'T_nr_peaks',
             'QRS_diff_mean', 'QRS_diff_median', 'QRS_diff_perc5', 'QRS_diff_perc95', 'QRS_diff_sd',
             'PR_diff_mean', 'PR_diff_median', 'PR_diff_perc5', 'PR_diff_perc95', 'PR_diff_sd',
             'RT_diff_mean', 'RT_diff_median', 'RT_diff_perc5', 'RT_diff_perc95', 'RT_diff_sd',
             'HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN',
             'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_TINN', 'HRV_HTI',
             'HRV_ULF','HRV_VLF','HRV_LF','HRV_HF','HRV_VHF','HRV_LFHF','HRV_LFn','HRV_HFn', 	'HRV_LnHF',
             'HRV_SD1','HRV_SD2', 'HRV_SD1SD2','HRV_S','HRV_CSI','HRV_CVI','HRV_CSI_Modified', 'HRV_PIP',
             'HRV_IALS','HRV_PSS','HRV_PAS','HRV_GI','HRV_SI','HRV_AI','HRV_PI','HRV_C1d','HRV_C1a','HRV_SD1d',
             'HRV_SD1a','HRV_C2d','HRV_C2a','HRV_SD2d','HRV_SD2a','HRV_Cd','HRV_Ca','HRV_SDNNd','HRV_SDNNa','HRV_ApEn',
             'HRV_SampEn','J_LF','J_HF','J_L/H']


    template_len = 180

    mean_names = ['MN_' + str(index) for index in range(template_len)]
    median_names = ['MD_' + str(index) for index in range(template_len)]
    perc5_names = ['P5_' + str(index) for index in range(template_len)]
    perc95_names = ['P95_' + str(index) for index in range(template_len)]
    sd_names = ['SD_' + str(index) for index in range(template_len)]

    wavelet = 'db3'

    wl_len = int(np.floor((template_len + pywt.Wavelet(wavelet).dec_len - 1) / 2))

    wl_mean_names = ['WLMN_' + str(index) for index in range(2*wl_len)]
    wl_median_names = ['WLMD_' + str(index) for index in range(2*wl_len)]
    wl_perc5_names = ['WLP5_' + str(index) for index in range(2*wl_len)]
    wl_perc95_names = ['WLP95_' + str(index) for index in range(2*wl_len)]
    wl_sd_names = ['WLSD_' + str(index) for index in range(2*wl_len)]

    typical_signal_names = mean_names + median_names + perc5_names + perc95_names + sd_names + wl_mean_names + \
                           wl_median_names + wl_perc5_names + wl_perc95_names + wl_sd_names

    names += typical_signal_names

    data = np.empty([dataframe.shape[0], len(names)])

    iteration = 0
    for row_index, row in dataframe.iterrows():
        print(row_index)

        # Retrieve ECG data
        ecg_signal = row[:lengths[iteration] + 1]
        ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=SAMPLING_RATE)

        # Find R-peaks
        peaks, info = nk.ecg_peaks(ecg_signal, sampling_rate=SAMPLING_RATE)

        # R amplitude
        R_amplitudes = ecg_signal[info['ECG_R_Peaks']]

        # Check if the signal is flipped
        # Check if we have enough peaks to retrieve more information
        if len(R_amplitudes) > 4:

            _, waves_peak = nk.ecg_delineate(ecg_signal, info, sampling_rate=300, show=False)

            # Q amplitude

            # remove nan values
            Q_amplitudes = [ecg_signal[peak_index] if str(peak_index) != 'nan' else - np.infty for peak_index in
                            waves_peak['ECG_Q_Peaks']]

            if np.sum([1 if np.abs(rpeak) > np.abs(Q_amplitudes[index]) else -1 for index, rpeak in
                       enumerate(R_amplitudes)]) < 0:
                print("flip", row_index)

                ecg_signal = -ecg_signal

                peaks, info = nk.ecg_peaks(ecg_signal, sampling_rate=300)

                # R amplitude
                R_amplitudes = ecg_signal[info['ECG_R_Peaks']]

                if len(R_amplitudes) > 4:
                    _, waves_peak = nk.ecg_delineate(ecg_signal, info, sampling_rate=300, show=False)

        data_temp = []
        if len(R_amplitudes) > 0:
            data_temp = [np.mean(R_amplitudes),
                         np.median(R_amplitudes),
                         np.percentile(R_amplitudes, q=5),
                         np.percentile(R_amplitudes, q=95),
                         np.std(R_amplitudes),
                         len(R_amplitudes)]
        else:
            empty = np.empty([6])
            empty[:] = np.NaN
            data_temp += empty.tolist()

        # length of signal
        data_new = [np.mean(lengths[iteration] / SAMPLING_RATE),
                    np.median(lengths[iteration] / SAMPLING_RATE),
                    np.percentile(lengths[iteration] / SAMPLING_RATE, q=5),
                    np.percentile(lengths[iteration] / SAMPLING_RATE, q=95),
                    np.std(lengths[iteration] / SAMPLING_RATE)]

        data_temp += data_new

        # signal
        data_new = [np.mean(ecg_signal),
                    np.median(ecg_signal),
                    np.percentile(ecg_signal, q=5),
                    np.percentile(ecg_signal, q=95),
                    np.std(ecg_signal)]

        data_temp += data_new

        # Check if we have enough peaks to retrieve more information
        if len(R_amplitudes) > 4:

            quality = nk.ecg_quality(ecg_signal, sampling_rate=SAMPLING_RATE)
            data_new = [np.mean(quality),
                        np.median(quality),
                        np.percentile(quality, q=5),
                        np.percentile(quality, q=95),
                        np.std(quality)]

            data_temp += data_new

            # Delineate the ECG signal
            # “ECG_P_Peaks”, “ECG_Q_Peaks”, “ECG_S_Peaks”, “ECG_T_Peaks”, “ECG_P_Onsets”, “ECG_T_Offsets”

            # _, waves_peak = nk.ecg_delineate(ecg_signal, info, sampling_rate=SAMPLING_RATE, show=False)

            # Q amplitude

            # remove nan values
            Q_peaks = [peak for peak in waves_peak['ECG_Q_Peaks'] if str(peak) != 'nan']

            if len(Q_peaks) > 0:
                Q_amplitudes = ecg_signal[Q_peaks]

                data_new = [np.mean(Q_amplitudes),
                            np.median(Q_amplitudes),
                            np.percentile(Q_amplitudes, q=5),
                            np.percentile(Q_amplitudes, q=95),
                            np.std(Q_amplitudes),
                            len(Q_amplitudes)]

                data_temp += data_new
            else:
                empty = np.empty([6])
                empty[:] = np.NaN
                empty[5] = 0
                data_temp += empty.tolist()

            # more than 1 Q-Peak => can build interval[s]
            if len(Q_peaks) > 1:
                Q_peaks_diff = [(Q_peaks[index + 1] - Q_peaks[index]) / SAMPLING_RATE
                                for index, item in enumerate(Q_peaks[:len(Q_peaks) - 1])]

                # QQ interval

                data_new = [np.mean(Q_peaks_diff),
                            np.median(Q_peaks_diff),
                            np.percentile(Q_peaks_diff, q=5),
                            np.percentile(Q_peaks_diff, q=95),
                            np.std(Q_peaks_diff)]

                data_temp += data_new

            # 0 or 1 Q-peak = no interval => return nan
            else:
                empty = np.empty([5])
                empty[:] = np.NaN
                data_temp += empty.tolist()

            # S amplitude

            # remove nan values
            S_peaks = [peak for peak in waves_peak['ECG_S_Peaks'] if str(peak) != 'nan']

            if len(S_peaks) > 0:
                S_amplitudes = ecg_signal[S_peaks]

                data_new = [np.mean(S_amplitudes),
                            np.median(S_amplitudes),
                            np.percentile(S_amplitudes, q=5),
                            np.percentile(S_amplitudes, q=95),
                            np.std(S_amplitudes),
                            len(S_amplitudes)]

                data_temp += data_new

            else:
                empty = np.empty([6])
                empty[:] = np.NaN
                empty[5] = 0
                data_temp += empty.tolist()

            # more than one S-peak
            if len(S_peaks) > 1:
                S_peaks_diff = [(S_peaks[index + 1] - S_peaks[index]) / SAMPLING_RATE
                                for index, item in enumerate(S_peaks[:len(S_peaks) - 1])]

                # SS interval

                data_new = [np.mean(S_peaks_diff),
                            np.median(S_peaks_diff),
                            np.percentile(S_peaks_diff, q=5),
                            np.percentile(S_peaks_diff, q=95),
                            np.std(S_peaks_diff)]

                data_temp += data_new

            # 0 or 1 S-peak = no interval => return nan
            else:
                empty = np.empty([5])
                empty[:] = np.NaN
                data_temp += empty.tolist()

            P_peaks = [peak for peak in waves_peak['ECG_P_Peaks'] if str(peak) != 'nan']

            if len(P_peaks) > 0:
                P_amplitudes = ecg_signal[P_peaks]

                data_new = [np.mean(P_amplitudes),
                            np.median(P_amplitudes),
                            np.percentile(P_amplitudes, q=5),
                            np.percentile(P_amplitudes, q=95),
                            np.std(P_amplitudes),
                            len(P_amplitudes)]

                data_temp += data_new

            else:
                empty = np.empty([6])
                empty[:] = np.NaN
                empty[5] = 0
                data_temp += empty.tolist()

            T_peaks = [peak for peak in waves_peak['ECG_T_Peaks'] if str(peak) != 'nan']

            if len(T_peaks) > 0:
                T_peaks = ecg_signal[T_peaks]

                data_new = [np.mean(T_peaks),
                            np.median(T_peaks),
                            np.percentile(T_peaks, q=5),
                            np.percentile(T_peaks, q=95),
                            np.std(T_peaks),
                            len(T_peaks)]

                data_temp += data_new

            else:
                empty = np.empty([6])
                empty[:] = np.NaN
                empty[5] = 0
                data_temp += empty.tolist()


            # QRS interval

            QRS_peaks_diff = []

            # compute difference between Q and S peak
            for index in range(len(waves_peak['ECG_Q_Peaks'])):
                if not (np.isnan(waves_peak['ECG_Q_Peaks'][index]) or np.isnan(waves_peak['ECG_S_Peaks'][index])):
                    QRS_peaks_diff.append(
                        (waves_peak['ECG_S_Peaks'][index] - waves_peak['ECG_Q_Peaks'][index]) / SAMPLING_RATE)

            if len(QRS_peaks_diff) > 0:
                data_new = [np.mean(QRS_peaks_diff),
                            np.median(QRS_peaks_diff),
                            np.percentile(QRS_peaks_diff, q=5),
                            np.percentile(QRS_peaks_diff, q=95),
                            np.std(QRS_peaks_diff)]

                data_temp += data_new

            else:
                empty = np.empty([5])
                empty[:] = np.NaN
                data_temp += empty.tolist()

            # PR interval

            PR_peaks_diff = []

            # compute difference between P and R peak
            for index in range(len(waves_peak['ECG_P_Peaks'])):
                if not np.isnan(waves_peak['ECG_P_Peaks'][index]):
                    PR_peaks_diff.append(
                        (info['ECG_R_Peaks'][index] - waves_peak['ECG_P_Peaks'][index]) / SAMPLING_RATE)

            if len(PR_peaks_diff) > 0:
                data_new = [np.mean(PR_peaks_diff),
                            np.median(PR_peaks_diff),
                            np.percentile(PR_peaks_diff, q=5),
                            np.percentile(PR_peaks_diff, q=95),
                            np.std(PR_peaks_diff)]

                data_temp += data_new
            else:
                empty = np.empty([5])
                empty[:] = np.NaN
                data_temp += empty.tolist()

            # RT interval

            RT_peaks_diff = []

            # compute difference between P and R peak
            for index in range(len(waves_peak['ECG_T_Peaks'])):
                if not np.isnan(waves_peak['ECG_T_Peaks'][index]):
                    RT_peaks_diff.append(
                        (waves_peak['ECG_T_Peaks'][index] - info['ECG_R_Peaks'][index]) / SAMPLING_RATE)

            if len(RT_peaks_diff) > 0:
                data_new = [np.mean(RT_peaks_diff),
                            np.median(PR_peaks_diff),
                            np.percentile(RT_peaks_diff, q=5),
                            np.percentile(RT_peaks_diff, q=95),
                            np.std(RT_peaks_diff)]

                data_temp += data_new

            else:
                empty = np.empty([5])
                empty[:] = np.NaN
                data_temp += empty.tolist()

            # Extract clean EDA and SCR features
            # explanation of features:
            # https://neurokit2.readthedocs.io/en/latest/functions.html?highlight=hrv%20time#neurokit2.hrv.hrv_time

            hrv_time = nk.hrv(peaks, sampling_rate=SAMPLING_RATE, show=False)

            data_new = hrv_time.values.tolist()[0]

            data_temp += data_new

            # Jannik
            # http://www.paulvangent.com/2016/03/21/analyzing-a-discrete-heart-rate-signal-using-python-part-2/
            rpeaks = info['ECG_R_Peaks']
            r_interval = [rpeaks[index+1]-rpeaks[index] for index in range(len(rpeaks)-1)]
            RR_x_new = np.linspace(rpeaks[0],rpeaks[-2],rpeaks[-2])
            f = interp1d(rpeaks[:-1], r_interval, kind='cubic')

            n = lengths[iteration] + 1 # Length of the signal
            frq = np.fft.fftfreq(n, d=(1 / SAMPLING_RATE)) # divide the bins into frequency categories
            frq = frq[range(int(n/2))] # Get single side of the frequency range

            Y = np.fft.fft(f(RR_x_new))/n # Calculate FFT

            try:
                Y = Y[range(int(n / 2))]
                lf = np.trapz(abs(Y[(frq >= 0.04) & (frq <= 0.15)]))

                hf = np.trapz(abs(Y[(frq >= 0.16) & (frq <= 0.5)]))  # Do the same for 0.16-0.5Hz (HF)

                data_new = [lf, hf, lf / hf]

                data_temp += data_new
            except IndexError as err:
                print(err)
                data_temp += [None, None, None]

        # if we don't have enough R peaks return vector of nan's
        else:
            empty = np.empty([len(names) - 16 - len(typical_signal_names)])
            empty[:] = np.NaN
            data_temp += empty.tolist()

        # Create a 'typical' heartbeat

        # Scaler = StandardScaler()
        # ecg_signal = Scaler.fit_transform(X=ecg_signal.reshape(-1, 1)).reshape(1, -1)[0].tolist()

        out = ecg.ecg(signal=ecg_signal, sampling_rate=SAMPLING_RATE, show=False)

        mean = np.mean(out['templates'], axis=0)
        median = np.median(out['templates'], axis=0)
        perc5 = np.percentile(out['templates'].astype(np.float64), axis=0, q=5)
        perc95 = np.percentile(out['templates'].astype(np.float64), axis=0, q=95)
        std = np.std(out['templates'].astype(np.float64), axis=0)

        data_new = np.concatenate((mean, median, perc5, perc95, std)).tolist()

        data_temp += data_new

        (wl_mean_cA, wl_mean_cD) = pywt.dwt(np.mean(out['templates'], axis=0),
                                            'db3', 'periodic')
        (wl_median_cA, wl_median_cD) = pywt.dwt(np.median(out['templates'], axis=0),
                                                'db3', 'periodic')
        (wl_perc5_cA, wl_perc5_cD) = pywt.dwt(np.percentile(out['templates'].astype(np.float64), axis=0, q=5),
                                              'db3', 'periodic')
        (wl_perc95_cA, wl_perc95_cD) = pywt.dwt(np.percentile(out['templates'].astype(np.float64), axis=0, q=95),
                                                'db3', 'periodic')
        (wl_sd_cA, wl_sd_cD) = pywt.dwt(np.std(out['templates'].astype(np.float64), axis=0),
                                        'db3', 'periodic')

        data_new = np.concatenate((wl_mean_cA, wl_mean_cD,
                                   wl_median_cA, wl_median_cD,
                                   wl_perc5_cA, wl_perc5_cD,
                                   wl_perc95_cA, wl_perc95_cD,
                                   wl_sd_cA, wl_sd_cD)).tolist()

        data_temp += data_new

        data[iteration] = data_temp

        iteration += 1

    features = pd.DataFrame(data, columns=names)

    return features


# Read data for training and testing
x_train = pd.read_csv("data/X_train.csv", index_col=0, header=0, low_memory=False,
                      dtype=float, na_values=['', '\n', '\\n'])
y_train = pd.read_csv("data/y_train.csv", index_col=0, header=0)
x_test = pd.read_csv("data/X_test.csv", index_col=0, header=0, low_memory=False,
                     dtype=float, na_values=['', '\n', '\\n'])

# Feature extraction
x_train_features = create_df(x_train)
x_train_features.to_csv('data4/X_train_features.csv')

x_test_features = create_df(x_test)
x_test_features.to_csv('data4/X_test_features.csv')
