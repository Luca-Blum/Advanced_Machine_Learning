import neurokit2 as nk
import numpy as np
import pandas as pd

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
    
    Also the output from nk.hrv_time which contains different measurements for the heart rate variation (HRV*) was added
    
    """

    names = ['R_ampl_mean', 'R_ampl_median', 'R_ampl_perc5', 'R_ampl_perc95', 'R_ampl_sd', 'R_nr_peaks',
             'len_mean', 'len_median', 'len_perc5', 'len_perc95', 'len_sd',
             'Qual_mean', 'Qual_median', 'Qual_perc5', 'Qual_perc95', 'Qual_sd',
             'Q_ampl_mean', 'Q_ampl_median', 'Q_ampl_perc5', 'Q_ampl_perc95', 'Q_ampl_sd', 'Q_nr_peaks',
             'Q_diff_mean', 'Q_diff_median', 'Q_diff_perc5', 'Q_diff_perc95', 'Q_diff_sd',
             'S_ampl_mean', 'S_ampl_median', 'S_ampl_perc5', 'S_ampl_perc95', 'S_ampl_sd', 'S_nr_peaks',
             'S_diff_mean', 'S_diff_median', 'S_diff_perc5', 'S_diff_perc95', 'S_diff_sd',
             'QRS_diff_mean', 'QRS_diff_median', 'QRS_diff_perc5', 'QRS_diff_perc95', 'QRS_diff_sd',
             'HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN',
             'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_TINN', 'HRV_HTI']

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
        data_new = [np.mean(lengths[iteration]/SAMPLING_RATE),
                    np.median(lengths[iteration]/SAMPLING_RATE),
                    np.percentile(lengths[iteration]/SAMPLING_RATE, q=5),
                    np.percentile(lengths[iteration]/SAMPLING_RATE, q=95),
                    np.std(lengths[iteration]/SAMPLING_RATE)]

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

            _, waves_peak = nk.ecg_delineate(ecg_signal, info, sampling_rate=SAMPLING_RATE, show=False)

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

        # if we don't have enough R peaks return vector of nan's
        else:
            empty = np.empty([len(names) - 25])
            empty[:] = np.NaN
            data_temp += empty.tolist()

        # Extract clean EDA and SCR features
        # explanation of features:
        # https://neurokit2.readthedocs.io/en/latest/functions.html?highlight=hrv%20time#neurokit2.hrv.hrv_time

        hrv_time = nk.hrv_time(peaks, sampling_rate=SAMPLING_RATE, show=False)

        data_new = hrv_time.values.tolist()[0]

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

#Feature extraction
x_train_features = create_df(x_train)
x_train_features.to_csv('data/x_train_features.csv')

x_test_features = create_df(x_test)
x_test_features.to_csv('data/x_test_features.csv')
