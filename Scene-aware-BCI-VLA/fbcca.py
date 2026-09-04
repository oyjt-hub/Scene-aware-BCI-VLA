import sys
# sys.path.append(r"C:\Users\32670\Desktop\脑控\yjj (2)\yjj\BCI")
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.cross_decomposition import CCA


class FBCCA(object):

    def __init__(self) -> None:

        self.ref_signals = None
        self.correlated_matrix = None
        self.predicted_result = None

    def filter(self, data, n_subbanks):

        """
        :param data ndarray, (N_c, N_l)
        :param n_subbanks int, 1, ..., 10
        :return filtered_data ndarray, (n_subanks, N_c, N_l)
        """

        down_samplerate = 250

        filtered_data = np.zeros((n_subbanks, data.shape[0], data.shape[1]))
        nyq = down_samplerate / 2

        pass_band = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78][0:n_subbanks]
        stop_band = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72][0:n_subbanks]
        high_cut_pass, high_cut_stop = 90, 100

        gpass, gstop, rp = 2, 40, 0.3

        for i in range(n_subbanks):
            wp = np.array([pass_band[i] / nyq, high_cut_pass / nyq])
            ws = np.array([stop_band[i] / nyq, high_cut_stop / nyq])
            [n, wn] = signal.cheb1ord(wp, ws, gpass, gstop)
            [b, a] = signal.cheby1(n, rp, wn, 'bandpass')
            data = signal.filtfilt(b, a, data, padlen=3 * (max(len(b), len(a)) - 1)).copy()
            filtered_data[i, :, :] = data

        return filtered_data

    def calculate_correlation_matrix(self, filtered_data, n_components=1):
        """
        :param filtered_data ndarray, (n_subbanks, N_c, N_l)
        :retrun correlated_matrix ndarray, (len(Nf_list), n_subbanks)
        """
        cca = CCA(n_components)
        correlated_matrix = np.zeros((self.ref_signals.shape[0], filtered_data.shape[0]))

        for freq_idx in range(self.ref_signals.shape[0]):
            for subbank_idx in range(filtered_data.shape[0]):
                matched_X = filtered_data[subbank_idx, :, :]
                # print(matched_X.shape, self.ref_signals[freq_idx, :, :].shape)
                cca.fit(matched_X.T, self.ref_signals[freq_idx, :, :].T)
                x_a, y_b = cca.transform(matched_X.T, self.ref_signals[freq_idx, :].T)
                corr = np.zeros(n_components)
                for i in range(0, n_components):
                    corr[i] = np.corrcoef(x_a[:, i], y_b[:, i])[0, 1]
                    correlated_matrix[freq_idx, subbank_idx] = np.max(corr)
        self.correlated_matrix = correlated_matrix

    def fbcca(self, data, n_subbanks):

        fb_coefs = [math.pow(i, -1.25) + 0.25 for i in range(1, n_subbanks + 1)]
        filtered_data = self.filter(data, n_subbanks)
        self.calculate_correlation_matrix(filtered_data)

        predicted_list = []
        for i in range(self.correlated_matrix.shape[0]):
            cor_vector = self.correlated_matrix[i, :]
            result = 0
            for fb_i in range(0, n_subbanks):
                w = fb_coefs[fb_i]
                rho = cor_vector[fb_i]
                result += w * (rho ** 2)
            predicted_list.append(result)
        self.predicted_result = np.array(predicted_list)

        return np.argmax(self.predicted_result)

    def get_filtered_data(self, data, n_subbanks):
        fb_coefs = [math.pow(i, -1.25) + 0.25 for i in range(1, n_subbanks + 1)]
        return self.filter(data, n_subbanks)



    def get_reference_signals(self, Nf_list, N_t, num_harmonics=3, down_samplerate=250):
        """
        :param Nf_list list
        :param N_l int, length of eeg signal
        :param num_harmonics int
        """

        t = np.arange(0, N_t, 1 / 250)
        ref_signals = []

        for f in Nf_list:
            ref_f = []
            for h in range(num_harmonics):
                sin_ref_sings = np.sin(2 * np.pi * (h + 1) * f * t)
                ref_f.append(sin_ref_sings)
                cos_ref_sings = np.cos(2 * np.pi * (h + 1) * f * t)
                ref_f.append(cos_ref_sings)
            ref_signals.append(ref_f)

        self.ref_signals = np.array(ref_signals)


if __name__ == '__main__':

    """
    calculate accuracy by fbcca
    """

    # test_mat_file = scio.loadmat('/home/yujinjie/workpace/YuJinjie_BCI/Data/SSVEP_Former/Benchmark/test/34_35member.mat')
    # test_data = test_mat_file['data']
    # test_label = test_mat_file['label'][:, 1]

    # Nf_list = [round(8.6 + 0.6 * s, 1) for s in range(12)]
    # fbcca = FBCCA()

    # ac = []
    # for i in range(1, 21):
    #     data_len = round(0.2 + i * 0.1, 2)
    #     fbcca.get_reference_signals(Nf_list, data_len)

    #     ac_i = 0
    #     for j in range(test_data.shape[0]):
    #         signs = test_data[j, :, int(0.65 * 250):int((data_len + 0.65) * 250)]
    #         pred_label = fbcca.fbcca(signs, 4)
    #         if pred_label == test_label[j]:
    #             ac_i = ac_i + 1
    #     ac_i = ac_i / test_data.shape[0]
    #     ac.append(ac_i)
    # print(ac)
    plt.figure(dpi=300, figsize=(5, 10))
    Benchmark_ac = [0.2013888888888889, 0.1875, 0.3125, 0.4652777777777778, 0.6111111111111112, 0.7013888888888888, 0.7916666666666666, 0.875, 0.9027777777777778, 0.9236111111111112, 0.9375, 0.9583333333333334, 0.9652777777777778, 0.9652777777777778, 0.9513888888888888, 0.9652777777777778, 0.9791666666666666, 0.9791666666666666, 0.9791666666666666, 0.9861111111111112]
    BETA_ac = [0.21666666666666667, 0.31666666666666665, 0.3875, 0.49722222222222223, 0.5847222222222223, 0.6444444444444445, 0.6944444444444444, 0.7291666666666666, 0.7625, 0.7944444444444444, 0.8236111111111111, 0.8416666666666667, 0.8680555555555556, 0.8819444444444444, 0.8930555555555556, 0.9138888888888889, 0.9166666666666666, 0.9097222222222222, 0.9180555555555555, 0.9222222222222223]
    x = [round(0.3 + i * 0.1, 2) for i in range(len(Benchmark_ac))]
    plt.plot(x, Benchmark_ac, '-x', color='g', label='Benchmark Dataset')
    plt.plot(x, BETA_ac, '-v', color='b', label='BETA Dataset')
    plt.title('from 0.3 Hz to 2.2 Hz accuracy')
    plt.ylim((0, 1.2))
    plt.xlim((0.25, 2.25))
    x_label = ['{}s'.format(round(0.3 + i * 0.1, 2)) for i in range(len(Benchmark_ac))]
    # print(x_label)
    # print(np.arange(0.3, 2.3, 0.1))
    plt.yticks(np.arange(0, 1.02, 0.1), fontsize=5)
    plt.xticks(np.arange(0.3, 2.3, 0.1), x_label, fontsize=5)
    plt.xlabel('Data length')
    plt.ylabel('Accuracy')
    num1, num2, num3, num4 = 0.78, 0.01, 3, 0
    legend_font = {'size': 5}
    plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4, prop=legend_font)
    plt.savefig(f'fbcca_ac.png')

    plt.show()

    # BETA : ac = [0.21666666666666667, 0.31666666666666665, 0.3875, 0.49722222222222223, 0.5847222222222223, 0.6444444444444445, 0.6944444444444444, 0.7291666666666666, 0.7625, 0.7944444444444444, 0.8236111111111111, 0.8416666666666667, 0.8680555555555556, 0.8819444444444444, 0.8930555555555556, 0.9138888888888889, 0.9166666666666666, 0.9097222222222222, 0.9180555555555555, 0.9222222222222223]
    # Benchmark : ac = [0.2013888888888889, 0.1875, 0.3125, 0.4652777777777778, 0.6111111111111112, 0.7013888888888888, 0.7916666666666666, 0.875, 0.9027777777777778, 0.9236111111111112, 0.9375, 0.9583333333333334, 0.9652777777777778, 0.9652777777777778, 0.9513888888888888, 0.9652777777777778, 0.9791666666666666, 0.9791666666666666, 0.9791666666666666, 0.9861111111111112]