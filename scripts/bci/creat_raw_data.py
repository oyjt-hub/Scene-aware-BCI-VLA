import mne
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def CreateRawData(ori_data, sample_rate):

    """
    接收EEG脑电信号并进行处理
    ori_data = (N_c, N_l)
    返回一个(N_c x N_l)的矩阵
    :param ori_data: ndarry, (N_c, N_l)
    :param sample_rate: int -> 1000
    :return: raw, 降采样至250Hz
    """
    # w1 = [2 * 3 / sample_rate]
    # w2 = [2 * 90 / sample_rate]
    # filter_order = 5
    # b, a = signal.butter(filter_order, [w1, w2], 'bandpass')
    # ori_data = signal.filtfilt(b, a, ori_data, axis=-1)
    # info = mne.create_info(
    #     ch_names=['Pz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO1', 'PO3', 'PO4', 'PO5',
    #                               'Oz', 'O1', 'O2'],
    #     ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg','eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg'],
    #     sfreq=sample_rate
    # )
    # 9 channels
    info = mne.create_info(
        ch_names=['Pz', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'O1', 'Oz', 'O2'],
        ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg','eeg'],
        sfreq=sample_rate
    )
    raw_date = mne.io.RawArray(ori_data, info)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw_date.set_montage(montage)

    # 消除基线漂移
    raw_date.filter(l_freq=0.3, h_freq=90)
    # 陷波滤波
    raw_date.notch_filter(freqs=50)

    # 降采样
    raw_date.resample(sfreq=250)
    data = raw_date.get_data()

    return data
#
#
# def Draw_Raw_Data(raw_data):
#
#     scalings = {'eeg': 2}
#
#     # 原始脑电图
#     raw_data.plot(n_channels=5, scalings=scalings, title='Data from arrays', show=True, block=True)
#     plt.show()
#
#
# def Draw_All_Data(raw_data):
#     # 叠加脑电图
#     sfreq = raw_data.info['sfreq']
#     data, times = raw_data[:, int(sfreq * 0):int(sfreq * 1)]
#     plt.title("Sample channels")
#     plt.plot(times, data.T)
#     plt.show()
#
#
# def Draw_Raw_Psd(raw_data):
#     # 脑电功率图
#     raw_data.plot_psd(area_mode='range', average=False)
#     plt.show()
#
#
# def Draw_Raw_Sensors(raw_data):
#     # 电极位置图
#     raw_data.plot_sensors(ch_type='eeg', show_names=True)
#     plt.show()


if __name__ == "__main__":

    N_c = 8
    N_l = 3000
    trail_data = np.random.rand(N_c, N_l)

    raw_data = CreateRawData(trail_data, sample_rate=1000)

    # Draw_Raw_Data(raw_data)

    print(raw_data.get_data().shape)
    print(raw_data.info['sfreq'])

    # Draw_Raw_Psd(raw_data)

    # eeg = raw_data.get_data()  # 一个二维数组

    # print(eeg.shape)
