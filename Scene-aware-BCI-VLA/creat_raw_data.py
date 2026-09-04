import mne
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def CreateRawData(ori_data, sample_rate):
    """
    接收EEG脑电信号并进行处理
    ori_data = (N_c, N_l)
    返回一个(N_c x N_l)的矩阵
    :param ori_data: ndarray, (N_c, N_l)
    :param sample_rate: int -> 1000
    :return: raw, 降采样至250Hz
    """
    # 确保 ori_data 的通道数与 info["ch_names"] 的通道数一致
    if ori_data.shape[0] != 9:
        raise ValueError(f"ori_data 的通道数 {ori_data.shape[0]} 不等于 info['ch_names'] 的通道数 9")

    # 9 channels
    info = mne.create_info(
        ch_names=['Pz', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'O1', 'Oz', 'O2'],
        ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg'],
        sfreq=sample_rate
    )
    raw_date = mne.io.RawArray(ori_data, info)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw_date.set_montage(montage)

    # 消除基线漂移
    raw_date.filter(l_freq=0.3, h_freq=90, filter_length='auto', phase='zero-double')
    #raw_date.filter(l_freq=4, h_freq=90, filter_length='1s', phase='zero-double')

    # 陷波滤波
    raw_date.notch_filter(freqs=50, filter_length='auto', phase='zero-double')
    #raw_date.notch_filter(freqs=50, filter_length='0.5s', phase='zero-double')

    # 降采样
    raw_date.resample(sfreq=250)
    data = raw_date.get_data()

    return raw_date  # 返回 RawArray 对象

def Draw_Raw_Data(raw_data):
    scalings = {'eeg': 2}
    raw_data.plot(n_channels=9, scalings=scalings, title='Data from arrays', show=True, block=True)
    plt.show()

def Draw_All_Data(raw_data):
    sfreq = raw_data.info['sfreq']
    data, times = raw_data[:, int(sfreq * 0):int(sfreq * 1)]
    plt.title("Sample channels")
    plt.plot(times, data.T)
    plt.show()

def Draw_Raw_Psd(raw_data):
    raw_data.plot_psd(area_mode='range', average=False)
    plt.show()

def Draw_Raw_Sensors(raw_data):
    raw_data.plot_sensors(ch_type='eeg', show_names=True)
    plt.show()

if __name__ == "__main__":
    N_c = 9  # 确保通道数为 9
    N_l = 3000
    trail_data = np.random.rand(N_c, N_l)

    raw_data = CreateRawData(trail_data, sample_rate=1000)

    # 绘制原始数据
    Draw_Raw_Data(raw_data)

    # 绘制所有数据
    Draw_All_Data(raw_data)

    # 绘制功率谱密度
    Draw_Raw_Psd(raw_data)

    # 绘制电极位置图
    Draw_Raw_Sensors(raw_data)

    # 打印数据形状和采样率
    print(raw_data.get_data().shape)
    print(raw_data.info['sfreq'])