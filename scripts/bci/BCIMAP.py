import mne
import matplotlib.pyplot as plt

# 随便建一个占位电极，只为了让 MNE 启动画图程序
info = mne.create_info(ch_names=['Cz'], sfreq=1000, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# 1. 画图（关闭名字显示，维持之前的标准头型比例）
fig = mne.viz.plot_sensors(info, show_names=False, kind='topomap', sphere=(0, 0, 0, 0.105))

# 2. 隐藏所有的黑点！
fig.axes[0].collections[0].set_visible(False)

# 3. 保存为 SVG 矢量图（PPT 最爱）和 PNG 高清图
fig.savefig('blank_head.svg', format='svg', bbox_inches='tight', transparent=True)
fig.savefig('blank_head.png', format='png', dpi=300, bbox_inches='tight', transparent=True)

print("✅ 空白头型底图已生成！请使用 blank_head.svg 插入 PPT。")