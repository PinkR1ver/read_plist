import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 创建一个简单的信号，例如一个正弦波
fs = 1000  # 采样频率，单位Hz
t = np.linspace(0, 1, fs, endpoint=False)  # 时间向量
signal_original = np.sin(2 * np.pi * 5 * t)  # 5 Hz的正弦波

# 绘制原始信号
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal_original)
plt.title('Original Signal')

# 压缩信号的时间尺度，例如压缩到原来的50%
compressed_ratio = 0.1
fs_new = int(fs * compressed_ratio)  # 新的采样频率
signal_compressed = signal.resample(signal_original, fs_new)

t_new = np.linspace(0, 1, fs_new, endpoint=False)  # 新的时间向量

# 绘制压缩后的信号
plt.subplot(2, 1, 2)
plt.plot(t_new, signal_compressed)
plt.title('Compressed Signal')

plt.tight_layout()
plt.show()