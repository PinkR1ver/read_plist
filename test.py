import numpy as np
from scipy.signal import firwin, lfilter
import matplotlib.pyplot as plt

# 设定滤波器参数
cutoff_freq = 0.2  # 截止频率，根据你的需要调整
fs = 1000  # 采样频率，单位Hz
taps = 100  # 滤波器的阶数

# 设计FIR高通滤波器系数
b = firwin(taps, cutoff_freq=cutoff_freq, window=('hamming',), pass_zero=False)

# 生成测试信号
t = np.linspace(0, 1, fs, endpoint=False)
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t)

# 应用高通滤波器
filtered_signal = lfilter(b, 1, signal)

# 绘制结果
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(t, filtered_signal)
plt.title('High Pass Filtered Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()