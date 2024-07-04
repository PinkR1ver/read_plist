import plistlib
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
from scipy import signal

# def read_plist(file_path):
#     with open(file_path, 'rb') as plist_file:
#         data = plistlib.load(plist_file)
#         return data

base_path = os.path.dirname(__file__)



def moving_average_filter(data, window_size):
    window_size = int(window_size)
    ma = np.cumsum(data, dtype=float)
    ma[window_size:] = ma[window_size:] - ma[:-window_size]
    return ma[window_size - 1:] / window_size

def butter_highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def open_file():
    file_path = filedialog.askopenfilename(
        title="选择文件",
        initialdir=base_path,
    )
    
    with open(file_path, 'rb') as plist_file:
        data = plistlib.load(plist_file)    
    
    # for filter_num in [5,10,15,20,30,50,100]:
    #     data_filter = moving_average_filter(data, filter_num)
        
    #     plt.figure()
    #     plt.plot(data_filter)
    #     plt.title(f'moving_average_filter, window_size={filter_num}')
    #     # plt.savefig(f'{os.path.basename(file_path)}_{filter_num}.png')
    
    plt.figure()
    plt.plot(data)
    plt.title(f'original_data')
    
    print(type(data))
    
    filter_data = butter_highpass_filter(data, 0.8, 1000, 5)
    plt.figure()
    plt.plot(filter_data)
    plt.title(f'butter_highpass_filter')
    
    filter_data = butter_lowpass_filter(filter_data, 50, 1000, 5)
    plt.figure()
    plt.plot(filter_data)
    plt.title(f'butter_lowpass_filter')
    
    filter_data = moving_average_filter(filter_data, 100)
    plt.figure()
    plt.plot(filter_data)
    plt.title(f'final_result')

    
    plt.show()
        
    
open_file()