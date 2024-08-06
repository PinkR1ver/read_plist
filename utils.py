import streamlit as st
import os
import plistlib
import bokeh.core.properties
from bokeh.plotting import figure
from bokeh.io.export import get_svgs
import bokeh.settings
import bokeh
import numpy as np
from scipy import signal
import pandas as pd
from zipfile import ZipFile

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

def average_downsample(data, window_size):
    for i in range(0, len(data), window_size):
        data[i:i+window_size] = np.mean(data[i:i+window_size])
        
    data = data[::window_size]
    return data

def get_key_points(data):
    
    key_points = [0]
    for i in range(1, len(data)-1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            key_points.append(i)
        elif data[i] < data[i-1] and data[i] < data[i+1]:
            key_points.append(i)
            
    key_points.append(len(data)-1)
    
    first_derivative = np.diff(data)
    second_derivative = np.diff(first_derivative)
    
    index = np.where((np.sign(second_derivative[:-1]) != np.sign(second_derivative[1:])) & (second_derivative[:-1] != 0) & (second_derivative[1:] != 0))[0] + 1
    key_points = list(set(key_points + index.tolist()))
    key_points.sort()
    
    return key_points
    

def data_connection(data, key_ponits_list):
    
    result = data.copy()

    for i in range(1, len(key_ponits_list)):
        result[key_ponits_list[i-1]:key_ponits_list[i] + 1] = np.linspace(data[key_ponits_list[i-1]], data[key_ponits_list[i]], key_ponits_list[i] - key_ponits_list[i-1] + 1)
    
    return result


def quantification(data, threshold, key_points_list):
    
    for i in range(0, len(key_points_list)):
        data[key_points_list[i]] = np.round(data[key_points_list[i]] / threshold) * threshold
        
    data_connect = data_connection(data, key_points_list)
        
    return data_connect


def get_diff(data):
    
    diff = np.diff(data)
    diff = np.insert(diff, 0, 0)
    
    return diff

def data_compress(data, compressed_ratio, key_points_list):
    
    key_points_cp = []
    for key_points in key_points_list:
        key_points_cp.append((key_points, data[key_points]))
    
    for i in range(0, len(key_points_cp)):
        key_points_cp[i] = (int(key_points_cp[i][0] * compressed_ratio), key_points_cp[i][1])
    
    data_len = key_points_cp[-1][0]
    
    data_cp = np.zeros(data_len + 1)
    for i in range(1, len(key_points_cp) - 1):
        left = key_points_cp[i-1][0]
        right = key_points_cp[i][0] + 1
        
        if right - left == 2:
            data_cp[left] = key_points_cp[i-1][1]
            continue
        
        left_start = key_points_cp[i-1][1]
        right_stop = key_points_cp[i][1]
        data_cp[left:right] = np.linspace(left_start, right_stop, right - left)
    
    key_points_return = key_points_list.copy()
    for i in range(0, len(key_points_return)):
        key_points_return[i] = int(key_points_return[i] * compressed_ratio)
    key_points_return = list(set(key_points_return))
    key_points_return[-1] = data_len - 1
    
    return data_cp, key_points_return


def batch_processing(file_list, sampling_frequency=50.0, high_pass_cutoff=0.1, high_pass_order=5, low_pass_cutoff=8.0, low_pass_order=5, moving_average_window=5, compress_ratio=0.2, quantification_threshold=0.5):
    
    
    result_csv_list = []
    fig_list = []
    acc_fig_list = []
    error_list = []
    
    for index, file in enumerate(file_list):
        
        try:
        
            bytes_data = file.getvalue()
            data = plistlib.loads(bytes_data)
                
            data_filtered = butter_highpass_filter(data, high_pass_cutoff, sampling_frequency, high_pass_order)
            data = data_filtered
            
            data_filtered = butter_lowpass_filter(data, low_pass_cutoff, sampling_frequency, low_pass_order)
            data = data_filtered
            
            data_filtered = moving_average_filter(data, moving_average_window)
            data = data_filtered
            
            key_points_list = get_key_points(data)
            data_connect = data_connection(data, key_points_list)
            
            data, key_points_list = data_compress(data, compress_ratio, key_points_list)
            
            data = quantification(data, quantification_threshold, key_points_list)
            
            acc = np.diff(data)
            acc = np.insert(acc, 0, 0)
            
            time = np.linspace(0, len(data) / sampling_frequency, len(data)) / compress_ratio
            
            result_csv = pd.DataFrame({'time': time, 'amplitude': data, 'acceleration': acc})
            
            fig = figure(title=f'{file.name}', x_axis_label='Time(s)', y_axis_label='Amplitude', width=800, height=400, y_range=(-40, 40))
            time = np.linspace(0, len(data) / sampling_frequency, len(data)) / compress_ratio
            _interval = len(time) // 2 * 2
            _interval = _interval // 2 * 2
            fig.xaxis.ticker = bokeh.models.tickers.SingleIntervalTicker(interval=2)
            fig.grid.grid_line_color = 'Black'
            fig.grid.grid_line_alpha = 0.2
            fig.line(time, data, line_width=2, line_color='red')
            
            acc_fig = figure(title=f'{file.name}_acc', x_axis_label='Time(s)', y_axis_label='Acceleration', width=800, height=400, y_range=(-40, 40))
            time = np.linspace(0, len(acc) / sampling_frequency, len(acc)) / compress_ratio
            _interval = len(time) // 2 * 2
            _interval = _interval // 2 * 2
            acc_fig.xaxis.ticker = bokeh.models.tickers.SingleIntervalTicker(interval=2)
            acc_fig.grid.grid_line_color = 'Black'
            acc_fig.grid.grid_line_alpha = 0.2
            acc_fig.line(time, acc, line_width=2, line_color='green')
            
            # save the result (csv file name, csv_file) as a tuple to result_csv_list
            result_csv_list.append((f'{file.name}.csv', result_csv))
            fig_list.append((f'{file.name}.svg', fig))
            acc_fig_list.append((f'{file.name}_acc.svg', acc_fig))
            
        except Exception as e:
            
            error_list.append((file.name, e))
            
    parameter_csv = pd.DataFrame({'sampling_frequency': [sampling_frequency], 'high_pass_cutoff': [high_pass_cutoff], 'high_pass_order': [high_pass_order], 'low_pass_cutoff': [low_pass_cutoff], 'low_pass_order': [low_pass_order], 'moving_average_window': [moving_average_window], 'compress_ratio': [compress_ratio], 'quantification_threshold': [quantification_threshold]})
    
    return result_csv_list, fig_list, acc_fig_list, parameter_csv, error_list
    