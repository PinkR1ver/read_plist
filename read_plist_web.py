import streamlit as st
import os
import plistlib
import bokeh.core.properties
from bokeh.plotting import figure
import bokeh.settings
import bokeh
import numpy as np
from scipy import signal

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


# def get_inflection_points(data):
    
#     inflection_points_index = np.zeros(len(data))
#     data = np.array(data)
#     for i in range(3, len(data)-3):
#         flag = np.zeros(6)
#         for index, j in enumerate([-3, -2, -1, 1 ,2, 3]):
#             flag[index] = (data[i+j] > data[i])

#         if flag.sum() == 6:
#             inflection_points_index[i] = 1
#         elif flag.sum() == 0:
#             inflection_points_index[i] = -1
    
#     return inflection_points_index
            

# def inflection_points_connection(data, inflection_points_index):
    
#     inflection_high = 0
#     inflection_low = 0
#     inflection_list = []
    
#     for i in range(0, len(inflection_points_index)):
#         if inflection_points_index[i] == 1 or inflection_points_index[i] == -1:
#             inflection_list.append(i)
#     inflection_list.append(len(data) - 1)

#     data = np.array(data)
#     data_connection = np.zeros(len(data))
#     for i in range(0, len(data) - 1):
#         if inflection_points_index[i] == 1:
#             data_connection[i] = data[i]
#             inflection_high = i
#         elif inflection_points_index[i] == -1:
#             data_connection[i] = data[i]
#             inflection_low = i
#         else:
#             if i == 0:
#                 data_connection[i] = data[i]
#             elif i == len(data):
#                 data_connection[i] = data[i]
#             elif inflection_high > inflection_low:
#                 inflection_tmp = inflection_list[inflection_list.index(inflection_high) + 1]
#                 data_connection[i] = data[inflection_high] + (data[inflection_tmp] - data[inflection_high]) / (inflection_tmp - inflection_high) * (i - inflection_high)
#             elif inflection_high < inflection_low:
#                 inflection_tmp = inflection_list[inflection_list.index(inflection_low) + 1]
#                 data_connection[i] = data[inflection_low] + (data[inflection_tmp] - data[inflection_low]) / (inflection_tmp - inflection_low) * (i - inflection_low)
            
            
#     return data_connection    

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

def hollow_downsampling(data, downsample_window=0):
    
    if downsample_window == 0:
        
        zero_interval = 0
        zero_interval_min = 99
        for i in range(len(data)):
            if data[i] == 0:
                zero_interval += 1
            else:
                zero_interval_min = min(zero_interval, zero_interval_min)
                zero_interval = 0

        downsample_window = zero_interval_min
    
    data_downsampled = []
    for i in range(0, len(data), downsample_window):
        if data[i:i+downsample_window].sum() == 0:
            data_downsampled.append(0)
        else:
            data_downsampled.append(data[i:i+downsample_window].sum())
            
    return data_downsampled


def quantification(data, threshold):
    
    data = np.array(data)
    data = np.round(data / threshold) * threshold
    
    return data


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
    
    data_cp = np.zeros(data_len)
    for i in range(1, len(key_points_cp) - 1):
        left = key_points_cp[i-1][0]
        right = key_points_cp[i][0] + 1
        
        if right - left == 2:
            data_cp[left] = key_points_cp[i-1][1]
            continue
        
        left_start = key_points_cp[i-1][1]
        right_stop = key_points_cp[i][1]
        data_cp[left:right] = np.linspace(left_start, right_stop, right - left)
    
    return data_cp

if __name__ == '__main__':
    
    base_path = os.path.dirname(__file__)

    with st.sidebar:
        
        st.title('Control the filter parameters')
        
        sampling_frequency = st.slider('Sampling frequency', 0.0, 10000.0, 1000.0)
        high_pass_cutoff = st.slider('High-pass filter cutoff frequency', 0.0, 1000.0, 0.8)
        high_pass_order = st.slider('High-pass filter order', 1, 10, 5)
        low_pass_cutoff = st.slider('Low-pass filter cutoff frequency', 0.0, 1000.0, 80.0)
        low_pass_order = st.slider('Low-pass filter order', 1, 10, 5)
        moving_average_window = st.slider('Moving average window size', 1, 100, 5)
        compress_ratio = st.slider('Compress ratio', 0.0, 1.0, 0.2)
        
    

    st.title('Read Plist File')
    st.write('This is a simple example of reading a plist file and applying a high-pass filter, a low-pass filter and a moving average filter to the data.')
    
    upload_file = st.file_uploader('Upload a plist file', type=['plist'])
    
    if upload_file is None:
        
        upload_file = os.path.join(base_path, '12344_left_VOG_lefteye_Horizontal.plist')
        upload_file_name = '12344_left_VOG_lefteye_Horizontal.plist'
        with open(upload_file, 'rb') as plist_file:
            data = plistlib.load(plist_file)
    
    else:
        
        bytes_data = upload_file.getvalue()
        data = plistlib.loads(bytes_data)
        upload_file_name = upload_file.name
    
    st.markdown('### Original Signal')
    
    # plot original signal
    fig = figure(title='Original Signal', x_axis_label='Data Index', y_axis_label='Amplitude', width=800, height=400)
    fig.line(range(len(data)), data, line_width=2)
    st.bokeh_chart(fig)
    
    st.markdown('### High Pass Filter')
    st.write('Firstly, the high-pass filter is applied to the original signal to correct the baseline drift.')
    
    data_filtered = butter_highpass_filter(data, high_pass_cutoff, sampling_frequency, high_pass_order)
    data = data_filtered
    
    fig = figure(title='Signal After highpass filter', x_axis_label='Data Index', y_axis_label='Amplitude', width=800, height=400)
    fig.line(range(len(data)), data, line_width=2)
    st.bokeh_chart(fig)
    
    st.markdown('### Low Pass Filter')
    st.write('Secondly, the low-pass filter is applied to the signal after high-pass filtering to remove high-frequency noise.')
    
    data_filtered = butter_lowpass_filter(data, low_pass_cutoff, sampling_frequency, low_pass_order)
    data = data_filtered
    
    fig = figure(title='Signal After lowpass filter', x_axis_label='Data Index', y_axis_label='Amplitude', width=800, height=400)
    fig.line(range(len(data)), data, line_width=2)
    st.bokeh_chart(fig)
    
    st.markdown('### Moving Average Filter')
    st.write('Finally, the moving average filter is applied to the signal after low-pass filtering to smooth the signal.')
    
    data_filtered = moving_average_filter(data, moving_average_window)
    data = data_filtered
    
    fig = figure(title='Signal After moving average filter', x_axis_label='Data Index', y_axis_label='Amplitude', width=800, height=400)
    fig.line(range(len(data)), data, line_width=2)
    st.bokeh_chart(fig)
    
    # st.markdown('### Get the Derivative of the Signal')
    # st.write('We get the derivative of the signal to get the acceleration')
    
    # diff = get_diff(data)
    
    # fig = figure(title='Derivative of the Signal', x_axis_label='Data Index', y_axis_label='Amplitude', width=800, height=400)
    # fig.line(range(len(diff)), diff, line_width=2)
    # st.bokeh_chart(fig)
    
    st.markdown('### Download Filtered Data')
    st.write('Here\'s the final result, we did the key point detection and use straight line to connect them')
    
    # data = average_downsample(data, data_downsampling_window)
    key_points_list = get_key_points(data)
    data_connect = data_connection(data, key_points_list)
    
    # inflection_points_index = get_inflection_points(data)
    # data = inflection_points_connection(data, inflection_points_index)
    
    # data = hollow_downsampling(data, downsample_window=data_downsampling_window)
    # data = quantification(data, quantification_threshold)
    
    fig = figure(title='After doing connection', x_axis_label='Data Index', y_axis_label='Amplitude', width=800, height=400)
    fig.line(range(len(data_connect)), data_connect, line_width=2)
    st.bokeh_chart(fig)
    
    data = data_connect
    data = data_compress(data, compress_ratio, key_points_list)
    
    fig = figure(title=f'{upload_file_name}', x_axis_label='Data Index', y_axis_label='Amplitude', width=800, height=400, y_range=(-40, 40))
    _interval = len(data) // 5
    _interval = _interval // 5 * 5
    fig.xaxis.ticker = bokeh.models.tickers.SingleIntervalTicker(interval=_interval)
    fig.grid.grid_line_color = 'Black'
    fig.grid.grid_line_alpha = 0.2
    fig.line(range(len(data)), data, line_width=2, line_color='red')
    st.bokeh_chart(fig)
    
        
    
