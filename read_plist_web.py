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

# def hollow_downsampling(data, downsample_window=0):
    
#     if downsample_window == 0:
        
#         zero_interval = 0
#         zero_interval_min = 99
#         for i in range(len(data)):
#             if data[i] == 0:
#                 zero_interval += 1
#             else:
#                 zero_interval_min = min(zero_interval, zero_interval_min)
#                 zero_interval = 0

#         downsample_window = zero_interval_min
    
#     data_downsampled = []
#     for i in range(0, len(data), downsample_window):
#         if data[i:i+downsample_window].sum() == 0:
#             data_downsampled.append(0)
#         else:
#             data_downsampled.append(data[i:i+downsample_window].sum())
            
#     return data_downsampled


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
    
    progess_bar = st.progress(0, text='Processing...')
    
    for index, file in enumerate(file_list):
        
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
        
        
        if index != len(file_list) - 1:
            progess_bar.progress((index + 1) / len(file_list), text='Processing...')
        else:
            progess_bar.empty()
            
    parameter_csv = pd.DataFrame({'sampling_frequency': [sampling_frequency], 'high_pass_cutoff': [high_pass_cutoff], 'high_pass_order': [high_pass_order], 'low_pass_cutoff': [low_pass_cutoff], 'low_pass_order': [low_pass_order], 'moving_average_window': [moving_average_window], 'compress_ratio': [compress_ratio], 'quantification_threshold': [quantification_threshold]})
    
    return result_csv_list, fig_list, acc_fig_list, parameter_csv
    
        

if __name__ == '__main__':
    
    base_path = os.path.dirname(__file__)

    with st.sidebar:
        
        mode = st.selectbox('Select the mode', ['Example: Single File Processing Procedures Detail', 'Batch Processing'], index=0)
        
        st.title('Control the filter parameters')
        
        sampling_frequency = st.slider('Sampling frequency', 0.0, 500.0, 50.0)
        high_pass_cutoff = st.slider('High-pass filter cutoff frequency', 0.0, 3.0, 0.1)
        high_pass_order = st.slider('High-pass filter order', 1, 10, 5)
        low_pass_cutoff = st.slider('Low-pass filter cutoff frequency', 0.0, 30.0, 8.0)
        low_pass_order = st.slider('Low-pass filter order', 1, 10, 5)
        moving_average_window = st.slider('Moving average window size', 1, 100, 5)
        compress_ratio = st.slider('Compress ratio', 0.0, 1.0, 0.2, step=0.05)
        quantification_threshold = st.slider('Quantification threshold', 0.0, 4.0, 0.5)
        
    
    if mode == 'Example: Single File Processing Procedures Detail':

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
        fig = figure(title='Original Signal', x_axis_label='Time(s)', y_axis_label='Amplitude', width=800, height=400)
        fig.line(range(len(data)), data, line_width=2)
        st.bokeh_chart(fig)
        
        st.markdown('### High Pass Filter')
        st.write('Firstly, the high-pass filter is applied to the original signal to correct the baseline drift.')
        
        data_filtered = butter_highpass_filter(data, high_pass_cutoff, sampling_frequency, high_pass_order)
        data = data_filtered
        
        fig = figure(title='Signal After highpass filter', x_axis_label='Time(s)', y_axis_label='Amplitude', width=800, height=400)
        time = np.linspace(0, len(data) / sampling_frequency, len(data))
        fig.line(time, data, line_width=2)
        st.bokeh_chart(fig)
        
        st.markdown('### Low Pass Filter')
        st.write('Secondly, the low-pass filter is applied to the signal after high-pass filtering to remove high-frequency noise.')
        
        data_filtered = butter_lowpass_filter(data, low_pass_cutoff, sampling_frequency, low_pass_order)
        data = data_filtered
        
        fig = figure(title='Signal After lowpass filter', x_axis_label='Time(s)', y_axis_label='Amplitude', width=800, height=400)
        time = np.linspace(0, len(data) / sampling_frequency, len(data))
        fig.line(time, data, line_width=2)
        st.bokeh_chart(fig)
        
        st.markdown('### Moving Average Filter')
        st.write('Finally, the moving average filter is applied to the signal after low-pass filtering to smooth the signal.')
        
        data_filtered = moving_average_filter(data, moving_average_window)
        data = data_filtered
        
        fig = figure(title='Signal After moving average filter', x_axis_label='Time(s)', y_axis_label='Amplitude', width=800, height=400)
        time = np.linspace(0, len(data) / sampling_frequency, len(data))
        fig.line(time, data, line_width=2)
        st.bokeh_chart(fig)
        
        # st.markdown('### Get the Derivative of the Signal')
        # st.write('We get the derivative of the signal to get the acceleration')
        
        # diff = get_diff(data)
        
        # fig = figure(title='Derivative of the Signal', x_axis_label='Time(s)', y_axis_label='Amplitude', width=800, height=400)
        # fig.line(range(len(diff)), diff, line_width=2)
        # st.bokeh_chart(fig)
        
        st.markdown('### Key Point Detection and Connection')
        st.write('Here\'s the final result, we did the key point detection and use straight line to connect them')
        
        # data = average_downsample(data, data_downsampling_window)
        key_points_list = get_key_points(data)
        data_connect = data_connection(data, key_points_list)
        
        # inflection_points_index = get_inflection_points(data)
        # data = inflection_points_connection(data, inflection_points_index)
        
        # data = hollow_downsampling(data, downsample_window=data_downsampling_window)
        # data = quantification(data, quantification_threshold)
        
        data = data_connect
        
        st.markdown('### Time Domain Data Compression and Quantification')
        
        fig = figure(title='After doing connection', x_axis_label='Time(s)', y_axis_label='Amplitude', width=800, height=400)
        time = np.linspace(0, len(data) / sampling_frequency, len(data))
        fig.line(time, data, line_width=2)
        st.bokeh_chart(fig)
        
        data, key_points_list = data_compress(data, compress_ratio, key_points_list)
        
        st.write('After data compress in time domain, we get the following signal:')
        
        fig = figure(title='After Data Compress', x_axis_label='Time(s)', y_axis_label='Amplitude', width=800, height=400)
        time = np.linspace(0, len(data) / sampling_frequency, len(data)) / compress_ratio
        fig.line(time, data, line_width=2)
        st.bokeh_chart(fig)
        
        data = quantification(data, quantification_threshold, key_points_list)
        
        st.write('After quantification, we get the following signal:')
        
        fig = figure(title='After Quantification', x_axis_label='Time(s)', y_axis_label='Amplitude', width=800, height=400)
        time = np.linspace(0, len(data) / sampling_frequency, len(data)) / compress_ratio
        fig.line(time, data, line_width=2)
        st.bokeh_chart(fig)
        
        st.markdown('### Save the Result')
        st.write('Here\'s the final result, you can download the graph and save the data to a .csv file.')
        
        fig = figure(title=f'{upload_file_name}', x_axis_label='Time(s)', y_axis_label='Amplitude', width=800, height=400, y_range=(-40, 40))
        time = np.linspace(0, len(data) / sampling_frequency, len(data)) / compress_ratio
        # _interval = len(data) // 2
        # _interval = _interval // 2 * 2
        # fig.xaxis.ticker = bokeh.models.tickers.SingleIntervalTicker(interval=_interval)
        _interval = len(time) // 2 * 2
        _interval = _interval // 2 * 2
        fig.xaxis.ticker = bokeh.models.tickers.SingleIntervalTicker(interval=2)
        fig.grid.grid_line_color = 'Black'
        fig.grid.grid_line_alpha = 0.2
        fig.line(time, data, line_width=2, line_color='red')
        st.bokeh_chart(fig)
        
        acc = np.diff(data)
        acc = np.insert(acc, 0, 0)
        
        result_csv = pd.DataFrame({'time': time, 'amplitude': data, 'acceleration': acc})
        st.download_button('Download the result as a CSV file', result_csv.to_csv(), 'result.csv', 'text/csv')
    
    if mode == 'Batch Processing':
    
        st.markdown('### Batch Processing')
        st.write('Above is example show, you can upload multiple files to process in batch.')
        
        upload_files = st.file_uploader('Upload multiple plist files', type=['plist'], accept_multiple_files=True)
        
        if upload_files:

            result_csv_list, fig_list, acc_fig_list, parameter_csv =  batch_processing(upload_files, sampling_frequency, high_pass_cutoff, high_pass_order, low_pass_cutoff, low_pass_order, moving_average_window, compress_ratio, quantification_threshold)
            zipObj = ZipFile('result.zip', 'w')
            for csv_name, csv_file in result_csv_list:
                csv_file = csv_file.to_csv()
                zipObj.writestr(csv_name, csv_file)
            zipObj.writestr('processing_parameter.csv', parameter_csv.to_csv())
            for fig_name, fig in fig_list:
                fig.output_backend = 'svg'
                fig_data = get_svgs(fig)
                fig_data = fig_data[0]
                fig_data = fig_data.encode('utf-8')
                zipObj.writestr(fig_name, fig_data)
            for acc_fig_name, acc_fig in acc_fig_list:
                acc_fig.output_backend = 'svg'
                acc_fig_data = get_svgs(acc_fig)
                acc_fig_data = acc_fig_data[0]
                acc_fig_data = acc_fig_data.encode('utf-8')
                zipObj.writestr(acc_fig_name, acc_fig_data)
            zipObj.close()
            st.download_button('Download the result as a ZIP file', open('result.zip', 'rb').read(), 'result.zip', 'application/zip')
            
    
    
        
    
