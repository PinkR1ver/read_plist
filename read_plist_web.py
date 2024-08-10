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


def quantification(data, threshold):
    
    data_cp = data.copy()
    
    for i in range(0, len(data)):
        data_cp[i] = np.round(data[i] / threshold) * threshold
        
    return data_cp


def get_diff(data):
    
    diff = np.diff(data)
    diff = np.insert(diff, 0, 0)
    
    return diff

# def data_compress(data, compressed_ratio, key_points_list):
    
#     key_points_cp = []
#     for key_points in key_points_list:
#         key_points_cp.append((key_points, data[key_points]))
    
#     for i in range(0, len(key_points_cp)):
#         key_points_cp[i] = (int(key_points_cp[i][0] * compressed_ratio), key_points_cp[i][1])
    
#     data_len = key_points_cp[-1][0]
    
#     data_cp = np.zeros(data_len + 1)
#     for i in range(1, len(key_points_cp) - 1):
#         left = key_points_cp[i-1][0]
#         right = key_points_cp[i][0] + 1
        
#         if right - left == 2:
#             data_cp[left] = key_points_cp[i-1][1]
#             continue
        
#         left_start = key_points_cp[i-1][1]
#         right_stop = key_points_cp[i][1]
#         data_cp[left:right] = np.linspace(left_start, right_stop, right - left)
    
#     key_points_return = key_points_list.copy()
#     for i in range(0, len(key_points_return)):
#         key_points_return[i] = int(key_points_return[i] * compressed_ratio)
#     key_points_return = list(set(key_points_return))
#     key_points_return[-1] = data_len - 1
    
#     return data_cp, key_points_return


def batch_processing(file_list, sampling_frequency=50.0, high_pass_cutoff=0.1, high_pass_order=5, low_pass_cutoff=8.0, low_pass_order=5, moving_average_window=5, quantification_threshold=0.1, y_axis_range_positive=15, y_axis_range_negative=-15):
    
    
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
            
            # key_points_list = get_key_points(data)
            # data_connect = data_connection(data, key_points_list)
            
            # data, key_points_list = data_compress(data, compress_ratio, key_points_list)
            
            data = quantification(data, quantification_threshold)
            
            acc = np.diff(data)
            acc = np.insert(acc, 0, 0)
            
            time = np.linspace(0, len(data) / sampling_frequency, len(data))
            
            result_csv = pd.DataFrame({'time': time, 'amplitude': data, 'acceleration': acc})
            
            fig = figure(title=f'{file.name}', x_axis_label='Time(s)', y_axis_label='Amplitude', width=800, height=400, y_range=(y_axis_range_negative, y_axis_range_positive))
            time = np.linspace(0, len(data) / sampling_frequency, len(data))
            _interval = len(time) // 2 * 2
            _interval = _interval // 2 * 2
            fig.xaxis.ticker = bokeh.models.tickers.SingleIntervalTicker(interval=2)
            fig.grid.grid_line_color = 'Black'
            fig.grid.grid_line_alpha = 0.2
            fig.line(time, data, line_width=2, line_color='red')
            
            acc_fig = figure(title=f'{file.name}_acc', x_axis_label='Time(s)', y_axis_label='Acceleration', width=800, height=400, y_range=(y_axis_range_positive, y_axis_range_negative))
            time = np.linspace(0, len(acc) / sampling_frequency, len(acc))
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
    
    couple_list = []
    esw_fig_list = []
    
    for index, file in enumerate(file_list):
        
        name = file.name
        label = name.split('_')[0]
        
        if 'pHIT' in name:
            if 'lefteye' in name:
                couple_list.append((label, 'lefteye', index))
            elif 'head' in name:
                couple_list.append((label, 'head', index))
                
    for i in range(len(couple_list) - 1):
        
        for j in range(i + 1, len(couple_list)):
            
            if couple_list[i][0] == couple_list[j][0]:
                
                if couple_list[i][1] == 'head':
                    head_file = file_list[couple_list[i][2]]
                    eye_file = file_list[couple_list[j][2]]
                else:
                    head_file = file_list[couple_list[j][2]]
                    eye_file = file_list[couple_list[i][2]]
                
                label = couple_list[i][0]
                
                try:
                    
                    esw, t, head_data, eye_data = enhanced_saccadic_wave(head_file, eye_file, sampling_frequency, high_pass_cutoff, high_pass_order, low_pass_cutoff, low_pass_order, moving_average_window, quantification_threshold)
                    esw_fig = figure(title=f'{label} Enhanced Saccadic Wave', x_axis_label='Time(s)', y_axis_label='Enhanced Saccadic Wave', width=800, height=400)
                    esw_fig.line(t, head_data, line_width=2, color='orange', legend_label='Head Movement')
                    esw_fig.line(t, eye_data, line_width=2, color='green', legend_label='Eye Movement')
                    esw_fig.line(t, esw, line_width=2, color='red', legend_label='Enhanced Saccadic Wave')
                    esw_fig_list.append((f'{label}_esw.svg', esw_fig))
                    
                except Exception as e:
                    
                    error_list.append((label, e))
            
    parameter_csv = pd.DataFrame({'sampling_frequency': [sampling_frequency], 'high_pass_cutoff': [high_pass_cutoff], 'high_pass_order': [high_pass_order], 'low_pass_cutoff': [low_pass_cutoff], 'low_pass_order': [low_pass_order], 'moving_average_window': [moving_average_window], 'quantification_threshold': [quantification_threshold]})
    
    return result_csv_list, fig_list, acc_fig_list, esw_fig_list, parameter_csv, error_list


def enhanced_saccadic_wave(head_file, eye_file, sampling_frequency=50.0, high_pass_cutoff=0.1, high_pass_order=5, low_pass_cutoff=8.0, low_pass_order=5, moving_average_window=5, quantification_threshold=0.1):
    
    try: 
        
        bytes_data = head_file.getvalue()
        head_data = plistlib.loads(bytes_data)
        
        bytes_data = eye_file.getvalue()
        eye_data = plistlib.loads(bytes_data)
        
        data = head_data
        
        data_filtered = butter_highpass_filter(data, high_pass_cutoff, sampling_frequency, high_pass_order)
        data = data_filtered
        
        data_filtered = butter_lowpass_filter(data, low_pass_cutoff, sampling_frequency, low_pass_order)
        data = data_filtered
        
        data_filtered = moving_average_filter(data, moving_average_window)
        data = data_filtered
        
        # key_points_list = get_key_points(data)
        # data_connect = data_connection(data, key_points_list)
        
        # data = data_connect
        
        # data, key_points_list = data_compress(data, compress_ratio, key_points_list)
        
        data = quantification(data, quantification_threshold)
        
        head_speed = np.diff(data)
        head_data = data
        
        data = eye_data
        
        data_filtered = butter_highpass_filter(data, high_pass_cutoff, sampling_frequency, high_pass_order)
        data = data_filtered
        
        data_filtered = butter_lowpass_filter(data, low_pass_cutoff, sampling_frequency, low_pass_order)
        data = data_filtered
        
        data_filtered = moving_average_filter(data, moving_average_window)
        data = data_filtered
        
        # key_points_list = get_key_points(data)
        # data_connect = data_connection(data, key_points_list)
        
        # data = data_connect
        
        # data, key_points_list = data_compress(data, compress_ratio, key_points_list)
        
        data = quantification(data, quantification_threshold)
        
        eye_speed = np.diff(data)
        eye_data = data
        
        enhanced_saccadic_wave = []
        for i in range(0, len(head_speed)):
            if head_speed[i] != 0:
                enhanced_saccadic_wave.append(eye_speed[i] / head_speed[i])
            else:
                enhanced_saccadic_wave.append(0)
                
        enhanced_saccadic_wave.insert(0, 0)
        enhanced_saccadic_wave = np.array(enhanced_saccadic_wave)
        
        t = np.linspace(0, len(enhanced_saccadic_wave) / sampling_frequency, len(enhanced_saccadic_wave))
        
        return enhanced_saccadic_wave, t, head_data, eye_data
    
    except Exception as e:
        
        return None, e, None, None
    
    

def enhanced_saccadic_wave_v2(head_file, eye_file, sampling_frequency=50.0, high_pass_cutoff=0.1, high_pass_order=5, low_pass_cutoff=8.0, low_pass_order=5, moving_average_window=5, quantification_threshold=0.1):
    
        try: 
        
            bytes_data = head_file.getvalue()
            head_data = plistlib.loads(bytes_data)
            
            bytes_data = eye_file.getvalue()
            eye_data = plistlib.loads(bytes_data)
            
            data = head_data
            
            data_filtered = butter_highpass_filter(data, high_pass_cutoff, sampling_frequency, high_pass_order)
            data = data_filtered
            
            data_filtered = butter_lowpass_filter(data, low_pass_cutoff, sampling_frequency, low_pass_order)
            data = data_filtered
            
            data_filtered = moving_average_filter(data, moving_average_window)
            data = data_filtered
            
            # key_points_list = get_key_points(data)
            # data_connect = data_connection(data, key_points_list)
            
            # data = data_connect
            
            # data, key_points_list = data_compress(data, compress_ratio, key_points_list)
            
            data = quantification(data, quantification_threshold)
            
            head_speed = np.diff(data)
            head_data = data
            
            data = eye_data
            
            data_filtered = butter_highpass_filter(data, high_pass_cutoff, sampling_frequency, high_pass_order)
            data = data_filtered
            
            data_filtered = butter_lowpass_filter(data, low_pass_cutoff, sampling_frequency, low_pass_order)
            data = data_filtered
            
            data_filtered = moving_average_filter(data, moving_average_window)
            data = data_filtered
            
            # key_points_list = get_key_points(data)
            # data_connect = data_connection(data, key_points_list)
            
            # data = data_connect
            
            # data, key_points_list = data_compress(data, compress_ratio, key_points_list)
            
            data = quantification(data, quantification_threshold)
            
            eye_speed = np.diff(data)
            eye_data = data
            
            
            
        except Exception as e:
            
            return None, e, None, None


def speed_filter(data, threshold=0.4):
    
    data_cp = data.copy()
    speed = np.diff(data)
    speed = np.insert(speed, 0, 0)
    
    key_idx = []
    key_idx_right = []
    key_idx_left = []
    
    for i in range(1, len(data_cp) - 1):
        if data_cp[i] > data_cp[i-1] and data_cp[i] > data_cp[i+1]:
            if speed[i] > threshold:
                key_idx_right.append(i)
        elif data_cp[i] < data_cp[i-1] and data_cp[i] < data_cp[i+1]:
            if speed[i] < -threshold:
                key_idx_left.append(i)
            
    key_idx.append(key_idx_right)
    key_idx.append(key_idx_left)
    
    return key_idx

def esw_filter(data, postive_or_negative_flag, threshold=0.4):
    
    for i in range(0, len(data)):
        pass

def test_function(head_file, eye_file, sampling_frequency=50.0, high_pass_cutoff=0.1, high_pass_order=5, low_pass_cutoff=8.0, low_pass_order=5, moving_average_window=5, quantification_threshold=0.1):

        mov_data = [0, 0]
        for index, file in enumerate([head_file, eye_file]):
            bytes_data = file.getvalue()
            data = plistlib.loads(bytes_data)
            
            data_filtered = butter_highpass_filter(data, high_pass_cutoff, sampling_frequency, high_pass_order)
            data = data_filtered
            
            data_filtered = butter_lowpass_filter(data, low_pass_cutoff, sampling_frequency, low_pass_order)
            data = data_filtered
        
            data_filtered = moving_average_filter(data, moving_average_window)
            mov_data[index] = data_filtered
            
        head_data = mov_data[0]
        eye_data = mov_data[1]
        
        head_speed = np.diff(head_data)
        head_speed = np.insert(head_speed, 0, 0)
        
        eye_speed = np.diff(eye_data)
        eye_speed = np.insert(eye_speed, 0, 0)
        
        key_idx = speed_filter(head_data)
        key_speed_right = head_speed[key_idx[0]]
        key_speed_left = head_speed[key_idx[1]]
        
        speed_df_right = pd.DataFrame({'speed_right':key_speed_right})
        speed_df_left = pd.DataFrame({'speed_left':key_speed_left})
        
        st.dataframe(speed_df_right)
        st.dataframe(speed_df_left)
        
        window_size = 25
        
        fig = figure(title='Movement Signal', x_axis_label='Time(s)', y_axis_label='Amplitidue', width=800, height=400)
        time = np.linspace(0, len(head_data) / sampling_frequency, len(head_data))
        fig.line(time, head_data, line_width=2, color='orange', legend_label='Head Movement')
        fig.line(time, eye_data, line_width=2, color='green', legend_label='Eye Movement')
        st.bokeh_chart(fig)
        
        fig = figure(title='Enhanced Saccadic Wave', x_axis_label='Time(s)', y_axis_label='Speed', width=800, height=400)
        time = np.linspace(0, window_size * 2 / sampling_frequency, window_size * 2)
        for i in range(0, len(key_idx[0])):
            fig.line(time, head_speed[key_idx[0][i] - window_size:key_idx[0][i] + window_size], line_width=2, color='orange', legend_label='Head Speed')
        for i in range(0, len(key_idx[1])):
            fig.line(time, -head_speed[key_idx[1][i] - window_size:key_idx[1][i] + window_size], line_width=2, color='orange', legend_label='Head Speed')
        for i in range(0, len(key_idx[0])):
            fig.line(time, -eye_speed[key_idx[0][i] - window_size:key_idx[0][i] + window_size], line_width=2, color='green', legend_label='Eye Speed')
        for i in range(0, len(key_idx[1])):
            fig.line(time, eye_speed[key_idx[1][i] - window_size:key_idx[1][i] + window_size], line_width=2, color='green', legend_label='Eye Speed')
        st.bokeh_chart(fig)
        
        esw_speed = []
        for idx in key_idx:
            for i in range(0, len(idx)):
                esw_speed_element = np.zeros(window_size * 2)
                for j in range(0, window_size * 2):
                    if abs(abs(head_speed[idx[i] - window_size + j]) - abs(eye_speed[idx[i] - window_size + j])) > 0.1 and (eye_speed[idx[i] - window_size + j] * head_speed[idx[i] - window_size + j]) < 0:
                        if head_speed[idx[i] - window_size + j] != 0:
                            esw_speed_element[j] =  abs(eye_speed[idx[i] - window_size + j] / head_speed[idx[i] - window_size + j])
                        # else:
                        #     esw_speed_element[j] = -eye_speed[idx[i] - window_size + j] / 0.01
                    else:
                        esw_speed_element[j] = 0
                esw_speed.append(esw_speed_element)
                
        
        fig = figure(title='Enhanced Saccadic Wave', x_axis_label='Time(s)', y_axis_label='Speed', width=800, height=400, y_range=(-3, 3))
        time = np.linspace(0, window_size * 2 / sampling_frequency, window_size * 2)
        for i in range(0, len(key_idx[0])):
            fig.line(time, head_speed[key_idx[0][i] - window_size:key_idx[0][i] + window_size], line_width=2, color='orange', legend_label='Head Speed')
        for i in range(0, len(key_idx[1])):
            fig.line(time, -head_speed[key_idx[1][i] - window_size:key_idx[1][i] + window_size], line_width=2, color='orange', legend_label='Head Speed')
        for i in range(0, len(key_idx[0])):
            fig.line(time, -eye_speed[key_idx[0][i] - window_size:key_idx[0][i] + window_size], line_width=2, color='green', legend_label='Eye Speed')
        for i in range(0, len(key_idx[1])):
            fig.line(time, eye_speed[key_idx[1][i] - window_size:key_idx[1][i] + window_size], line_width=2, color='green', legend_label='Eye Speed')
        for i in range(0, len(esw_speed)):
            fig.line(time, esw_speed[i], line_width=2, color='red', legend_label='Enhanced Saccadic Wave')
        st.bokeh_chart(fig)
        
        for i in range(0, len(key_idx[0])):
            
            fig = figure(title=f'Right head signal with speed {head_speed[key_idx[0][i]]}', x_axis_label='Time(s)', y_axis_label='Amplitidue', width=800, height=400)
            time = np.linspace(0, window_size * 2 / sampling_frequency, window_size * 2)
            head_signal = head_data[key_idx[0][i] - window_size:key_idx[0][i] + window_size]
            eye_signal = eye_data[key_idx[0][i] - window_size:key_idx[0][i] + window_size]
            fig.line(time, head_signal, line_width=2, color='orange')
            fig.line(time, eye_signal, line_width=2, color='green')
            st.bokeh_chart(fig)
            
            fig = figure(title=f'Right head move speed with speed {head_speed[key_idx[0][i]]}', x_axis_label='Time(s)', y_axis_label='Speed', width=800, height=400)
            time = np.linspace(0, window_size * 2 / sampling_frequency, window_size * 2)
            fig.line(time, head_speed[key_idx[0][i] - window_size:key_idx[0][i] + window_size], line_width=2, color='orange')
            fig.line(time, -eye_speed[key_idx[0][i] - window_size:key_idx[0][i] + window_size], line_width=2, color='green')
            fig.line(time, esw_speed[i], line_width=2, color='red')
            st.bokeh_chart(fig)
            
        for i in range(0, len(key_idx[1])):
            
            fig = figure(title=f'Left head signal with speed {head_speed[key_idx[1][i]]}', x_axis_label='Time(s)', y_axis_label='Amplitidue', width=800, height=400)
            time = np.linspace(0, window_size * 2 / sampling_frequency, window_size * 2)
            head_signal = head_data[key_idx[1][i] - window_size:key_idx[1][i] + window_size]
            eye_signal = eye_data[key_idx[1][i] - window_size:key_idx[1][i] + window_size]
            fig.line(time, head_signal, line_width=2, color='orange')
            fig.line(time, eye_signal, line_width=2, color='green')
            st.bokeh_chart(fig)
            
            fig = figure(title=f'Left head move speed with speed {head_speed[key_idx[1][i]]}', x_axis_label='Time(s)', y_axis_label='Speed', width=800, height=400)
            time = np.linspace(0, window_size * 2 / sampling_frequency, window_size * 2)
            fig.line(time, -head_speed[key_idx[1][i] - window_size:key_idx[1][i] + window_size], line_width=2, color='orange')
            fig.line(time, eye_speed[key_idx[1][i] - window_size:key_idx[1][i] + window_size], line_width=2, color='green')
            fig.line(time, esw_speed[len(key_idx[0]) + i], line_width=2, color='red')
            st.bokeh_chart(fig)
            
    

if __name__ == '__main__':
    
    base_path = os.path.dirname(__file__)

    with st.sidebar:
        
        mode = st.selectbox('Select the mode', ['Example: Single File Processing Procedures Detail', 'Batch Processing', 'Enhanced Saccadic Wave', 'test mode'], index=0)
        
        st.title('Control the filter parameters')
        
        sampling_frequency = st.slider('Sampling frequency', 0.0, 500.0, 50.0)
        high_pass_cutoff = st.slider('High-pass filter cutoff frequency', 0.0, 3.0, 0.1)
        high_pass_order = st.slider('High-pass filter order', 1, 10, 5)
        low_pass_cutoff = st.slider('Low-pass filter cutoff frequency', 0.0, 30.0, 8.0)
        low_pass_order = st.slider('Low-pass filter order', 1, 10, 5)
        moving_average_window = st.slider('Moving average window size', 1, 100, 5)
        quantification_threshold = st.slider('Quantification threshold', 0.01, 4.0, 0.1)
        y_axis_range_positive = st.slider('Y positve axis range', -40, 40, 10)
        y_axis_range_negative = st.slider('Y negative axis range', -40, 40, -10)
        
        # compress_ratio = st.slider('Compress ratio', 0.0, 1.0, 0.2, step=0.05)
        
    
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
        fig = figure(title='Original Signal', x_axis_label='Index', y_axis_label='Amplitude', width=800, height=400)
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
        
        # st.markdown('### Key Point Detection and Connection')
        # st.write('Here\'s the final result, we did the key point detection and use straight line to connect them')
        
        # key_points_list = get_key_points(data)
        # data_connect = data_connection(data, key_points_list)
        
        # data = data_connect
        
        # st.markdown('### Time Domain Data Quantification')
        
        # fig = figure(title='After doing connection', x_axis_label='Time(s)', y_axis_label='Amplitude', width=800, height=400)
        # time = np.linspace(0, len(data) / sampling_frequency, len(data))
        # fig.line(time, data, line_width=2)
        # st.bokeh_chart(fig)
        
        # data, key_points_list = data_compress(data, compress_ratio, key_points_list)
        
        # st.write('After data compress in time domain, we get the following signal:')
        
        # fig = figure(title='After Data Compress', x_axis_label='Time(s)', y_axis_label='Amplitude', width=800, height=400)
        # time = np.linspace(0, len(data) / sampling_frequency, len(data)) / compress_ratio
        # fig.line(time, data, line_width=2)
        # st.bokeh_chart(fig)
        
        data = quantification(data, quantification_threshold)
        
        st.write('After quantification, we get the following signal:')
        
        fig = figure(title='After Quantification', x_axis_label='Time(s)', y_axis_label='Amplitude', width=800, height=400)
        time = np.linspace(0, len(data) / sampling_frequency, len(data))
        fig.line(time, data, line_width=2)
        st.bokeh_chart(fig)
        
        st.markdown('### Save the Result')
        st.write('Here\'s the final result, you can download the graph and save the data to a .csv file.')
        
        fig = figure(title=f'{upload_file_name}', x_axis_label='Time(s)', y_axis_label='Amplitude', width=800, height=400, y_range=(y_axis_range_negative, y_axis_range_positive))
        time = np.linspace(0, len(data) / sampling_frequency, len(data))

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
            
            if 'upload_files' not in st.session_state or st.session_state.upload_files != upload_files:
                st.session_state.upload_files = upload_files
                
                progress_bar = st.progress(0, text='Processing...')
                result_csv_list, fig_list, acc_fig_list, esw_fig_list, parameter_csv, error_list =  batch_processing(upload_files, sampling_frequency, high_pass_cutoff, high_pass_order, low_pass_cutoff, low_pass_order, moving_average_window, quantification_threshold, y_axis_range_positive, y_axis_range_negative)
                progress_bar.progress(25, 'Saving...')
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
                    
                for esw_fig_name, esw_fig in esw_fig_list:
                    esw_fig.output_backend = 'svg'
                    esw_fig_data = get_svgs(esw_fig)
                    esw_fig_data = esw_fig_data[0]
                    esw_fig_data = esw_fig_data.encode('utf-8')
                    zipObj.writestr(esw_fig_name, esw_fig_data)
                
                error_list = pd.DataFrame(error_list, columns=['file_name', 'error'])
                zipObj.writestr('error_list.csv', error_list.to_csv())
                
                zipObj.close()
                progress_bar.progress(100, 'Done!')
            st.download_button('Download the result as a ZIP file', open('result.zip', 'rb').read(), 'result.zip', 'application/zip')
            
    if mode == 'Enhanced Saccadic Wave':
        
        st.title('Enhanced Saccadic Wave')
        st.write('This is a simple example of calculating the enhanced saccadic wave from the head and eye movement data.')
        
        st.markdown('### Upload the head movement data')
        head_file = st.file_uploader('Upload the head movement data', type=['plist'])
        
        st.markdown('### Upload the eye movement data')
        eye_file = st.file_uploader('Upload the eye movement data', type=['plist'])
        
        if head_file and eye_file:
            
            progress_bar = st.progress(0, text='Processing...')
            
            esw, t, head_data, eye_data = enhanced_saccadic_wave(head_file, eye_file, sampling_frequency, high_pass_cutoff, high_pass_order, low_pass_cutoff, low_pass_order, moving_average_window, quantification_threshold)
            
            progress_bar.progress(100, 'Done!')
            
            if esw is not None:
                
                fig = figure(title='Enhanced Saccadic Wave', x_axis_label='Time(s)', y_axis_label='Enhanced Saccadic Wave', width=800, height=400)
                fig.line(t, head_data, line_width=2, color='orange', legend_label='Head Movement')
                fig.line(t, eye_data, line_width=2, color='green', legend_label='Eye Movement')
                # fig.line(t, esw, line_width=2, color='red', legend_label='Enhanced Saccadic Wave')
                st.bokeh_chart(fig)
                
            else:
                
                st.write('Error occurred during the processing.')
                
    if mode == 'test mode':
        
        head_file = st.file_uploader('Upload a head plist file', type=['plist'])
        eye_file = st.file_uploader('Upload a eye plist file', type=['plist'])
        
        if head_file and eye_file:
            test_function(head_file, eye_file, sampling_frequency, high_pass_cutoff, high_pass_order, low_pass_cutoff, low_pass_order, moving_average_window, quantification_threshold)