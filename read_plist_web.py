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

if __name__ == '__main__':
    
    base_path = os.path.dirname(__file__)

    with st.sidebar:
        
        st.title('Control the filter parameters')
        
        sampling_frequency = st.slider('Sampling frequency', 0.0, 10000.0, 1000.0)
        high_pass_cutoff = st.slider('High-pass filter cutoff frequency', 0.0, 100.0, 0.8)
        high_pass_order = st.slider('High-pass filter order', 1, 10, 5)
        low_pass_cutoff = st.slider('Low-pass filter cutoff frequency', 0.0, 100.0, 50.0)
        low_pass_order = st.slider('Low-pass filter order', 1, 10, 5)
        moving_average_window = st.slider('Moving average window size', 1, 100, 5)
    

    st.title('Read Plist File')
    st.write('This is a simple example of reading a plist file and applying a high-pass filter, a low-pass filter and a moving average filter to the data.')
    
    upload_file = st.file_uploader('Upload a plist file', type=['plist'])
    
    if upload_file is None:
        
        upload_file = os.path.join(base_path, '12344_left_VOG_lefteye_Horizontal.plist')
        with open(upload_file, 'rb') as plist_file:
            data = plistlib.load(plist_file)
    
    else:
        
        bytes_data = upload_file.getvalue()
        data = plistlib.loads(bytes_data)
    
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
        
    
