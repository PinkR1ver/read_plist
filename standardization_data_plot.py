import json
import os
import plistlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import matplotlib.colors as mcolors

class ExperimentPlotter:
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.anchors = {}
        self.window_size = 1500  # 时间窗大小，可以根据需要调整
        self.colors = self.generate_colors()

    def generate_colors(self):
        # 生成足够多的不同颜色
        base_colors = list(mcolors.TABLEAU_COLORS.values())
        num_patients = len(self.data)
        num_experiments = max(len(self.data[patient][exp_type]) 
                              for patient in self.data 
                              for exp_type in self.data[patient])
        total_colors_needed = num_patients * num_experiments
        
        colors = []
        for _ in range((total_colors_needed // len(base_colors)) + 1):
            colors.extend(base_colors)
        
        return colors[:total_colors_needed]

    def plot_experiments(self):
        for exp_type in ['1', '2']:
            anchor_part = 'lefteye' if exp_type == '1' else 'head'
            self.select_anchors(exp_type, anchor_part)
            
            for part in ['head', 'lefteye', 'righteye']:
                self.plot_experiment_type(exp_type, part)

    def select_anchors(self, exp_type, part):
        for patient in self.data:
            if exp_type in self.data[patient]:
                for exp_num in self.data[patient][exp_type]:
                    if part in self.data[patient][exp_type][exp_num]:
                        self.select_anchor(exp_type, part, patient, exp_num)

    def select_anchor(self, exp_type, part, patient, exp_num):
        file_path = self.data[patient][exp_type][exp_num][part]['data_path']
        with open(file_path, 'rb') as f:
            signal = plistlib.load(f)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(f"Select Anchor for Patient {patient}, Experiment Type {exp_type}, Number {exp_num} ({part})")
        ax.plot(signal)

        cursor = Cursor(ax, useblit=True, color='red', linestyle='--')
        
        def on_click(event):
            if event.inaxes:
                self.anchors[(patient, exp_type, exp_num)] = int(event.xdata)
                plt.close(fig)

        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

    def plot_experiment_type(self, exp_type, part):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(f"Experiment Type {exp_type} - {part}")

        color_index = 0
        for patient in self.data:
            if exp_type in self.data[patient]:
                for exp_num in self.data[patient][exp_type]:
                    if part in self.data[patient][exp_type][exp_num]:
                        file_path = self.data[patient][exp_type][exp_num][part]['data_path']
                        with open(file_path, 'rb') as f:
                            signal = plistlib.load(f)
                        
                        anchor = self.anchors.get((patient, exp_type, exp_num), 0)
                        start = max(0, anchor - self.window_size // 2)
                        end = min(len(signal), anchor + self.window_size // 2)
                        aligned_signal = signal[start:end]
                        
                        # 如果信号长度不足window_size，用0填充
                        if len(aligned_signal) < self.window_size:
                            aligned_signal = np.pad(aligned_signal, 
                                                    (0, self.window_size - len(aligned_signal)), 
                                                    'constant')
                        
                        ax.plot(aligned_signal, color=self.colors[color_index], alpha=0.7, 
                                label=f"{patient} - Exp {exp_num}")
                        color_index += 1

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlabel("Time")
        ax.set_ylabel("Signal Value")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    base_path = os.path.dirname(__file__)
    json_file = os.path.join(base_path, "experiment_data.json")
    
    plotter = ExperimentPlotter(json_file)
    plotter.plot_experiments()