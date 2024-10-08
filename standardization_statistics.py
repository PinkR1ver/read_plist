import json
import os
import plistlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class ExperimentStatistics:
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.stages = [0, 5, 10, 15, 25, 0, -5, -10, -15, -25]
        self.collected_data = {
            '1': {'lefteye': {stage: [] for stage in self.stages}},
            '2': {'head': {stage: [] for stage in self.stages}}
        }
        self.skipped_experiments = []

    def process_experiments(self):
        for exp_type in ['1', '2']:
            part = 'lefteye' if exp_type == '1' else 'head'
            for patient in self.data:
                if exp_type in self.data[patient]:
                    for exp_num in self.data[patient][exp_type]:
                        if part in self.data[patient][exp_type][exp_num]:
                            self.process_single_experiment(exp_type, part, patient, exp_num)

    def process_single_experiment(self, exp_type, part, patient, exp_num):
        file_path = self.data[patient][exp_type][exp_num][part]['data_path']
        try:
            with open(file_path, 'rb') as f:
                signal = plistlib.load(f)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_title(f"Select stage boundaries - Patient {patient}, Experiment Type {exp_type}, Number {exp_num} ({part})")
            ax.plot(signal)

            stage_boundaries = []

            skip_button_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
            skip_button = Button(skip_button_ax, 'Skip')

            def on_click(event):
                if event.inaxes:
                    x = int(event.xdata)
                    stage_boundaries.append(x)
                    color = 'r' if len(stage_boundaries) % 2 == 1 else 'g'
                    ax.axvline(x=x, color=color, linestyle='--')
                    fig.canvas.draw()

                    if len(stage_boundaries) >= 20:  # 10 stages, 2 boundaries each
                        plt.close(fig)
                        try:
                            self.collect_stage_data(signal, stage_boundaries, exp_type, part, patient, exp_num)
                        except Exception as e:
                            print(f"Error collecting stage data: {e}")
                            self.skipped_experiments.append((exp_type, part, patient, exp_num))

            def skip_experiment(event):
                self.skipped_experiments.append((exp_type, part, patient, exp_num))
                print(f"Skipped experiment: Type {exp_type}, Part {part}, Patient {patient}, Number {exp_num}")
                plt.close(fig)

            fig.canvas.mpl_connect('button_press_event', on_click)
            skip_button.on_clicked(skip_experiment)

            plt.show()
        except Exception as e:
            print(f"Error processing experiment: Type {exp_type}, Part {part}, Patient {patient}, Number {exp_num}")
            print(f"Error details: {e}")
            self.skipped_experiments.append((exp_type, part, patient, exp_num))

    def collect_stage_data(self, signal, stage_boundaries, exp_type, part, patient, exp_num):
        for i in range(0, len(stage_boundaries), 2):
            start = stage_boundaries[i]
            end = stage_boundaries[i+1] if i+1 < len(stage_boundaries) else len(signal)
            stage_data = signal[start:end]
            
            # Convert to list if it's a numpy array, otherwise leave as is
            if isinstance(stage_data, np.ndarray):
                stage_data = stage_data.tolist()
            
            stage_value = self.stages[i//2]
            self.collected_data[exp_type][part][stage_value].append({
                'patient': patient,
                'exp_num': exp_num,
                'data': stage_data
            })

    def save_collected_data(self):
        output = {
            'collected_data': self.collected_data,
            'skipped_experiments': self.skipped_experiments
        }

        with open('collected_experiment_data.json', 'w') as f:
            json.dump(output, f, indent=4)

        print("Collected data has been saved to collected_experiment_data.json")
        print(f"Total skipped experiments: {len(self.skipped_experiments)}")

if __name__ == "__main__":
    base_path = os.path.dirname(__file__)
    json_file = os.path.join(base_path, "experiment_data.json")
    
    stats = ExperimentStatistics(json_file)
    stats.process_experiments()
    stats.save_collected_data()