import os
import plistlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

class AngleSelector:
    def __init__(self, signal):
        self.signal = signal
        self.angles = [-25, -10, -5, 5, 10, 25]
        self.angle_segments = {angle: [] for angle in self.angles}
        self.current_angle_index = 0

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line, = self.ax.plot(signal)
        self.ax.axhline(y=0, color='r', linestyle='--')  # Add baseline at y=0
        
        self.span = SpanSelector(
            self.ax, self.onselect, 'horizontal', useblit=True,
            props=dict(alpha=0.5, facecolor='red')
        )

        self.ax.set_title(f'Select data range for {self.angles[self.current_angle_index]}°')
        plt.show()

    def onselect(self, xmin, xmax):
        indmin, indmax = np.searchsorted(self.line.get_xdata(), (xmin, xmax))
        indmax = min(len(self.signal) - 1, indmax)

        current_angle = self.angles[self.current_angle_index]
        self.angle_segments[current_angle] = self.signal[indmin:indmax]

        self.current_angle_index += 1
        if self.current_angle_index < len(self.angles):
            self.ax.set_title(f'Select data range for {self.angles[self.current_angle_index]}°')
            self.fig.canvas.draw()
        else:
            plt.close()
            self.plot_results()

    def plot_results(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        for angle, segment in self.angle_segments.items():
            ax.plot(segment, label=f'{angle}°')
        
        ax.axhline(y=0, color='r', linestyle='--', label='Baseline')  # Add baseline at y=0
        
        ax.legend()
        ax.set_title('Eye Movement Signal Data for Each Angle')
        ax.set_xlabel('Data Points')
        ax.set_ylabel('Signal Value')
        plt.show()

        for angle, segment in self.angle_segments.items():
            print(f'Mean value for {angle}°: {np.mean(segment)}')

if __name__ == '__main__':
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, 'data', 'standardization', '2024070503_right_VOG_lefteye_Horizontal.plist')
    
    signal = plistlib.load(open(data_path, 'rb'))
    
    selector = AngleSelector(signal)