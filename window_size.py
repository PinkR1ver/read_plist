import plistlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np

def load_plist(file_path):
    with open(file_path, 'rb') as f:
        return plistlib.load(f)

class StageSelectorTool:
    def __init__(self, data):
        self.data = data
        self.stages = []
        self.current_stage = []
        
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line, = self.ax.plot(data)
        
        self.rs = RectangleSelector(self.ax, self.line_select_callback,
                                    useblit=True, button=[1],
                                    minspanx=5, minspany=5, spancoords='pixels',
                                    interactive=True)
        
        self.ax.set_title("Select the range for each stage (press 'q' to quit)")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        plt.show()

    def line_select_callback(self, eclick, erelease):
        x1, _ = eclick.xdata, eclick.ydata
        x2, _ = erelease.xdata, erelease.ydata
        self.current_stage = [int(min(x1, x2)), int(max(x1, x2))]

    def on_key(self, event):
        if event.key == 'enter':
            if self.current_stage:
                self.stages.append(self.current_stage)
                start, end = self.current_stage
                self.ax.axvline(x=start, color='r', linestyle='--')
                self.ax.axvline(x=end, color='r', linestyle='--')
                self.fig.canvas.draw()
                print(f"Stage {len(self.stages)}: from {start} to {end}, containing {end-start+1} data points")
                self.current_stage = []
        elif event.key == 'q':
            plt.close(self.fig)
            self.print_summary()

    def print_summary(self):
        print("\nSummary:")
        for i, (start, end) in enumerate(self.stages, 1):
            print(f"Stage {i}: from {start} to {end}, containing {end-start+1} data points")

if __name__ == "__main__":
    file_path = r".\data\standardization\wyc\wyc3_pHIT_lefteye.plist"
    data = load_plist(file_path)
    
    selector = StageSelectorTool(data)