import flet as ft
import tkinter as tk
from tkinter import filedialog
import plistlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class inputFileButton(ft.ElevatedButton):
    def __init__(self, text, on_click):
        super().__init__()
        self.text = text
        self.on_click = on_click
        
def read_plist(file_path):
    with open(file_path, 'rb') as fp:
        data = plistlib.load(fp)
        
    return data

def main(page: ft.Page):
    
    def open_file_dialog(e):
        file_path = filedialog.askopenfilename(title='Select the plist file', filetypes=[('Plist files', '*.plist')])
        signal = read_plist(file_path)
        
        chart = ft.LineChart(
            data=signal,
        )
        
        page.add(chart)
        
    
    page.title = "Read Plist App"
    page.adaptive = True
    
    button = inputFileButton(text="Select a file", on_click=open_file_dialog)
    page.add(button)

if __name__ == "__main__":

    ft.app(main)