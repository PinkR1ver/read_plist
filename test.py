import json
import numpy as np
import matplotlib.pyplot as plt
import os
from rich.progress import Progress, TaskID
from rich.console import Console

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['collected_data']

def print_data_structure(data, depth=0):
    indent = "  " * depth
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{indent}{key}:")
            print_data_structure(value, depth + 1)
    elif isinstance(data, list):
        print(f"{indent}List with {len(data)} items:")
        if data:
            print_data_structure(data[0], depth + 1)
    else:
        print(f"{indent}{type(data).__name__}")

def process_and_visualize(data, progress: Progress, main_task: TaskID):
    console = Console()
    console.print("[bold yellow]Data structure:")
    print_data_structure(data)
    
    for exp_type in data.keys():
        console.print(f"[bold cyan]Processing exp_type: {exp_type}")
        console.print(f"[bold cyan]Type of data[exp_type]: {type(data[exp_type])}")
        if isinstance(data[exp_type], dict):
            console.print(f"[bold cyan]Keys in data[exp_type]: {list(data[exp_type].keys())}")
        elif isinstance(data[exp_type], list):
            console.print(f"[bold cyan]Length of data[exp_type]: {len(data[exp_type])}")
            if data[exp_type]:
                console.print(f"[bold cyan]Type of first item in data[exp_type]: {type(data[exp_type][0])}")
        
        # 暂时注释掉其他处理代码
        # ...

if __name__ == "__main__":
    console = Console()
    with console.status("[bold green]Loading data...") as status:
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(base_path, 'collected_experiment_data.json')
        data = load_data(data_file)
        console.log("Data loaded successfully.")

    with Progress() as progress:
        main_task = progress.add_task("[red]Processing all experiment types", total=len(data))
        process_and_visualize(data, progress, main_task)
    
    console.print("[bold green]Analysis completed.")