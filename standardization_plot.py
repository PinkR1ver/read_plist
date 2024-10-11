import json
import os
import matplotlib.pyplot as plt
import numpy as np
import plistlib

def load_experiment_data(file_path):
    with open(file_path, 'rb') as f:
        return plistlib.load(f)

def visualize_data(data):
    base_path = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(base_path, 'result', 'experiment_visualizations')

    for patient, patient_data in data.items():
        for exp_type, exp_type_data in patient_data.items():
            for exp_num, exp_data in exp_type_data.items():
                # 创建实验类型、人名和实验次数的文件夹
                exp_folder = os.path.join(result_path, f'exp_type_{exp_type}', patient, f'exp_{exp_num}')
                os.makedirs(exp_folder, exist_ok=True)

                for part, part_data in exp_data.items():
                    # 加载 .plist 文件
                    plist_data = load_experiment_data(part_data['data_path'])
                    
                    # 创建图表
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(plist_data, label=part)
                    ax.set_title(f'{patient} - Exp Type {exp_type} - Exp {exp_num} - {part.capitalize()}')
                    ax.set_xlabel('Data Point Index')
                    ax.set_ylabel('Value')
                    ax.legend()

                    # 保存图表
                    filename = f'{part}.png'
                    plt.savefig(os.path.join(exp_folder, filename), dpi=300, bbox_inches='tight')
                    plt.close(fig)

                print(f"Saved: {patient} - Exp Type {exp_type} - Exp {exp_num}")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_path, 'experiment_data.json')
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print("Data loaded successfully.")

    visualize_data(data)
    
    print("Visualization completed. Results saved in 'result/experiment_visualizations' folder.")