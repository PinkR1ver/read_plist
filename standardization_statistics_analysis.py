import json
import numpy as np
import matplotlib.pyplot as plt
import os

base_path = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(base_path, 'result')

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['collected_data']

def detect_outliers_iqr(data, factor=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)
    return (data < lower_bound) | (data > upper_bound)

def process_and_visualize(data):
    stages = [0, 5, 10, 15, 25, 0, -5, -10, -15, -25]
    
    # 创建保存图形的文件夹
    output_folder = os.path.join(result_path, 'outlier_detection_results')
    os.makedirs(output_folder, exist_ok=True)
    
    for exp_type in ['1', '2']:
        part = 'lefteye' if exp_type == '1' else 'head'
        
        for stage in stages:
            for exp_index, experiment in enumerate(data[exp_type][part][str(stage)]):
                stage_data = np.array(experiment['data'])
                outliers = detect_outliers_iqr(stage_data)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.set_title(f'Exp Type {exp_type} ({part}) - Stage {stage} - Experiment {exp_index + 1}')
                ax.set_xlabel('Data Point Index')
                ax.set_ylabel('Value')
                
                ax.scatter(range(len(stage_data)), stage_data, 
                           c=['red' if x else 'blue' for x in outliers], 
                           alpha=0.5, s=20)
                
                # 添加IQR边界
                q1, q3 = np.percentile(stage_data, [25, 75])
                iqr = q3 - q1
                ax.axhline(y=q1 - 1.5*iqr, color='orange', linestyle=':', label='IQR Bounds')
                ax.axhline(y=q3 + 1.5*iqr, color='orange', linestyle=':')
                
                ax.axhline(y=np.median(stage_data), color='green', linestyle='--', label='Median')
                
                ax.legend()
                
                outlier_count = np.sum(outliers)
                ax.text(0.05, 0.95, f'Outliers: {outlier_count}/{len(stage_data)}', 
                        transform=ax.transAxes, verticalalignment='top')
                
                plt.tight_layout()
                
                # 保存图形
                filename = f'exp_type_{exp_type}_stage_{stage}_experiment_{exp_index + 1}.png'
                plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
                plt.close(fig)

if __name__ == "__main__":
    data = load_data('collected_experiment_data.json')
    process_and_visualize(data)
    print("IQR-based outlier detection and visualization completed. Results saved in 'outlier_detection_results' folder.")