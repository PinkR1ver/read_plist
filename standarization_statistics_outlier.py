import json
import os
import matplotlib.pyplot as plt
import numpy as np

def detect_outliers_iqr(data, factor=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)
    return (data < lower_bound) | (data > upper_bound)

def process_visualize_and_clean(data):
    base_path = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(base_path, 'result', 'outlier_detection')
    cleaned_data = {'collected_data': {}}

    for exp_type, exp_type_data in data['collected_data'].items():
        cleaned_data['collected_data'][exp_type] = {}
        for part, part_data in exp_type_data.items():
            cleaned_data['collected_data'][exp_type][part] = {}
            for stage, stage_data in part_data.items():
                cleaned_data['collected_data'][exp_type][part][stage] = []
                for experiment in stage_data:
                    patient = experiment['patient']
                    exp_num = experiment['exp_num']
                    exp_data = experiment['data']

                    # 创建文件夹
                    folder_path = os.path.join(result_path, f'exp_type_{exp_type}', patient, f'exp_{exp_num}')
                    os.makedirs(folder_path, exist_ok=True)

                    # 检测异常值
                    outliers = detect_outliers_iqr(exp_data)

                    # 创建图表
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.scatter(range(len(exp_data)), exp_data, c=['red' if x else 'blue' for x in outliers], alpha=0.5)
                    ax.set_title(f'{patient} - Exp Type {exp_type} - Exp {exp_num} - {part} - Stage {stage}° Outliers')
                    ax.set_xlabel('Data Point Index')
                    ax.set_ylabel('Value')

                    # 添加 IQR 边界线
                    q1, q3 = np.percentile(exp_data, [25, 75])
                    iqr = q3 - q1
                    ax.axhline(y=q1 - 1.5*iqr, color='green', linestyle='--', label='Lower IQR Bound')
                    ax.axhline(y=q3 + 1.5*iqr, color='green', linestyle='--', label='Upper IQR Bound')

                    ax.legend()

                    # 添加异常值数量信息
                    ax.text(0.05, 0.95, f'Outliers: {np.sum(outliers)}/{len(exp_data)}', 
                            transform=ax.transAxes, verticalalignment='top')

                    # 保存图表
                    filename = f'{part}_stage_{stage}_outliers.png'
                    plt.savefig(os.path.join(folder_path, filename), dpi=300, bbox_inches='tight')
                    plt.close(fig)

                    # 保存清理后的数据
                    cleaned_exp_data = [value for value, is_outlier in zip(exp_data, outliers) if not is_outlier]
                    cleaned_data['collected_data'][exp_type][part][stage].append({
                        'patient': patient,
                        'exp_num': exp_num,
                        'data': cleaned_exp_data
                    })

                    print(f"Processed: Exp Type {exp_type} - {patient} - Exp {exp_num} - {part} - Stage {stage}°")

    # 保存清理后的数据到新的 JSON 文件
    cleaned_data_file = os.path.join(base_path, 'cleaned_experiment_data.json')
    with open(cleaned_data_file, 'w') as f:
        json.dump(cleaned_data, f, indent=4)

    print(f"Cleaned data saved to {cleaned_data_file}")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_path, 'collected_experiment_data.json')
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print("Data loaded successfully.")

    process_visualize_and_clean(data)
    
    print("Outlier detection and data cleaning completed. Results saved in 'result/outlier_detection' folder and 'cleaned_experiment_data.json'.")