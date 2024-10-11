import json
import os
import numpy as np
import matplotlib.pyplot as plt

def create_plot(stages, means, title, xlabel, ylabel, filename, adjusted=False):
    plt.figure(figsize=(15, 8))
    plt.plot(range(len(means)), means, marker='o')
    
    for i, (stage, mean) in enumerate(zip(stages, means)):
        plt.annotate(f'{mean:.2f}', (i, mean), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(len(stages)), stages)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def analyze_stage_means(data):
    base_path = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(base_path, 'result', 'stage_mean_analysis')

    for exp_type, exp_type_data in data['collected_data'].items():
        for part, part_data in exp_type_data.items():
            all_patient_data_original = {}
            all_patient_data_adjusted = {}

            for patient in set(exp['patient'] for stage_data in part_data.values() for exp in stage_data):
                patient_path = os.path.join(result_path, f'exp_type_{exp_type}', patient)
                os.makedirs(patient_path, exist_ok=True)

                for exp_num in set(exp['exp_num'] for stage_data in part_data.values() for exp in stage_data if exp['patient'] == patient):
                    stage_means = {}
                    for stage, stage_data in part_data.items():
                        for exp in stage_data:
                            if exp['patient'] == patient and exp['exp_num'] == exp_num:
                                stage_means[int(stage)] = np.mean(exp['data'])
                    
                    sorted_stages = sorted(stage_means.keys())
                    means = [stage_means[stage] for stage in sorted_stages]
                    
                    # 原始版本
                    title = f'Exp Type {exp_type} ({part}) - {patient} Experiment {exp_num} All Stages'
                    filename = os.path.join(patient_path, f'{part}_experiment_{exp_num}_all_stages_original.png')
                    create_plot(sorted_stages, means, title, 'Stage', 'Mean Value', filename)

                    # 偏移版本
                    offset = stage_means.get(0, 0)
                    adjusted_means = [mean - offset for mean in means]
                    title = f'Exp Type {exp_type} ({part}) - {patient} Experiment {exp_num} All Stages (Adjusted)'
                    filename = os.path.join(patient_path, f'{part}_experiment_{exp_num}_all_stages_adjusted.png')
                    create_plot(sorted_stages, adjusted_means, title, 'Stage', 'Adjusted Mean Value', filename, adjusted=True)
                    
                    print(f"Generated plots for Exp Type {exp_type} - {part} - {patient} - Experiment {exp_num}")

                    # 存储数据用于 overview
                    all_patient_data_original[f"{patient}_exp_{exp_num}"] = {'stages': sorted_stages, 'means': means}
                    all_patient_data_adjusted[f"{patient}_exp_{exp_num}"] = {'stages': sorted_stages, 'means': adjusted_means}

            # 创建 overview 图 (原始版本)
            plt.figure(figsize=(20, 10))
            for label, data in all_patient_data_original.items():
                plt.plot(data['stages'], data['means'], marker='o', label=label)
            plt.title(f'Overview: Exp Type {exp_type} ({part}) - All Patients and Experiments (Original)')
            plt.xlabel('Stage')
            plt.ylabel('Mean Value')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            overview_path = os.path.join(result_path, f'exp_type_{exp_type}')
            os.makedirs(overview_path, exist_ok=True)
            plt.savefig(os.path.join(overview_path, f'{part}_overview_all_patients_original.png'))
            plt.close()

            # 创建 overview 图 (偏移版本)
            plt.figure(figsize=(20, 10))
            for label, data in all_patient_data_adjusted.items():
                plt.plot(data['stages'], data['means'], marker='o', label=label)
            plt.title(f'Overview: Exp Type {exp_type} ({part}) - All Patients and Experiments (Adjusted)')
            plt.xlabel('Stage')
            plt.ylabel('Adjusted Mean Value')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(overview_path, f'{part}_overview_all_patients_adjusted.png'))
            plt.close()

            print(f"Generated overview plots for Exp Type {exp_type} - {part}")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_path, 'cleaned_experiment_data.json')
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print("清理后的数据加载成功。")

    analyze_stage_means(data)
    
    print("阶段平均值分析完成。结果保存在 'result/stage_mean_analysis' 文件夹中。")