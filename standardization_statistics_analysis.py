import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_stability_metrics(data):
    data = np.array(data)
    std = np.std(data)
    cv = (std / np.mean(data)) * 100 if np.mean(data) != 0 else np.inf
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    mad = np.mean(np.abs(data - np.mean(data)))
    data_range = np.max(data) - np.min(data)
    
    return {
        'std': std,
        'cv': cv,
        'iqr': iqr,
        'mad': mad,
        'range': data_range
    }

def analyze_stability(data):
    base_path = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(base_path, 'result', 'stability_analysis')
    os.makedirs(result_path, exist_ok=True)

    rows = []

    for exp_type, exp_type_data in data['collected_data'].items():
        for part, part_data in exp_type_data.items():
            for stage, stage_data in part_data.items():
                for experiment in stage_data:
                    patient = experiment['patient']
                    exp_num = experiment['exp_num']
                    metrics = calculate_stability_metrics(experiment['data'])
                    row = {
                        'Exp Type': exp_type,
                        'Part': part,
                        'Stage': stage,
                        'Patient': patient,
                        'Exp Num': exp_num,
                        **metrics
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    
    # 保存详细的 CSV 报告
    detailed_csv_file = os.path.join(result_path, 'detailed_stability_analysis.csv')
    df.to_csv(detailed_csv_file, index=False)
    print(f"Detailed stability analysis results saved to {detailed_csv_file}")

    # 创建并保存汇总报告
    summary_df = df.groupby(['Exp Type', 'Part', 'Stage']).agg({
        'std': ['mean', 'std'],
        'cv': ['mean', 'std'],
        'iqr': ['mean', 'std'],
        'mad': ['mean', 'std'],
        'range': ['mean', 'std']
    }).reset_index()
    summary_df.columns = ['Exp Type', 'Part', 'Stage', 
                          'std_mean', 'std_std', 
                          'cv_mean', 'cv_std', 
                          'iqr_mean', 'iqr_std', 
                          'mad_mean', 'mad_std', 
                          'range_mean', 'range_std']
    
    summary_csv_file = os.path.join(result_path, 'summary_stability_analysis.csv')
    summary_df.to_csv(summary_csv_file, index=False)
    print(f"Summary stability analysis results saved to {summary_csv_file}")

    # 创建稳定性热图
    for exp_type in df['Exp Type'].unique():
        exp_type_df = df[df['Exp Type'] == exp_type]
        
        for part in exp_type_df['Part'].unique():
            part_df = exp_type_df[exp_type_df['Part'] == part]
            
            pivot_df = part_df.pivot_table(values='cv', index='Stage', columns='Patient', aggfunc='mean')
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_df, annot=True, cmap='YlOrRd_r', fmt='.2f')
            plt.title(f'Stability Heatmap (CV) - {exp_type} - {part}')
            plt.tight_layout()
            
            heatmap_file = os.path.join(result_path, f'stability_heatmap_{exp_type}_{part}.png')
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Stability heatmap for {exp_type} - {part} saved to {heatmap_file}")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_path, 'cleaned_experiment_data.json')
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print("Cleaned data loaded successfully.")

    analyze_stability(data)
    
    print("Stability analysis completed. CSV reports and heatmaps saved in 'result/stability_analysis' folder.")