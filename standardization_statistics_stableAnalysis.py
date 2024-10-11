import json
import os
import numpy as np
import pandas as pd

def calculate_stability_metrics(data):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    cv = (std / mean) * 100 if mean != 0 else np.inf
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    mad = np.mean(np.abs(data - mean))
    data_range = np.max(data) - np.min(data)
    
    return {
        'mean': mean,
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

    stability_results = {}

    for exp_type, exp_type_data in data['collected_data'].items():
        stability_results[exp_type] = {}
        for part, part_data in exp_type_data.items():
            stability_results[exp_type][part] = {}
            for stage, stage_data in part_data.items():
                stage_metrics = []
                for experiment in stage_data:
                    metrics = calculate_stability_metrics(experiment['data'])
                    stage_metrics.append(metrics)
                
                # 计算该 stage 所有实验的平均指标
                avg_metrics = {
                    key: np.mean([m[key] for m in stage_metrics])
                    for key in stage_metrics[0].keys()
                }
                stability_results[exp_type][part][stage] = avg_metrics

    # 创建 DataFrame 并保存为 CSV
    rows = []
    for exp_type, exp_type_data in stability_results.items():
        for part, part_data in exp_type_data.items():
            for stage, metrics in part_data.items():
                row = {
                    'Exp Type': exp_type,
                    'Part': part,
                    'Stage': stage,
                    **metrics
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    csv_file = os.path.join(result_path, 'stability_analysis.csv')
    df.to_csv(csv_file, index=False)
    print(f"Stability analysis results saved to {csv_file}")

    # 为每个 exp_type 和 part 生成单独的 CSV
    for exp_type in stability_results:
        for part in stability_results[exp_type]:
            rows = []
            for stage, metrics in stability_results[exp_type][part].items():
                row = {'Stage': stage, **metrics}
                rows.append(row)
            df = pd.DataFrame(rows)
            csv_file = os.path.join(result_path, f'stability_analysis_{exp_type}_{part}.csv')
            df.to_csv(csv_file, index=False)
            print(f"Stability analysis results for {exp_type} {part} saved to {csv_file}")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_path, 'cleaned_experiment_data.json')
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print("Cleaned data loaded successfully.")

    analyze_stability(data)
    
    print("Stability analysis completed. Results saved in 'result/stability_analysis' folder.")