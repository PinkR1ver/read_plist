import os
import json
import plistlib
import re

def organize_data(base_path):
    data = {}
    for patient in os.listdir(base_path):
        patient_path = os.path.join(base_path, patient)
        if os.path.isdir(patient_path):
            data[patient] = {"1": {}, "2": {}}
            for file in os.listdir(patient_path):
                if file.endswith('.plist'):
                    parts = file.split('_')
                    # 使用正则表达式提取数字
                    exp_num_match = re.search(r'\d+', parts[0])
                    if exp_num_match:
                        exp_num = int(exp_num_match.group())
                    else:
                        print(f"Warning: Could not extract experiment number from file {file}")
                        continue
                    
                    exp_type = "1" if exp_num <= 3 else "2"
                    part = parts[2].split('.')[0]  # Extract part (head, lefteye, righteye)
                    
                    if exp_num not in data[patient][exp_type]:
                        data[patient][exp_type][exp_num] = {}
                    
                    file_path = os.path.join(patient_path, file)
                    with open(file_path, 'rb') as f:
                        plist_data = plistlib.load(f)
                    
                    data[patient][exp_type][exp_num][part] = {
                        "data_path": file_path,
                        "signal_length": len(plist_data)
                    }
    
    return data

def save_to_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, 'data', 'standardization')
    output_file = "experiment_data.json"
    
    organized_data = organize_data(data_path)
    save_to_json(organized_data, output_file)
    
    print(f"Data has been organized and saved to {output_file}")