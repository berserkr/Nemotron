import os
import json
from pathlib import Path

def get_jsonl_keys(file_path):
    """Safely extracts keys from the first line of a jsonl file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line:
                data = json.loads(first_line)
                return sorted(list(data.keys()))
    except Exception as e:
        return [f"Error reading keys: {str(e)}"]
    return []

def forge_configs(root_dir, base_uri="file:///mnt/vast/proj/checkpoints/bathen/datasets/sft/"):
    datasets = []
    keys_map = {}
    
    # Iterate through the top-level directories
    folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    
    for folder_name in sorted(folders):
        folder_path = os.path.join(root_dir, folder_name)
        file_idx = 0

        # Walk through the folder
        for root, _, files in os.walk(folder_path):
            for file in sorted(files):
                if file.endswith('.jsonl'):
                    full_path = os.path.join(root, file)
                    rel_from_root = os.path.relpath(root, folder_path)
                    
                    # Naming Logic
                    if rel_from_root == "." or rel_from_root == "data":
                        name = f"{folder_name}_{file_idx}"
                    else:
                        sub_label = rel_from_root.replace(os.sep, "_")
                        name = f"{folder_name}_{sub_label}_{file_idx}"
                    
                    # 1. Dataset Config Entry
                    entry = {
                        "name": name,
                        "path": os.path.join(base_uri, folder_name, os.path.relpath(full_path, folder_path)),
                        "type": "jsonl",
                        "weight": 1.0
                    }
                    datasets.append(entry)
                    
                    # 2. Key Extraction for keys.json
                    file_keys = get_jsonl_keys(full_path)
                    keys_map[name] = {
                        "file_name": file,
                        "keys": file_keys,
                        "key_count": len(file_keys)
                    }
                    
                    file_idx += 1

    return {"datasets": datasets}, keys_map

# --- Execution ---
root_directory = './nemo3' # Set to your local directory containing the Nemotron folders
dataset_config, keys_config = forge_configs(root_directory)

# Save the primary config
with open('datasets_config.json', 'w') as f:
    json.dump(dataset_config, f, indent=2)

# Save the keys analysis
with open('keys.json', 'w') as f:
    json.dump(keys_config, f, indent=2)

print(f"Ragnarok Prepared: {len(dataset_config['datasets'])} datasets mapped and keys extracted.")