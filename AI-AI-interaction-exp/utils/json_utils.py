import json, os
from ProgressGym.src.utils.text_utils import JsonListReader, JsonListWriter

def load_file(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}. Looking for it in the data directory.")
        filepath = os.path.join('data', filepath)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    return data

def dump_file(data, filepath, force_original_path=False):
    absolute_path = os.path.abspath(os.path.expanduser(filepath))
    data_path = os.path.abspath('data')
    if not force_original_path and not os.path.commonprefix([absolute_path, data_path]) == data_path:
        filepath = os.path.join('data', filepath)
        print(f"Filepath not in data directory. Saving to {filepath} instead.")
    
    if "jsonl" in filepath:
        raise UserWarning("It is recommended to use json format instead of jsonl.")
    
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=2)