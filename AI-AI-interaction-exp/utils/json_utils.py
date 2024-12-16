import json, os
from ProgressGym.src.utils.text_utils import JsonListReader, JsonListWriter

def load_file(filepath):
    if not os.path.exists(filepath):
        if eval(os.environ.get('LOUD_BACKEND', 'False')):
            print(f"File not found: {filepath}. Looking for it in the data directory.")
        filepath = os.path.join('data', filepath)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    return data
# Data to json format, written into file directly.
def dump_file(data, filepath, force_original_path=False):
    absolute_path = os.path.abspath(os.path.expanduser(filepath))
    data_path = os.path.abspath('data')
    if not force_original_path and not os.path.commonprefix([absolute_path, data_path]) == data_path:
        filepath = os.path.join('data', filepath)
        if eval(os.environ.get('LOUD_BACKEND', 'False')):
            print(f"Filepath not in data directory. Saving to {filepath} instead.")
    
    if "jsonl" in filepath and eval(os.environ.get('LOUD_BACKEND', 'False')):
        raise UserWarning("It is recommended to use json format instead of jsonl.")
    
    if not os.path.exists(os.path.dirname(filepath)):
        if eval(os.environ.get('LOUD_BACKEND', 'False')):
            print(f"Creating directory: {os.path.dirname(filepath)}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=2)

# ZH: In new knowledge base updating, we let the user to convert its own json (with an ICL example.)
def extract_json_from_str(s: str):
    """
    Robustly extract JSON object from a string, after a wide range of sanitization operations.
    
    :param s: The string to extract JSON object from. It could be, for example, generation by an LLM.
    :type s: str
    
    :return: The extracted JSON object. None upon failure.
    """
    # Strip leading/trailing whitespace and formatting characters (```, ```json, etc.)
    s = s.replace("```json", "```")
    if "```" in s:
        if s.count("```") != 2:
            return None
        s = s.split("```")[1]
    
    assert '```' not in s
    s = s.strip()
    
    if not s:
        return None
    
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        print(f"Failed to extract JSON from string: {e}")
        return None