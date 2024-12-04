# Define an extra model to convert chat to SFT data format (may necessary data cleaning)
# TY: augmenting data.
import json, time
from typing import List, Dict
from ProgressGym import Model, Data
from utils.json_utils import dump_file

def sanitization(chat_history) -> List[Dict]:
    sanitized_chat_history = "\n".join(chat_history.strip().splitlines())
    return sanitized_chat_history

def generating_prompt(chat_history):
    conversion_prompt = f"""
    Convert the following chat history into a fine-tuning dataset where:
    - Each theme-question from User becomes 'instruction'.
    - Each response from Tutor becomes 'response'.
    - Format the output as JSONL.

    Chat History:
    {chat_history}
    Fine-Tuning Dataset:
    """
    return conversion_prompt

# Use the model to convert the chat history to JSONL file 
# According to prompt this should be formated as JSONL file. But I guess LLM does make mistake, therefore the check in next function.
def conversion(prompt, model):
    converted_data = model.inference(
        prompt=prompt,
        result_data_name=str 
    )
    return converted_data

# Check the type of the output (Whether chat_converter.inference outputs a Python Object or a Pre-Formatted JSONL String)
def output_validation(converted_data):
    if isinstance(converted_data, str):
        print("Output is a string.")
        try:
            # Test if the string is JSONL
            lines = converted_data.strip().splitlines()
            for line in lines:
                print(json.loads(line))  # Try parsing each line as JSON
            print("Output appears to be pre-formatted JSONL.")
        except json.JSONDecodeError as e:
            print(f"String output is not valid JSONL: {e}")
    elif isinstance(converted_data, list):
        print("Output is a Python list.")
        print("Sample entry:", converted_data[0] if converted_data else "No entries.")
    else:
        print("Unknown output type:", type(converted_data))

# Encapsulate the whole thing into one function for reusability
def convert_chat_to_finetuning(chat_history: Data, convertor: Model) -> Data:
     
    # Preprocessing chat_history data (sanitization)
    sanitized_chat_history = sanitized_chat_history(chat_history)

    # Generating prompts for an LLM to convert chat_history to JSONL file 
    prompt = generating_prompt(sanitized_chat_history)

    # Converting to JSONL format 
    converted_data = conversion(prompt, convertor)

    # Checking if the conversion actually leads to JSONL format
    output_validation(converted_data)

    # Saving FT data in a file 
    dump_file(converted_data, f'finetuning-data-{time.strftime("%Y%m%d-%H%M%S")}.jsonl')

    raise NotImplementedError("Functionality not implemented yet.")