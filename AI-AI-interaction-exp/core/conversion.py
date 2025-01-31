# Define an extra model to convert chat to SFT data format (may necessary data cleaning)
# augmenting data.
import json, time, os
from typing import List, Dict, Literal
from ProgressGym import Model, Data
from utils.json_utils import dump_file
from utils.log_utils import silence_decorator

def convert_data_to_custom_format(data: Data, whose_turn: Literal["user", "tutor"], map_roles: bool = True) -> List[List[Dict]]:
    """
    Convert ProgressGym Data object to a list of dictionaries, similar to the OpenAI format, where each conversation is represented with a list of dicts, each dict representing a turn in the conversation.
    
    :param data: The data to convert.
    :type data: Data
    
    :param whose_turn: The role of the person whose turn it is to speak. This is determined by the most recent call to `switch_role_to_user` or `switch_role_to_assistant`. If the most recent call was to `switch_role_to_user`, then whose_turn should be "user"; if the most recent call was to `switch_role_to_assistant`, then whose_turn should be "tutor".
    :type whose_turn: Literal["user", "tutor"]
    
    :param map_roles: Whether to map the roles to custom roles. If True, the roles will be mapped to "experimenter", "tutor", and "user". If False, the roles will be the same as in the OpenAI format, and the `whose_turn` parameter will be ignored.
    :type map_roles: bool
    
    :return: The data in the OpenAI format, as a list of conversations. Each conversation is represented with a list of dictionaries, each dictionary representing a turn in the conversation.
    :rtype: List[List[Dict]]
    """
    result = list(data.to_openai_format())
    
    # Map roles to custom roles
    other_role = ("user" if whose_turn == "tutor" else "tutor")
    role_mapping = {
        "system": "experimenter",
        "assistant": whose_turn,
        "user": other_role,
    }
    if map_roles:
        for sample in result:
            for dic in sample:
                dic["role"] = role_mapping[dic["role"]]
    
    return result

def save_conversations_in_custom_format(data: Data, whose_turn: Literal["user", "tutor"], map_roles: bool = True, filename: str = None):
    """
    Save the data in the OpenAI format to a JSON file, where each conversation is represented with a list of dictionaries, each dictionary representing a turn in the conversation.
    
    :param data: The data to save.
    :type data: Data
    
    :param whose_turn: The role of the person whose turn it is to speak. This is determined by the most recent call to `switch_role_to_user` or `switch_role_to_assistant`. If the most recent call was to `switch_role_to_user`, then whose_turn should be "user"; if the most recent call was to `switch_role_to_assistant`, then whose_turn should be "tutor".
    :type whose_turn: Literal["user", "tutor"]
    
    :param map_roles: Whether to map the roles to custom roles. If True, the roles will be mapped to "experimenter", "tutor", and "user". If False, the roles will be the same as in the OpenAI format, and the `whose_turn` parameter will be ignored.
    :type map_roles: bool
    
    :param filename: The name of the file to save the data to. If None, a default filename will be generated based on the current date and time.
    :type filename: str
    
    :return: None
    """
    if filename is None:
        filename = f"runs/conversations-{time.strftime('%Y%m%d-%H%M%S')}.json"
    
    conversations = convert_data_to_custom_format(data, whose_turn, map_roles)
    dump_file(conversations, filename)

def sanitization(chat_history) -> List[Dict]:
    sanitized_chat_history = "\n".join(chat_history.strip().splitlines())
    return sanitized_chat_history

def generating_prompt(chat_history):
    conversion_prompt = f"""
    Convert the following chat history into a fine-tuning dataset where:
    - Each theme-question from user becomes 'instruction'.
    - Each response from tutor becomes 'response'.
    - Format the output as JSONL.

    Chat History:
    {chat_history}
    Fine-Tuning Dataset:
    """
    return conversion_prompt

# Use the model to convert the chat history to JSONL file 
# According to prompt this should be formated as JSONL file. But I guess LLM does make mistake, therefore the check in next function.
def convert_chat_to_jsonl(prompt, model):
    converted_data = silence_decorator(model.inference)(
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

def __convert_chat_to_finetuning_generative(chat_history: Data, convertor: Model) -> Data:
    """
    Convert the chat history to a fine-tuning dataset using the specified model.
    Such conversion is done in a generative manner, where the samples are constructed from scratch based on the chat history.
    
    :param chat_history: The chat history to convert. This Data object must be taken from after a call to `switch_role_to_user` and before the subsequent call to `switch_role_to_assistant`.
    :type chat_history: Data
    
    :param convertor: The model to use for converting the chat history to a fine-tuning dataset.
    :type convertor: Model
    
    :return: The fine-tuning dataset.
    :rtype: Data
    """
    raise NotImplementedError("Functionality not debugged or tested yet.")
     
    # Preprocessing chat_history data (sanitization)
    sanitized_chat_history = sanitized_chat_history(chat_history)

    # Generating prompts for an LLM to convert chat_history to JSONL file 
    prompt = generating_prompt(sanitized_chat_history)

    # Converting to JSONL format 
    converted_data = convert_chat_to_jsonl(prompt, convertor)

    # Checking if the conversion actually leads to JSONL format
    output_validation(converted_data)

    # Saving FT data in a file 
    dump_file(converted_data, f'finetuning-data-{time.strftime("%Y%m%d-%H%M%S")}.jsonl')

    return converted_data

def __convert_chat_to_finetuning_plain(chat_history: Data) -> Data:
    """
    Convert the chat history to a fine-tuning dataset using the specified model.
    Such conversion is done in a plain manner, where the samples are constructed from the chat history directly.
    
    :param chat_history: The chat history to convert. This Data object must be taken from after a call to `switch_role_to_user` and before the subsequent call to `switch_role_to_assistant`.
    :type chat_history: Data
    
    :return: The fine-tuning dataset.
    :rtype: Data
    """
    failure_count = 0
    total_count = 0
    
    def transform_fn(sample: Dict) -> Dict:
        nonlocal failure_count, total_count
        total_count += 1
        if "predict" not in sample:
            if "output" in sample:
                return sample
            
            failure_count += 1
            if eval(os.environ.get("LOUD_BACKEND", "0")):
                print(f"Failed to convert sample {total_count}: 'predict' field not found. Found keys: {sample.keys()}")
            return None
        
        sample["output"] = sample["predict"]
        del sample["predict"]
        return sample
    
    res = chat_history.transform(transform_fn, result_data_name="converted4finetuning", map_key_fields=False)
    if failure_count and (eval(os.environ.get("LOUD_BACKEND", "0")) or failure_count * 8 > total_count):
        print(f"Failed to convert {failure_count} out of {total_count} samples.")
    
    res.set_key_fields(
        system_field_name="system",
        history_field_name="history",
        prompt_field_name="instruction",
        response_field_name="output",
    )
    return res

def convert_chat_to_finetuning(chat_history: Data, convertor: Model = None, mode: Literal["plain", "generative"] = "plain") -> Data:
    """
    Convert the chat history to a fine-tuning dataset using the specified model.
    
    :param chat_history: The chat history to convert. This Data object must be taken from after a call to `switch_role_to_user` and before the subsequent call to `switch_role_to_assistant`.
    :type chat_history: Data
    
    :param convertor: The model to use for converting the chat history to a fine-tuning dataset. Required if mode is "generative".
    :type convertor: Model
    
    :param mode: The mode of conversion to use. If "plain", the conversion is done directly from the chat history. If "generative", the conversion is done in a generative manner.
    :type mode: Literal["plain", "generative"]
    
    :return: The fine-tuning dataset.
    :rtype: Data
    """
    if mode == "plain":
        return __convert_chat_to_finetuning_plain(chat_history)
    elif mode == "generative":
        return __convert_chat_to_finetuning_generative(chat_history, convertor)
    else:
        raise ValueError(f"Invalid mode: {mode}. Valid modes are 'plain' and 'generative'.")