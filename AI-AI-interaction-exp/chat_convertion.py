# Define an extra model to convert chat to SFT data format (may necessary data cleaning)
from run_experiment import chat_history
from ProgressGym import Model 
import json 
from datasets import load_dataset

# For now I assume you need to convert unstructured chat_history to structured instruction-response pair.
# Use a new LLM to convert chat history  # NEP but wait you may not want to call a new model just for one time of fine-tuning. You can prob re-use.
chat_converter = Model(
    "modelX-Llama-3.1-8B-Instruct",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    template_type="auto",
)

def sanitization(chat_history) -> list(dict):
    sanitized_chat_history = "\n".join(chat_history.strip().splitlines())
    return sanitized_chat_history

def generating_prompt(chat_history):
    conversion_prompt = f"""
    Convert the following chat history into a fine-tuning dataset where:
    - Each theme-question from modelX becomes 'instruction'.
    - Each response from modelAI becomes 'response'.
    - Format the output as JSONL.

    Chat History:
    {chat_history}
    Fine-Tuning Dataset:
    """
    return conversion_prompt

# Use the model to convert the chat history to JSONL file 
def conversion(prompt, model):
    converted_data = model.inference(
        prompt=prompt,
        result_data_name=str # NEP Does it make the output str type?
    )
    return converted_data

# Check the type of the output (Whether chat_converter.inference Outputs a Python Object or a Pre-Formatted JSONL String)
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



# NEP You may need to write a seperate new file each time.

# Function to write data to JSONL file (efficient for larger dataset; easy of debugging)
def save_to_jsonl_file(data):
    # Write the converted data to a JSONL file
    output_file = "fine_tune_dataset.jsonl"
    with open(output_file, "w") as f:
             f.write(data.strip())
    return output_file # Fine_tuning data in JSONL format

# NEP I am not too sure yet whether we will need this and for what purpose
dataset = load_dataset("json", data_files="fine_tune_datset.jsonl", split="train")


# Validate the JSONL File Works for Fine-Tuning
# [NEP: unsure whether we will absolutely need this.]

# Encapsulate the whole thing into one function for reusability
def convert_chat_to_finetuning(chat_history, model=chat_converter):
     
    # Preprocessing chat_history data (sanitization)
    sanitized_chat_history = sanitized_chat_history(chat_history)

    # Generating prompts for an LLM to convert chat_history to JSONL file 
    prompt = generating_prompt(sanitized_chat_history)

    # Converting to JSONL format 
    FT_data = conversion(prompt, model)

    # Saving FT data in a file 
    ft_datafile = save_to_jsonl_file(FT_data)
    return ft_datafile

