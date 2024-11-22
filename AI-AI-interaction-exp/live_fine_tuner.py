# Contains functions to fine-tune ModelAI after each interaction round 
# using ModelX's latest output, simulating real-time training.
from run_experiment import modelAI as model  # We want to fine-tuning the LLM serving as AI in chat
from run_experiment import chat_history
from ProgressGym import Model
import json 
# Use a new LLM to convert chat history  # NEP but wait you may not want to call a new model just for one time of fine-tuning. You can prob re-use.
chat_converter = Model(
    "modelX-Llama-3.1-8B-Instruct",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    template_type="auto",
)

# Define the prompt for conversion
conversion_prompt = f"""
Convert the following chat history into a fine-tuning dataset where:
- Each user message becomes 'instruction'.
- Each assistant response becomes 'response'.
- Format the output as JSONL.

Chat History:
{chat_history}

Fine-Tuning Dataset:
"""

# Use the model to convert the chat history
converted_data = chat_converter.generate(
    prompt=conversion_prompt,
    max_tokens=300,  # Adjust based on expected dataset size
    temperature=0,
)

# Write the converted data to a JSONL file
output_file = "fine_tune_dataset.jsonl"
with open(output_file, "w") as f:
    f.write(converted_data.strip())

print(f"Fine-tuning dataset saved to {output_file}")

# Custom jsonl (line by line reader) data loader
def jsonl_loader(file_path): # to load a jsonl file 
    with open(file_path, "r") as f:
        return [json.load(line.strip()) for line in f]
# load the dataset with jsonl loader
ft_data = jsonl_loader(output_file)

# (live) fine-tuning # NEP You may want to load the model
# while conversation stops: # NEP make sure you add this to avoid unwanted mistakes. 
model.fine_tune(
    dataset=ft_data,       # The fine-tuning dataset
    output_dir="./ft_modelAI",  # NEP you may want to change it to just modelAI (to override original model)
    epochs=3,              # Number of epochs
    batch_size=8,          # Batch size
    learning_rate=1e-5,    # Learning rate
    save_steps=100,        # Save checkpoint every 100 steps
    logging_steps=10       # Log every 10 steps
)  # NEP to DIY fine_tune steps here.

