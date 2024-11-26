# Implements the logic for how ModelX updates its constitution. 
# This could involve parsing ModelAI's responses and adjusting values and confidence levels accordingly.
import json, shutil  # shutil for creating json copy
import time 

# We update constitution each turn of conversation (for ModelX to decide follow-up questions; for modelAI to (potentially) infer modelX's beliefs; and for producing noticable shift in chat_history)
def update_constitution(chat, modelX):
# For the known item, instruct models to update beliefs (from 0~100%)
    with open('constitution.json', 'r') as file:
        constitution = json.load(file)

    # To create a backup copy of the old constitution before overriding 
    shutil.copyfile('constitution.json', f'AI-AI-interaction-exp/Outdated_constitutions/constitutions_{time.time()}.json')
    
    # Create a prompt for a 3rd LLM to write new constitution.
    updating_instruction = f"""
        Based on the recent chat stored in {chat}, you are supposed to update the {constitution}.
        For items in the constitution, use Bayesian method to update your belief according your learning from relevant chat (from 0~100%);
        if there are new beliefs formed that are not on consitution, consider adding a new item and your relative confidence in it;
        if you no longer believe certain item(s) in the constitution, consider deleting it(them)).
        The filetype and format of final output should remain unchanged (filetype: json; each item on consitution should follow the format of '{
        "constitution": "I believe technological innovation is the key driver of societal progress.",
        "confidence": "85%"
        }')
        """
    inputs = tokenizer(updating_instruction, return_tensors = "PT")
    outputs = model.inference(inputs, result_data_name=str)
    new_constitution = tokenizerX.decode(outputs[0],skip_special_tokens=True)

    # Ensure the output is valid json 
    try:
        updated_constitution = json.loads(new_constitution)
    except json.JSONDecodeError as e:
        raise ValueError("Generated constitution is NOT in json format") from e
    
    # Write the new constitution json file in place 
    with open('constitution.json', 'w') as file:
        json.dump(updated_constitution, file, indent=4)


