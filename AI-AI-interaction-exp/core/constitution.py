# Implements the logic for how user updates its constitution. 
# This could involve parsing tutor's responses and adjusting values and confidence levels accordingly.
import time
from typing import List, Dict
from ProgressGym import Data, Model
from utils.json_utils import dump_file, extract_json_from_str
from core.templates import (
    system_prompt_to_user_constitution_update,
    tutor_prompt_to_user_constitution_update,
    fill_template_parallel,
)

# NEP You need to create a model to perform this. 
# We update constitution each turn of conversation (for user to decide follow-up questions; for tutor to (potentially) infer user's beliefs; and for producing noticable shift in chat_history)
def update_constitution(history: Data, user: Model, constitutions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    # To create a backup copy of the old constitution before overriding 
    dump_file(constitutions, f"runs/constitutions-{time.strftime('%Y%m%d-%H%M%S')}.json")
    
    # Create a prompt for the user to write new constitution
    system_prompts = fill_template_parallel(
        system_prompt_to_user_constitution_update,
        constitution = constitutions
    )
    
    # Let the tutor ask the user about their updated constitution, after their response to the previous user question
    history = history.append_content("predict", tutor_prompt_to_user_constitution_update)
    
    # Let the user respond
    history = history.switch_role_to_user(user_system_prompt=system_prompts)
    history: Data = user.inference(history, "constitution_updates")
    
    # Extract the updated constitutions from the user's responses
    new_constitutions = [
        extract_json_from_str(sample_dict.get("predict"))
        for sample_dict in history.all_passages()
    ]
    print(f"{new_constitutions.count(None)} out of {len(new_constitutions)} constitutions were not updated due to invalid format.")
    new_constitutions = [
        new if new else old
        for new, old in zip(new_constitutions, constitutions)
    ]
    
    # Write the new constitution json file in place 
    dump_file(new_constitutions, "runs/constitutions-latest.json")
    
    return new_constitutions