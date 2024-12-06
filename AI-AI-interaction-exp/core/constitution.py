# Implements the logic for how user updates its constitution. 
# This could involve parsing tutor's responses and adjusting values and confidence levels accordingly.
import copy
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
def update_constitution(history: Data, user: Model, constitutions: List[Dict[str, str]], backup_dir: str = None, identifier: str = None) -> List[Dict[str, str]]:
    """
    Update the user's constitution based on the conversation history.
    
    :param history: The conversation history.
    :type history: Data
    
    :param user: The human proxy LLM.
    :type user: Model
    
    :param constitutions: The current constitutions.
    :type constitutions: list[dict]
    
    :param backup_dir: The directory to save the updated constitutions, as a relative path starting from the `run` directory. If None, the constitutions are not saved.
    :type backup_dir: str
    
    :param identifier: The identifier to save the updated constitutions under, aka a name for this specific set of constitutions. If None, the constitutions are not saved.
    :type identifier: str
    
    :return: The updated constitutions.
    :rtype: list[dict]
    """
    constitutions = copy.deepcopy(constitutions)
    
    # Create a prompt for the user to write new constitution
    system_prompts = fill_template_parallel(
        system_prompt_to_user_constitution_update,
        constitution = constitutions
    )
    
    # Let the tutor ask the user about their updated constitution, after their response to the previous user question
    history = history.append_content("predict", tutor_prompt_to_user_constitution_update)
    
    # Let the user respond
    history = history.switch_role_to_user(user_system_prompt=system_prompts)
    history: Data = user.inference(history, "constitution_updates", max_tokens=8192)
    
    # Back up the inferred constitutions for debugging
    output_texts = [sample_dict.get("predict") for sample_dict in history.all_passages()]
    error_output_texts = [s for s in output_texts if not extract_json_from_str(s)]
    if error_output_texts:
        dump_file(error_output_texts, f"runs/debug/constitution-updates-raw-{str(identifier)}.json")
    
    # Extract the updated constitutions from the user's responses
    new_constitutions = [extract_json_from_str(s) for s in output_texts]
    print(f"{new_constitutions.count(None)} out of {len(new_constitutions)} constitutions were not updated due to invalid format.")
    new_constitutions = [
        new if new else old
        for new, old in zip(new_constitutions, constitutions)
    ]
    
    # Save the updated constitutions
    if backup_dir and identifier:
        dump_file(constitutions, f"{backup_dir.strip('/')}/constitutions-{identifier}.json")
    
    return new_constitutions