# Implements the logic for how user updates its knowledge base. 
# This could involve parsing tutor's responses and adjusting values and confidence levels accordingly.
import copy
from typing import List, Dict
from ProgressGym import Data, Model
from utils.json_utils import dump_file # extract_json_from_str
from utils.log_utils import silence_decorator
from core.templates import (
    system_prompt_to_user_knowledge_update,
    tutor_prompt_to_user_knowledge_update,
    fill_template_parallel,
)
import json
import random

# We update knowledge each turn of conversation (for user to decide follow-up questions; for tutor to (potentially) infer user's beliefs; and for producing noticable shift in chat_history)
def update_knowledge_base(history: Data, user: Model, knowledge: List[Dict[str, str]], backup_dir: str = None, identifier: str = None) -> List[Dict[str, str]]:
    """
    Update the user's knowledge base based on the conversation history.
    
    :param history: The conversation history.
    :type history: Data
    
    :param user: The human proxy LLM.
    :type user: Model
    
    :param knowledge: The current knowledge base.
    :type knowledge: list[dict]
    
    :param backup_dir: The directory to save the updated knowledge base, as a relative path starting from the `run` directory. If None, the knowledge bases are not saved.
    :type backup_dir: str
    
    :param identifier: The identifier to save the updated knowledge under, aka a name for this specific set of knowledge. If None, the knowledge bases are not saved.
    :type identifier: str
    
    :return: The updated knowledge base.
    :rtype: list[dict]
    """
    knowledge = copy.deepcopy(knowledge)

    # add item in knowledge base 
    new_knowledge_item = [sample_dict.get("predict") for sample_dict in history.all_passages()]   # NEP need to change this to only the newly added item
    random_order = random.randin(0, len(knowledge))
    knowledge.insert(random_order, new_knowledge_item)    

    # swap two items on the knowledge base 
    

    # reorder the whole knowledge base 


    # Create a prompt for the user to write new knowledge
    system_prompts = fill_template_parallel(
        system_prompt_to_user_knowledge_update,
        knowledge = knowledge
    )
    # NEP Maybe this is not essential. At least not works well with the prompt I wrote 
    # Let the tutor ask the user about their updated knwoledge, after their response to the previous user question
    history = history.append_content("predict", tutor_prompt_to_user_knowledge_update)
    # NEP I do not understand what is "predict" here

    # Let the user respond
    history = history.switch_role_to_user(user_system_prompt=system_prompts)
    history: Data = silence_decorator(user.inference)(history, "knowledge_updates", max_tokens=8192)
    
    # Back up the inferred knowledge base for debugging
    # NEP need to change this to only the newly added item
    output_text = [sample_dict.get("predict") for sample_dict in history.all_passages()]
    error_output_texts = [s for s in output_text if not extract_json_from_str(s)]
    if error_output_texts:
        dump_file(error_output_texts, f"runs/debug/knowledge-updates-raw-{str(identifier)}.json")
    

    
    # Save the updated knowledge base
    if backup_dir and identifier:
        dump_file(knowledge, f"{backup_dir.strip('/')}/knowledge-{identifier}.json")
    
    return knowledge