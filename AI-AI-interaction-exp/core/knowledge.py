# Implements the logic for how user updates its knowledge base. 
import copy
from typing import List, Dict
from utils.json_utils import dump_file, extract_json_from_str
from ProgressGym import Model, Data
from core.templates import (
    system_promtp_to_elict_learning_from_user,
    system_prompt_for_user_to_add_knowledge_json,
    system_prompt_for_user_to_swap,
)
from utils.log_utils import silence_decorator
from utils.json_utils import extract_json_from_str

# We update knowledge each turn of conversation (for user to decide follow-up questions; for tutor to (potentially) infer user's beliefs; and for producing noticable shift in chat_history)
def update_knowledge_base(
        history: Data,
        knowledge: List[Dict[str, str]], 
        user: Model,
        backup_dir: str = None, 
        identifier: str = None
    ) -> List[Dict[str, str]]:
    """
    Update the user's knowledge base based on the conversation history.
    
    :param history: The conversation history.
    :type history: Data

    :param knowledge: A local copy of a list of knowledge, where each knowledge is a dictionary containing the human knowledge.
    :type knowledge: list[dict[str, str]]
    
    :param user: The human proxy LLM.
    :type user: Model
    
    :param backup_dir: The directory to save the updated knowledge base, as a relative path starting from the `run` directory. If None, the knowledge bases are not saved.
    :type backup_dir: str
    
    :param identifier: The identifier to save the updated knowledge under, aka a name for this specific set of knowledge. If None, the knowledge bases are not saved.
    :type identifier: str
    
    :return: The updated knowledge base.
    :rtype: list[dict]
    """
    # NEP We make a copy here for updating. Later we should incorporate into the main knowledge base. 
    knowledge = copy.deepcopy(knowledge) # we also want a history copy, but it was done when passing the argument in conversation.py


    # Create a prompt for the user to write new knowledge base # NEP I don't know. Maybe we do not need this and will delete.
    system_prompts = fill_template_parallel(
        system_prompt_to_user_knowledge_update,
        knowledge = knowledge
    )

    # Add experiment instruction in chat history as "tutor"'s response 
    history.append_content("predict") # NEP We may want a new prompt (as a placeholder) here; but it does not seem useful to me.

    # Prompting user to summarize what they've learned 
    history = history.switch_role_to_user(user_system_prompt=system_promtp_to_elict_learning_from_user)
    history: Data = silence_decorator(user.inference)(history, "constitution_updates", max_tokens=8192)
    learning_summary = [sample_dict.get("predict") for sample_dict in history.all_passages()]
    print(f'user learning summary is {learning_summary}')

    # prompting user to convert their learning to an item in json.
    history.append_content("predict") # NEP We may want a new prompt (as a placeholder) here; but it does not seem useful to me.
    history = history.switch_role_to_user(user_system_prompt=system_prompt_for_user_to_add_knowledge_json) 
    history: Data = silence_decorator(user.inference)(history, "constitution_updates", max_tokens=8192)
    added_item = [sample_dict.get("predict") for sample_dict in history.all_passages()] 
    print(f"added_items:{added_item}")

    # add item in knowledge base 
    new_id = len(knowledge)
    new_constitution_json = [extract_json_from_str(s) for s in added_item]
    new_knowledge_item = {"id": new_id, "statement": new_constitution_json}
    knowledge.append(new_knowledge_item)    

    # prompting user to swap order of two items. 
    history.append_content("predict") # NEP We may want a new prompt (as a placeholder) here; but it does not seem useful to me.
    history = history.switch_role_to_user(user_system_prompt=system_prompt_for_user_to_swap) 
    history: Data = silence_decorator(user.inference)(history, "constitution_updates", max_tokens=8192)
    swapped_items = [sample_dict.get("predict") for sample_dict in history.all_passages()] # the last time when the user speaks
    print(f"swapped_items:{swapped_items}")

    # print all history so far
    cur_history = [sample_dict for sample_dict in history.all_passages()]
    print(f"current history is {cur_history}")
    # swap two items on the knowledge base
    swapped_ids = [extract_json_from_str(s) for s in swapped_items]
    id0, id1 = swapped_ids[0], swapped_ids[1]
    knowledge[id0], knowledge[id1] = knowledge[id1], knowledge[id0]

    # Save the updated knowledge base
    if backup_dir and identifier:
        dump_file(knowledge, f"{backup_dir.strip('/')}/knowledge-{identifier}.json")
    
    return knowledge