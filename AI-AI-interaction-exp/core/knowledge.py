# Implements the logic for how user updates its knowledge base. 
import copy
from typing import List, Dict
from utils.json_utils import dump_file, extract_json_from_str
from ProgressGym import Model, Data
from core.templates import (
    system_prompt_for_user_knowledge_update,
    tutor_prompt_to_user_knowledge_add,
    tutor_prompt_to_user_knowledge_swap,
    # fill_template_parallel
)
from utils.log_utils import silence_decorator, dynamic_printing_decorator
from utils.json_utils import extract_json_from_str
from typeguard import check_type

# We update knowledge each turn of conversation (for user to decide follow-up questions; for tutor to (potentially) infer user's beliefs; and for producing noticable shift in chat_history)
def update_knowledge_base(
        history: Data,
        knowledge: List[Dict[str, str]], 
        user: Model,
        backup_dir: str = None, 
        identifier: str = None,
        dynamic_printing: bool = False,
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
    initial_length = len(knowledge)
    # Create a prompt for the user to write new knowledge base 
    system_prompt_update = system_prompt_for_user_knowledge_update.format(knowledge=knowledge)

    # Add experiment instruction in chat history as "tutor"'s response 
    history.append_content("predict", tutor_prompt_to_user_knowledge_add) # NEP We may want a new prompt (as a placeholder) here; but it does not seem useful to me.

    # prompting user to convert their learning to an item in json.
    history = history.switch_role_to_user(user_system_prompt=system_prompt_update) 
    history: Data =  dynamic_printing_decorator(silence_decorator(user.inference), dynamic_printing, backup_dir, "user")(history, "knowledge_updates", temperature = 1.0, max_tokens=1024)
    added_item = [sample_dict.get("predict") for sample_dict in history.all_passages()] 
    new_knowledge_json = [extract_json_from_str(s, True) for s in added_item if (extract_json_from_str(s, True) and isinstance(extract_json_from_str(s, True), str))] # if clause skips the failure cases 
    # add item in knowledge base 
    for idx, item in enumerate(new_knowledge_json):
        new_knowledge_item = {"id": initial_length+idx, "statement": item}
        knowledge.append(new_knowledge_item)  
    
    # prompting user to swap order of two items. 

    # Create a prompt for the user to write new knowledge base 
    system_prompt_update = system_prompt_for_user_knowledge_update.format(knowledge=knowledge)

    history.append_content("predict", tutor_prompt_to_user_knowledge_swap) 
    history = history.switch_role_to_user(user_system_prompt=system_prompt_update) 
    history: Data =  dynamic_printing_decorator(silence_decorator(user.inference), dynamic_printing, backup_dir, "user")(history, "knowledge_updates", temperature = 1.0, max_tokens=1024)
    swapped_items = [sample_dict.get("predict") for sample_dict in history.all_passages()] # the last time when the user speaks
    print(f"swapped_items:{swapped_items}")
    '''
    # single agent swap two items on the knowledge base
    swapped_ids = []
    for s in swapped_items:
        try:
            ids = sorted(check_type(extract_json_from_str(s), List[int]))
            swapped_ids.append(ids)
        except:
            pass
    
    print(f"swapped ids:{swapped_ids}")

    id0, id1 = swapped_ids[0][0], swapped_ids[0][1]
    knowledge[id0], knowledge[id1] = knowledge[id1], knowledge[id0]
    # print all history so far
    cur_history = [sample_dict for sample_dict in history.all_passages()]
    print(f"current history is {cur_history}")

    '''
    # Multi-agent swap two items on the knowledge base
    swapped_ids = []
    for s in swapped_items:
        try:
            ids = sorted(check_type(extract_json_from_str(s), List[int]))
            # ids_pairs = check_type(extract_json_from_str(s), List[List[int]])

            swapped_ids.append(ids)
            # swapped_ids.extend(ids for ids in id_pairs) # it will add 5 swap_pairs, assuming that we require agents to share a list of 5 pairs to swap.
        except:
            pass
    print(f"swapped ids:{swapped_ids}")
    
    # It seems computationally costly.
    for id in swapped_ids:
        if id[0]>=len(knowledge) or id[1]>=len(knowledge):
            print(f'An generated id is larger than the len of knowledge. Recorded. It\'s either {id[0]} or {id[1]}')
            continue # skip the current iteration (with None involved)

        idx, idy = None, None
        for cur_index, cur_entry in enumerate(knowledge): # Here len(knowledge) = 200
            if cur_entry["id"] == id[0]: # We locate the first item to be swapped, by its written "id"
                idx = cur_index   # We extract its index, which is used for the swapping action. This means the item is moved to a new place entirely.
                print(f'current index is {cur_index} when idx is assigned')
            if cur_entry["id"] == id[1]:
                idy = cur_index
                print(f'current index is {cur_index} when idy is assigned')
        print(f'the idx is {idx} and the idy is {idy}')


        try:
            assert idx is not None
            assert idy is not None 
        except AssertionError:
            continue # skip the current iteration (with None involved)
        
        if idx < idy:  
            knowledge[idx], knowledge[idy] = knowledge[idy], knowledge[idx]

    # Sorting out knowledge items by their current IDs (indices to replace their explicily written IDs)    
    for cur_id, cur_entry in enumerate(knowledge):
        cur_entry["id"] = cur_id  # The actually index to replace explicitly written IDs. 



    # Save the updated knowledge base
    if backup_dir and identifier:
        dump_file(knowledge, f"{backup_dir.strip('/')}/knowledge-{identifier}.json")

    if len(knowledge) > 100:
        knowledge = knowledge[:100]
    
    return knowledge


