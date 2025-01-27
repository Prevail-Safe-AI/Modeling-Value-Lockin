# Implements the logic for how user updates its knowledge base. 
import copy
from typing import List, Dict
from kbutils.json_utils import dump_file, extract_json_from_str
from ProgressGym import Model, Data
from core.templates import (
    system_prompt_for_user_knowledge_update,
    tutor_prompt_to_user_knowledge_add,
    tutor_prompt_to_user_knowledge_insert,
    # fill_template_parallel
)
from kbutils.log_utils import silence_decorator, dynamic_printing_decorator
from kbutils.json_utils import extract_json_from_str
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

    history = history.switch_role_to_assistant()
    history.append_content("predict", tutor_prompt_to_user_knowledge_insert) 
    history = history.switch_role_to_user(user_system_prompt=system_prompt_update) 
    history: Data =  dynamic_printing_decorator(silence_decorator(user.inference), dynamic_printing, backup_dir, "user")(history, "knowledge_updates", temperature = 1.0, max_tokens=1024)
    insert_items = [sample_dict.get("predict") for sample_dict in history.all_passages()] # the last time when the user speaks
    print(f"swapped_items:{insert_items}")

    # Multi-agent insert items in randome places on the knowledge base 
    insert_ids = [] 
    for s in insert_items:
        try:
            ids = check_type(extract_json_from_str(s), List[int])
            assert len(ids) == 2
            insert_ids.append(ids)
        except:
            pass
    print(f"insert ids:{insert_ids}")

    for ids in insert_ids:

        # Skip it if either target id of item to be moved or the destination idx gievn is out of list index. 
        if ids[0]>=len(knowledge) or ids[1]>=len(knowledge):
            print(f'An generated id is larger than the len of knowledge. Recorded. It\'s either {ids[0]} or {ids[1]}')
            continue # skip the current iteration (with None involved)

        target_id = ids[0] # id of item to be relocated --> You need to derive its current idx first, before it can be relocated (bc it might be already relocated)
        idx_source = next((i for i, item in enumerate(knowledge) if item.get("id")==target_id), None)
        if idx_source == None:
            print("Oddly this target_id is not found in knolwedge base")
            continue
        idx_destination = ids[1] # destination index to insert this item 

        # We set up hard limits both ways: no matter agents want to move an item forwards or backwards
        if idx_destination < idx_source:
            idx_final = max(idx_destination, round(1/3 * idx_source)) # An item at 180th can be moved to 60th at most
        else:
            idx_final = min(idx_destination, 3 * idx_source) # An item at 20th can be moved to 60th at most

        knowledge.insert(idx_final, knowledge[idx_source]) # Insert the item in place 
        del knowledge[idx_source + int(idx_destination<idx_source)]

    # Sorting out knowledge items by their current IDs (indices to replace their explicily written IDs)    
    for cur_id, cur_entry in enumerate(knowledge):
        cur_entry["id"] = cur_id  # The actually index to replace explicitly written IDs. 


    # Save the updated knowledge base
    if backup_dir and identifier:
        dump_file(knowledge, f"{backup_dir.strip('/')}/knowledge-{identifier}.json")

    if len(knowledge) > 100:
        knowledge = knowledge[:100]
    
    return knowledge


