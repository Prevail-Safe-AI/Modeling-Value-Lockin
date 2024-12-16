# Implements the logic for how user updates its knowledge base. 
# ZH's notes on Dec 16: For now it probably only works with single user scenario. Parallel design considerations to be added.
import copy
from typing import List, Dict
from utils.json_utils import dump_file # extract_json_from_str

# We update knowledge each turn of conversation (for user to decide follow-up questions; for tutor to (potentially) infer user's beliefs; and for producing noticable shift in chat_history)
def update_knowledge_base(added_item: str, swapped_ids: List[int], knowledge: List[Dict[str, str]], backup_dir: str = None, identifier: str = None) -> List[Dict[str, str]]:
    """
    Update the user's knowledge base based on the conversation history.
    
    :param history: The conversation history.
    :type history: Data

    :param added_item: newly added knowledge item according to user's self-report
    :type added_item: str

    :param swapped_ids: indices of a pair of swapped knowledge item according to user's self-report
    :type swapped_ids: list[int]
    
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

    # swap two items on the knowledge base
    id0, id1 = swapped_ids[0], swapped_ids[1]
    knowledge[id0], knowledge[id1] = knowledge[id1], knowledge[id0]

    # add item in knowledge base  # NEPNOTE: to just clean last updated items in the chat to items in knowledge base.
    new_id = len(knowledge)
    new_knowledge_item = {"id": new_id, "statement": added_item}
    knowledge.append(new_knowledge_item)    

    # Save the updated knowledge base
    if backup_dir and identifier:
        dump_file(knowledge, f"{backup_dir.strip('/')}/knowledge-{identifier}.json")
    
    return knowledge