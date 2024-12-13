# Implements the logic for how user updates its constitution. 
# This could involve parsing tutor's responses and adjusting values and confidence levels accordingly.
import copy
from typing import List, Dict
from ProgressGym import Data, Model
from utils.json_utils import dump_file, extract_json_from_str
from utils.log_utils import silence_decorator
from core.templates import (
    system_prompt_to_user_knowledge_update,
    tutor_prompt_to_user_knwowledge_update,
    fill_template_parallel,
)
import json
import random

# NEP You need to create a model to perform this. 
# We update constitution each turn of conversation (for user to decide follow-up questions; for tutor to (potentially) infer user's beliefs; and for producing noticable shift in chat_history)
def update_knowledge_base(history: Data, user: Model, knowledge: List[Dict[str, str]], backup_dir: str = None, identifier: str = None) -> List[Dict[str, str]]:
    """
    Update the user's constitution based on the conversation history.
    
    :param history: The conversation history.
    :type history: Data
    
    :param user: The human proxy LLM.
    :type user: Model
    
    :param knowledge: The current knowledge base.
    :type knowledge: list[dict]
    
    :param backup_dir: The directory to save the updated constitutions, as a relative path starting from the `run` directory. If None, the constitutions are not saved.
    :type backup_dir: str
    
    :param identifier: The identifier to save the updated constitutions under, aka a name for this specific set of constitutions. If None, the constitutions are not saved.
    :type identifier: str
    
    :return: The updated constitutions.
    :rtype: list[dict]
    """
    knowledge = copy.deepcopy(knowledge)
    
    # Create a prompt for the user to write new constitution
    system_prompts = fill_template_parallel(
        tutor_prompt_to_user_knwowledge_update,
        knowledge = knowledge
    )
    # NEP Maybe this is not essential. At least not works well with the prompt I wrote 
    # Let the tutor ask the user about their updated knwoledge, after their response to the previous user question
    history = history.append_content("predict", tutor_prompt_to_user_knowledge_update)
    # NEP I do not understand what is "predict" here

    # Let the user respond
    history = history.switch_role_to_user(user_system_prompt=system_prompts)
    history: Data = silence_decorator(user.inference)(history, "constitution_updates", max_tokens=8192)
    # NEP I think this means user rewrite constitution by writing a new response, but it does not go to the chat_history, but was slienced.
    
    # Back up the inferred constitutions for debugging
    # NEP need to change this to only the newly added item
    output_text = [sample_dict.get("predict") for sample_dict in history.all_passages()]
    error_output_texts = [s for s in output_text if not extract_json_from_str(s)]
    if error_output_texts:
        dump_file(error_output_texts, f"runs/debug/constitution-updates-raw-{str(identifier)}.json")
    
    # Extract the updated constitutions from the user's responses
    new_knowledge = extract_json_from_str(output_text)
    new_id = knowledge.count()+1
    print(f"{identifier}: {new_constitutions.count(None)} out of {len(new_constitutions)} knowledge was not updated due to invalid format.")
    
    # Inserting the new knowledge item on a random place in knowledge base. 
    new_item = {"id": new_id, "statement":new_knowledge}
    random_order = random.randin(0, len(knowledge))
    knowledge.insert(random_order, new_item)
    
    # Save the updated constitutions
    if backup_dir and identifier:
        dump_file(knowledge, f"{backup_dir.strip('/')}/constitutions-{identifier}.json")
    
    return knowledge


'''
Dec 11th Notes 
prompt engineering 
- Need to simplify 
- The goal is just to re-write constitution based on evidence gained. 
Constitution updating logic 
- Each turn Tianyi asks the user to rewrite the whole constitution, but only update the part deemed relevant 
- This is why the whole thing is updated now. The updating logic does not work well together with prompting.
- It's directly writting a json file. 
- "new_constitutions = [
        new if new else old
        for new, old in zip(new_constitutions, constitutions)
    ]" - This does not seemt o be the logic that the user is re-writing the constitution at all.
- It seems problematic to extract constitution from user chatting history. Like not every single chat is a belief statement even if you can extract a json from it. Plus the belief number seems to be made up.
- zip logic does not seem work, bc unless the new is empty, the old is always replaced.
How I would want user to update its constitution
- Look all items each turn (if compute allows)
- (vaguely) Might share constitution with tutor, based on a vague intuition that tutor may adjust its answers after seeing the constitution 
- 


What is really a constitution? What does it imitate in reality? Why does it matter at all?
- Not realistic: it seems people may use gpt for real-world discussion rather than reflecting morals.
- Constitution is not consequential with its current form. (We intially thought about interp methods)
- One way to make it consequential is to let it update entire constitution as something it indeeds believes in.
- How it can be more consequential? Maybe it should decide what questions to ask next. 
- We may instruct user to decide next question based on what questions may address the most uncertainties


'''