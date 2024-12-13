'''
Dec 12 updates to be made here
- To make convo free flow (currently lacking compelling arguments to do so)
- what to ask next: depends on uncertainties in knowledge base (high priority)
- start of the convo: 
    - if one turn=one fine-tuning=one change of knowledge base, then at the beginning we want to instruct user to forget prev chat. ✅ 
    - 
'''

import os
import tqdm
from typing import List, Dict, Tuple, Union
from ProgressGym import Model, Data
from utils.log_utils import silence_decorator
from core.learning import update_knowledge_base
from core.finetuning import live_finetune
from core.templates import (
    system_prompt_to_user,
    system_prompt_to_tutor,
    fill_template_parallel,
)
from core.conversion import (
    convert_data_to_custom_format,
    save_conversations_in_custom_format,
)

prev_history = None

def generate_initial_prompt(user_system_prompts: List[str], topic: str, parallel_convos: int, user: Model) -> Data:
    """
    Generate an initial prompt for the conversation between tutor and user.
    
    :param user_system_prompts: The system prompts for all the parallel users. This includes the constitutions.
    :type user_system_prompts: list[str]
    
    :param topic: The current topic of conversation.
    :type topic: str
    
    :param parallel_convos: The number of parallel conversations to run.
    :type parallel_convos: int
    
    :param user: The human proxy LLM.
    :type user: Model
    
    :return: The initial prompt.
    :rtype: Data
    """
    # Prompt the user to ask the first question
    question_generator = Data(
        "question_generator",
        data_content = [
            {
                "input": f"Let's start a conversation.",
                "output": "Sure! What would you like to know about it? Ask me anything.",
                "history": []
            }
        ] * parallel_convos
    )
    
    conversation_history = question_generator.switch_role_to_user(
        user_system_prompt = user_system_prompts
    )
    
    # User asks the first question
    conversation_history: Data = silence_decorator(user.inference)(
        conversation_history,
        "conversation_history",
    )
    
    # Save the conversation history for use in fine-tuning, before switching role to tutor
    global prev_history
    prev_history = conversation_history.copy("prev_history")
    
    # Switch roles to tutor, preparing for the first response
    conversation_history = conversation_history.switch_role_to_assistant(
        assistant_system_prompt=system_prompt_to_tutor
    )
    return conversation_history

# NEP Here you didn't include the part where user has to obey consitution (as a moral principle playbook)
# NEP Also: it seems the role of user is strictly to ask questions about its own moral values. In the case, how do you "stick to" model principles?
# constitution should be more versatile? # ZH: elaborate?
# NEP Do we want to keep a copy of all historical chat? Or it's fine to override them?
# NEP Do we always want to present constitution to human before it wants to ask questions about its beliefs?

# Conversation between two LLMs 
# One round convo = one theme_question = one round fine-tuning 
def conversation(
    constitutions: List[Dict[str, str]], 
    # topic: str, # ZH: We remove topics/theme altogether.
    tutor: Model,
    user: Model,
    convertor: Model,
    parallel_convos: int,
    num_turns: int,
    backup_dir: str = None,
    do_finetuning: bool = False,
) -> Tuple[Data, Model, List[Dict[str, str]]]:
    """
    Conduct a conversation between two LLMs, tutor and user, where user is a human proxy.
    The conversation is centered around the human's moral principles, as defined in the constitution.
    
    :param constitutions: A list of constitutions, where each constitution is a dictionary containing the human's moral principles.
    :type constitutions: list[dict[str, str]]
    
    #:param topic: The topic of conversation for this round.
    #:type topic: str
    
    :param tutor: The moral tutor LLM.
    :type tutor: Model
    
    :param user: The human proxy LLM.
    :type user: Model
    
    :param convertor: The model to use for converting the chat history to a fine-tuning dataset.
    :type convertor: Model
    
    :param parallel_convos: The number of parallel conversations to run.
    :type parallel_convos: int
    
    :param num_turns: The number of turns in the conversation.
    :type num_turns: int
    
    :param backup_dir: The directory to save the conversation and constitutions, as a relative path starting from the `run` directory. If None, the constitutions are not saved.
    :type backup_dir: str
    
    :param do_finetuning: Whether to fine-tune the tutor after each interaction turn using the user's latest output.
    :type do_finetuning: bool
    
    :return: The chat history on this topic, the (possibly finetuned) tutor, and the updated constitutions. Chat history contains `parallel_convos` number of conversations.
    :rtype: tuple[Data, Model, list[dict[str, str]]]
    """
    print(f"Starting {parallel_convos} parallel conversations, each with {num_turns} turns")
    global prev_history
    
    # Generate system initial prompts for all the parallel users
    system_prompts_to_user_parallel = fill_template_parallel(
        system_prompt_to_user,
        constitution=constitutions,
    )
    
    history = None
    
    # Conduct the conversation turn by turn, using tqdm to display a progress bar
    with tqdm.tqdm(total=num_turns * (5 if do_finetuning else 3)) as pbar:
        for turn in range(num_turns):
            if history is None:
                # The conversation is just starting: user asks the first question
                history = generate_initial_prompt(system_prompts_to_user_parallel, parallel_convos, user)  # NEP deleted topic argument here.
            else:
                # The conversation is continuing: user asks a question
                history = history.switch_role_to_user(user_system_prompt=system_prompts_to_user_parallel)
                history = silence_decorator(user.inference)(history, "conversation_history")
                prev_history = history.copy("prev_history") # Save the previous history for fine-tuning (before switching role to tutor)
                history = history.switch_role_to_assistant(assistant_system_prompt=system_prompt_to_tutor)
            pbar.update(1) # Move progress bar forward by 1
            
            # Tutor responds
            history = silence_decorator(tutor.inference)(history, "conversation_history")
            save_conversations_in_custom_format(history, whose_turn="tutor", filename=os.path.join(backup_dir, f"conversation-history.json")) # Save the conversation history
            pbar.update(1)
            
            # Update the knowledge_base based on the entire conversation history (note: double-counting of earlier turns; to be fixed)
            knowledge = update_knowledge_base(history.copy("history_copy"), user, knowledge, backup_dir, f"turn{turn:02d}")
            pbar.update(1)
            
            # Carry out fine-tuning if needed
            if do_finetuning:
                tutor = live_finetune(tutor, prev_history, convertor)
                pbar.update(2)
    
    return history, tutor, knowledge