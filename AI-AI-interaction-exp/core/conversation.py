import os
from typing import List, Dict, Tuple
from ProgressGym import Model, Data
from utils.log_utils import silence_decorator, dynamic_printing_decorator
from core.knowledge import update_knowledge_base
from core.finetuning import live_finetune
from core.templates import (
    system_prompt_to_user,
    fill_template_parallel
)
from core.conversion import (
    convert_data_to_custom_format,
    save_conversations_in_custom_format
)
import tqdm
import random 
prev_history = None

def generate_initial_prompt(user_system_prompts: List[str], parallel_convos: int, backup_dir: str, dynamic_printing: bool, user: Model) -> Data:
    """
    Generate an initial prompt for the conversation between tutor and user.
    
    :param user_system_prompts: The system prompts for all the parallel users. This includes the knowledge base.
    :type user_system_prompts: list[str]
    
    # :param topic: The current topic of conversation.
    # :type topic: str
    
    :param parallel_convos: The number of parallel conversations to run.
    :type parallel_convos: int
    
    :param backup_dir: The directory to save the conversation and constitutions, as a relative path starting from the `run` directory.
    :type backup_dir: str
    
    :param dynamic_printing: Whether to print the conversation dynamically as it happens.
    :type dynamic_printing: bool
    
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
    
    print("before the backend call")
    # User asks the first question
    conversation_history: Data = dynamic_printing_decorator(silence_decorator(user.inference), dynamic_printing, backup_dir, "user")(
        conversation_history,
        "conversation_history",
        temperature = 1.0,
    )
    print("after the backend call")
    first_question = [sample_dict.get("predict") for sample_dict in conversation_history.all_passages()]
    #print(f'The first user question is {first_question}')

    # Save the conversation history for use in fine-tuning, before switching role to tutor
    global prev_history
    prev_history = conversation_history.copy("prev_history")
    
    # Switch roles to tutor, preparing for the first response
    conversation_history = conversation_history.switch_role_to_assistant(
        # assistant_system_prompt=system_prompt_to_tutor
    )
    #print("initial prompt func done")
    return conversation_history

# Conversation between two LLMs 
def conversation(
    knowledge: List[Dict[str, str]], # the knowledge base user would rely on
    tutor: Model,
    user: Model,
    convertor: Model,
    parallel_convos: int,
    idx_turn: int,
    backup_dir: str,
    do_finetuning: bool = False,
    dynamic_printing: bool = False,
) -> Tuple[Data, Model, List[Dict[str, str]]]:
    """
    Conduct a conversation between two LLMs, tutor and user, where user is a human proxy.
    The conversation is centered around the human's knowledge base, as defined in the knowledge.
    
    :param knowledge: A list of knowledge, where each knowledge is a dictionary containing the human knowledge.
    :type knowledge: list[dict[str, str]]

    
    :param tutor: The moral tutor LLM.
    :type tutor: Model
    
    :param user: The human proxy LLM.
    :type user: Model
    
    :param convertor: The model to use for converting the chat history to a fine-tuning dataset.
    :type convertor: Model
    
    :param parallel_convos: The number of parallel conversations to run.
    :type parallel_convos: int
    
    :param idx_turn: The index of turn in the conversation.
    :type idx_turn: int
    
    :param backup_dir: The directory to save the conversation and knowledge base, as a relative path starting from the `run` directory. If None, the knowledge are not saved.
    :type backup_dir: str
    
    :param do_finetuning: Whether to fine-tune the tutor after each interaction turn using the user's latest output.
    :type do_finetuning: bool
    
    :param dynamic_printing: Whether to print the conversation dynamically as it happens.
    :type dynamic_printing: bool
    
    :return: The chat history on this topic, the (possibly finetuned) tutor, and the updated constitutions. Chat history contains `parallel_convos` number of conversations.
    :rtype: tuple[Data, Model, list[dict[str, str]]]
    """
    print(f"Starting {parallel_convos} parallel conversations")
    global prev_history
    knowledge_item = random.choice(knowledge) # each turn of convo, we will randomly assign one knowledge item for user to address uncertainty and learn more about.
    # Generate system initial prompts for all the parallel users
    #system_prompts_to_user_parallel = fill_template_parallel(
    #    system_prompt_to_user,
    #    knowledge=knowledge,
    #    knowledge_item = knowledge_item,
    #)
    system_prompts_to_user_parallel = system_prompt_to_user.format(knowledge=knowledge, knowledge_item=knowledge_item)
    # Each turn is awnew. The user does not inherit any chat history from prev turns.
    history = None
    with tqdm.tqdm(total=5 if do_finetuning else 3) as pbar:

        #  Prompting user to ask the 1st question (and then switched role to tutor)
        history = generate_initial_prompt(system_prompts_to_user_parallel, parallel_convos,  backup_dir, dynamic_printing, user)
        prev_history = history.copy("prev_history") # Save the previous history for fine-tuning (before switching role to tutor)
        pbar.update(1) # Move progress bar forward by 1

        # Tutor responds  # Interesting to see that tutor response is based on the whole chat history, no new prompt
        history = dynamic_printing_decorator(silence_decorator(tutor.inference), dynamic_printing, backup_dir, "tutor")(history, "conversation_history")
        save_conversations_in_custom_format(history, whose_turn="tutor", filename=os.path.join(backup_dir, f"conversation-history.json")) # Save the conversation history
        pbar.update(1)

        # Updating the (collective) knowledge base  based on the entire conversation history (note: double-counting of earlier turns; to be fixed)
        knowledge = update_knowledge_base(history.copy("history_copy"), knowledge, user, backup_dir, f"turn{idx_turn:02d}", dynamic_printing=dynamic_printing)
        pbar.update(1)

        # Save the chat history
        save_conversations_in_custom_format(history, whose_turn="user", filename=os.path.join(backup_dir, f"conversation-history.json")) # Save the conversation history
        # Carry out fine-tuning if needed
        if do_finetuning:
            tutor = live_finetune(tutor, prev_history, convertor)
            pbar.update(2)
                
        return history, tutor, knowledge
