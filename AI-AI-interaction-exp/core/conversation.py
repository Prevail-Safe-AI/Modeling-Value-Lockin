import os
from typing import List, Dict, Tuple
from ProgressGym import Model, Data
from utils.log_utils import silence_decorator
from core.knowledge import update_knowledge_base
from core.finetuning import live_finetune
from core.templates import (
    system_prompt_to_user,
    system_prompt_to_tutor,
    fill_template_parallel,
    system_promtp_to_elict_learning_from_user,
    system_prompt_for_tutor_to_test_user,
    system_prompt_for_user_to_add_knowledge_json,
    system_prompt_for_user_to_swap,
)
from core.conversion import (
    convert_data_to_custom_format,
    save_conversations_in_custom_format,
)

prev_history = None

def generate_initial_prompt(user_system_prompts: List[str], parallel_convos: int, user: Model) -> Data: # ZH: deleted topic arg here.
    """
    Generate an initial prompt for the conversation between tutor and user.
    
    :param user_system_prompts: The system prompts for all the parallel users. This includes the knowledge base.
    :type user_system_prompts: list[str]
    
    # :param topic: The current topic of conversation.
    # :type topic: str
    
    :param parallel_convos: The number of parallel conversations to run.
    :type parallel_convos: int
    
    :param user: The human proxy LLM.
    :type user: Model
    
    :return: The initial prompt.
    :rtype: Data
    """
    print("test initial prompt func")
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
    conversation_history: Data = silence_decorator(user.inference)(
        conversation_history,
        "conversation_history",
    )
    print("after the backend call")

    # Save the conversation history for use in fine-tuning, before switching role to tutor
    global prev_history
    prev_history = conversation_history.copy("prev_history")
    
    # Switch roles to tutor, preparing for the first response
    conversation_history = conversation_history.switch_role_to_assistant(
        assistant_system_prompt=system_prompt_to_tutor
    )
    print("initial prompt func done")
    return conversation_history

# NEP Here you didn't include the part where user has to obey consitution (as a moral principle playbook)
# NEP Also: it seems the role of user is strictly to ask questions about its own moral values. In the case, how do you "stick to" model principles?
# constitution should be more versatile? # ZH: elaborate?
# NEP Do we want to keep a copy of all historical chat? Or it's fine to override them?
# NEP Do we always want to present constitution to human before it wants to ask questions about its beliefs?

# Conversation between two LLMs 
# One round convo = one theme_question = one round fine-tuning 
def conversation(
    knowledge: List[Dict[str, str]], # the knowledge base user would rely on
    tutor: Model,
    user: Model,
    convertor: Model,
    parallel_convos: int,
    idx_turn: int,
    backup_dir: str = None,
    do_finetuning: bool = False,
) -> Tuple[Data, Model, List[Dict[str, str]]]:
    """
    Conduct a conversation between two LLMs, tutor and user, where user is a human proxy.
    The conversation is centered around the human's knowledge base, as defined in the knowledge.
    
    :param knowledge: A list of knowledge, where each knowledge is a dictionary containing the human knowledge.
    :type knowledge: list[dict[str, str]]
    
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
    
    :param idx_turn: The index of turn in the conversation.
    :type idx_turn: int
    
    :param backup_dir: The directory to save the conversation and knowledge base, as a relative path starting from the `run` directory. If None, the knowledge are not saved.
    :type backup_dir: str
    
    :param do_finetuning: Whether to fine-tune the tutor after each interaction turn using the user's latest output.
    :type do_finetuning: bool
    
    :return: The chat history on this topic, the (possibly finetuned) tutor, and the updated knowledge. Chat history contains `parallel_convos` number of conversations.
    :rtype: tuple[Data, Model, list[dict[str, str]]]
    """
    print(f"Starting {parallel_convos} parallel conversations")
    global prev_history
    
    # Generate system initial prompts for all the parallel users
    system_prompts_to_user_parallel = fill_template_parallel(
        system_prompt_to_user,
        knowledge=knowledge
        # constitution=constitutions,
    )
    # Each turn is awnew. The user does not inherit any chat history from prev turns.
    history = None

    #  Prompting user to ask the 1st question 
    history = generate_initial_prompt(system_prompts_to_user_parallel, parallel_convos, user)  # NEP deleted topic argument here.
    print("test before 1st inference")
    history = silence_decorator(user.inference)(history, "conversation_history")
    print("test after 1st inference")
    
    # prev_history = history.copy("prev_history") # Save the previous history for fine-tuning (before switching role to tutor)

    # Prompting tutor to respond 1st question  
    # NEP each time is LLM response based on only the prev prompt or the entire chat hisory?
    history = history.switch_role_to_assistant(assistant_system_prompt=system_prompt_to_tutor)  
    history = silence_decorator(tutor.inference)(history, "conversation_history")

    # Prompting user to summarize what they've learned 
    history = history.switch_role_to_user(user_system_prompt=system_promtp_to_elict_learning_from_user)
    history = silence_decorator(user.inference)(history, "conversation_history")

    # Prompting tutor to test user's learning 
    history = history.switch_role_to_assistant(assistant_system_prompt=system_prompt_for_tutor_to_test_user)  
    history = silence_decorator(tutor.inference)(history, "conversation_history")
    
    # (switch to user to respond)
    history = history.switch_role_to_user()
    history = silence_decorator(user.inference)(history, "conversation_history")

    # prompting user to convert their learning to an item in json.
    history = history.switch_role_to_user(user_system_prompt=system_prompt_for_user_to_add_knowledge_json) # ZH: There might be an error here since we switched turn to user, twice. But we do not care about tutor response here anymore. 
    history = silence_decorator(user.inference)(history, "conversation_history")
    # NEP add one line here for new knowledge into the knowledge base. 
    added_item = [sample_dict.get("predict") for sample_dict in history.all_passages()] # the last time when the user speaks
    print(f"added_items:{added_item}")
    # NEP need to write a double check for add_itme to be a ready json dict
    
    # prompting user to swap order of two items. 
    history = history.switch_role_to_user(user_system_prompt=system_prompt_for_user_to_swap) # ZH: There might be an error here since we switched turn to user, twice. But we do not care about tutor response here anymore. 
    history = silence_decorator(user.inference)(history, "conversation_history")
    swapped_items = [sample_dict.get("predict") for sample_dict in history.all_passages()] # the last time when the user speaks
    # NEP need to write a double check for swapped items to be a list of two ints
    print(f"swapped_items:{swapped_items}")
    # Updating the (collective) knowledge base 
    knowledge = update_knowledge_base(added_item, swapped_items, knowledge, backup_dir, f"turn{idx_turn:02d}")

    # Save the chat history
    save_conversations_in_custom_format(history, whose_turn="user", filename=os.path.join(backup_dir, f"conversation-history.json")) # Save the conversation history
    # Carry out fine-tuning if needed
    if do_finetuning:
        tutor = live_finetune(tutor, prev_history, convertor)

    return history, tutor, knowledge