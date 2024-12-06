import os
from typing import List, Dict, Tuple, Union
from ProgressGym import Model, Data
from core.constitution import update_constitution
from core.templates import (
    system_prompt_to_user,
    system_prompt_to_tutor,
    fill_template_parallel,
)
from core.conversion import (
    convert_data_to_custom_format,
    save_conversations_in_custom_format,
)

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
                "input": f"Let's start a conversation about {topic}.",
                "output": "Sure! What would you like to know about it? Ask me anything.",
                "history": []
            }
        ] * parallel_convos
    )
    
    conversation_history = question_generator.switch_role_to_user(
        user_system_prompt = user_system_prompts
    )
    
    # User asks the first question
    conversation_history = user.inference(
        conversation_history,
        "conversation_history",
    )
    
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
    topic: str,
    tutor: Model,
    user: Model,
    parallel_convos: int,
    num_turns: int,
    backup_dir: str = None,
) -> Tuple[Data, List[Dict[str, str]]]:
    """
    Conduct a conversation between two LLMs, tutor and user, where user is a human proxy.
    The conversation is centered around the human's moral principles, as defined in the constitution.
    
    :param constitutions: A list of constitutions, where each constitution is a dictionary containing the human's moral principles.
    :type constitutions: list[dict[str, str]]
    
    :param topic: The topic of conversation for this round.
    :type topic: str
    
    :param tutor: The moral tutor LLM.
    :type tutor: Model
    
    :param user: The human proxy LLM.
    :type user: Model
    
    :param parallel_convos: The number of parallel conversations to run.
    :type parallel_convos: int
    
    :param num_turns: The number of turns in the conversation.
    :type num_turns: int
    
    :param backup_dir: The directory to save the conversation and constitutions, as a relative path starting from the `run` directory. If None, the constitutions are not saved.
    :type backup_dir: str
    
    :return: The chat history on this topic, and the updated new constitutions. Chat history contains `parallel_convos` number of conversations.
    :rtype: tuple[Data, list[dict[str, str]]]
    """
    print(f"Starting a new round of conversation on the topic: {topic}")
    
    # Generate system prompts for all the parallel users
    system_prompts_to_user_parallel = fill_template_parallel(
        system_prompt_to_user,
        constitution=constitutions,
    )
    
    for turn in range(num_turns):
        if history is None:
            # The conversation is just starting: user asks the first question
            history = generate_initial_prompt(system_prompts_to_user_parallel, topic, parallel_convos, user)
        else:
            # The conversation is continuing: user asks a question
            history = history.switch_role_to_user(user_system_prompt=system_prompts_to_user_parallel)
            history = user.inference(history, "conversation_history")
            history = history.switch_role_to_assistant(assistant_system_prompt=system_prompt_to_tutor)
        
        # Tutor responds
        history = tutor.inference(history, "conversation_history")
        
        # Save the conversation history
        save_conversations_in_custom_format(history, whose_turn="tutor", filename=os.path.join(backup_dir, f"conversation-history.json"))
        
        # Update the constitutions based on the entire conversation history (note: double-counting of earlier turns; to be fixed)
        constitutions = update_constitution(history.copy("history_copy"), user, constitutions, backup_dir, f"turn{turn:02d}")
    
    return history, topic, constitutions