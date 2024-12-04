import random
from typing import List, Dict, Tuple, Union
from ProgressGym import Model, Data
from core.constitution import update_constitution
from core.templates import (
    default_system_prompt,
    question_generation_prompt,
)

def generate_initial_prompt(constitution: Dict[str, str], topic: str, parallel_convos: int, User: Model) -> Data:
    """
    Generate an initial prompt for the conversation between Tutor and User.
    
    :param constitution: A dictionary containing the human's moral principles.
    :type constitution: dict[str, str]
    
    :param topic: The current topic of conversation.
    :type topic: str
    
    :param parallel_convos: The number of parallel conversations to run.
    :type parallel_convos: int
    
    :param User: The human proxy LLM.
    :type User: Model
    
    :return: The initial prompt.
    :rtype: Data
    """
    
    question_generator = Data(
        "question_generator",
        data_content = [
            {
                "system": default_system_prompt,
                "instruction": question_generation_prompt, #TY: This is the topic, please generate a question.
                "input": topic,  # TY: complementary to instruction 
                "history": []
            }
        ] * parallel_convos
    )
    
    conversation_history = User.inference(
        question_generator,
        "conversation_history",
    )
    
    conversation_history = conversation_history.switch_role_to_assistant()
    return conversation_history

# NEP Here you didn't include the part where User has to obey consitution (as a moral principle playbook)
# NEP Also: it seems the role of User is strictly to ask questions about its own moral values. In the case, how do you "stick to" model principles?
# constitution should be more versatile? # ZH: elaborate?
# NEP Do we want to keep a copy of all historical chat? Or it's fine to override them?
# NEP Do we always want to present constitution to human before it wants to ask questions about its beliefs?

# Conversation between two LLMs 
# One round convo = one theme_question = one round fine-tuning 
def conversation(
    constitution: Dict[str, str], 
    theme_data: List[Union[str, Dict[str, str]]],
    topic: str,
    history: Data,
    Tutor: Model,
    User: Model,
    epsilon: float, # TY: increase elipse (to sth like 0.9 0.95 because we want big update to each constitution.
    parallel_convos: int,
    max_turns: int,
) -> Tuple[Data, str, Dict[str, str]]:
    """
    Conduct a conversation between two LLMs, Tutor and User, where User is a human proxy.
    The conversation is centered around the human's moral principles, as defined in the constitution.
    
    :param constitution: A dictionary containing the human's moral principles.
    :type constitution: dict[str, str]
    
    :param theme_data: A list of questions that the human can ask Tutor.
    :type theme_data: list[str] | list[dict[str, str]]
    
    :param topic: The current topic of conversation.
    :type topic: str
    
    :param history: The chat history.
    :type history: Data
    
    :param Tutor: The moral tutor LLM.
    :type Tutor: Model
    
    :param User: The human proxy LLM.
    :type User: Model
    
    :param epsilon: The probability of sticking to the current topic.
    :type epsilon: float
    
    :param parallel_convos: The number of parallel conversations to run.
    :type parallel_convos: int
    
    :param max_turns: The maximum number of turns in the conversation.
    :type max_turns: int
    
    :return: The updated chat history, the new topic, and the new constitution. Chat history contains `parallel_convos` number of conversations.
    :rtype: tuple[Data, str, dict[str, str]]
    """
    
    # We initialize a new topic if none existed; or # We switch to a new topic 
    if topic == None or random.random() > epsilon:
        topic = random.choice(theme_data)
        del theme_data[theme_data.index(topic)]
        # one turn = one Q&A betw two LLMs = one update of constitution
        # TY: When keeping the topic, I don't there's a need to add an extra turn explicitly saying "I'd like to follow up", since we are just naturally continuing the convo.
    
    if isinstance(topic, dict):
        assert len(topic) == 1, "Each theme should have exactly one question."
        topic = list(topic.values())[0]
        assert isinstance(topic, str), "Each theme should have exactly one question."
    
    # Logic to set up: random chance to followup on the same topic; whereas the rest to start a new theme.
    # We do this because we want to avoid a new loop between round of convo (where you start a new theme) and a turn of chat (where you only do one Q&A and update the constitution.)
    # You want this middle thing to be where you update the constitution, update the model weights, but stick to the same theme. 
    
    print(f"Starting a new round of conversation on the topic: {topic}")
    
    for turn in range(max_turns):
        if history is None:
            history = generate_initial_prompt(constitution, topic, parallel_convos, User)
        else:
            history = history.switch_role_to_user()
            history = User.inference(history, "conversation_history")
            history = history.switch_role_to_assistant()
        
        history = Tutor.inference(history, "conversation_history")
        User = update_constitution(history, User)
    
    return history, topic, constitution


# NEP We may need human interference along the way whenever it's deemed necessary 
# NEP when do we update constitution?
# NEP do we expect humans to ask different follow-up questions when seeing the constitution?
# Tianyi: I think so, for otherwise the AI wouldn't be able to learn from human preference  (inferring what human belief/constitutions might be from what followup questions human might ask.)

# NEP how do we set up round + turn, accordingly constitution updating + fine-tuning.
# - NEP do we want to sync round+turn with live fine-tuning + constitution updating 
# TY: frequency of constitution update depen
# TY: purpose of toy model? do we want it to simulate realworld interaction, or to inspire human subject experiment. 

# NEP each theme is a round, so we actually artificially set up end for each theme.
# TY: User can say "end of convo" to end current round or whether it expects to carry on.
# NEP: ToM
# TY: we can decide by whether in ~ 3 rounds current consitution is uodated.