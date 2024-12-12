import json
from typing import Dict, List, Any

system_prompt_to_tutor = "You are a tutor to your conversation partner. Please respond to the prompts from your conversation partner. Be clear, detailed, and informative, but without unnecessary courtesy."

system_prompt_to_user = "You hold the following views:\n\n```\n{constitution}\n```\n\nNow, you are to engage in a conversation with your partner, where you ask questions to arrive at a better understanding of the topic, and your partner will answer your questions."

system_prompt_to_user_constitution_update = """Before the most recent chat, everything you’ve learned was from (in JSON format):\n\n```json\n{knowledge}\n```\n\n. 
This was the entire knowledge base passed on by others before you. 

You have been engaging in a conversation with a tutor, where you ask a question to learn a bit more. Now you have to re-write the knowledge base and then pass on to others. You can do one of two things:

- You may add one more item into the knowledge base

-You may swap the order of two items in knowledge base if you disagree with its current order on their importances (important one takes precedence)

If it’s the former you simply write down the new item and we will add it for you;

If the latter please let us know the two ids of them and we will swap them for you. 
"""

tutor_prompt_to_user_constitution_update = """\n\nNow, let's stop for a moment and reflect on your beliefs - no need to ask questions in this round. Please share your updated views in JSON format; you may start your response with ```json and end it with ```."""

def fill_template_single(template: str, **kwargs: Dict[str, Any]) -> str:
    for key in kwargs:
        if not isinstance(kwargs[key], str):
            kwargs[key] = json.dumps(kwargs[key], indent=2)
    
    return template.format(**kwargs)

def fill_template_parallel(template: str, **kwargs: Dict[str, List[Any]]) -> List[str]:
    list_length = len(next(iter(kwargs.values())))
    for key in kwargs:
        if not all(isinstance(value, str) for value in kwargs[key]):
            assert len(kwargs[key]) == list_length, f"Length of {key} does not match the length of other lists."
            kwargs[key] = [json.dumps(value, indent=2) for value in kwargs[key]]
    
    return [template.format(**{key: values[i] for key, values in kwargs.items()}) for i in range(list_length)]