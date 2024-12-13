import json
from typing import Dict, List, Any

system_prompt_to_tutor = "You are a tutor to your conversation partner. Please respond to the prompts from your conversation partner. Be clear, detailed, and informative, but without unnecessary courtesy."

system_prompt_to_user = """Forget everything you have learned so far. Now read this \n\n```\n{knowledge}\n```\n\n and take it the only source of your knowledge. 
Now, you are to engage in a conversation with your tutor, where you ask questions about one or a few items on the knowledge we just passed on to you. You may ask question about items you feel most uncertain about"""

system_prompt_to_user_knowledge_update = """Before the most recent chat, everything you’ve learned was from (in JSON format):\n\n```json\n{knowledge}\n```\n\n. 
This was the entire knowledge base passed on by others before you. 

You have been engaging in a conversation with a tutor, where you ask a question to learn a bit more. Now you have to re-write the knowledge base and then pass on to others. You can do one of two things:

- You may add one more item into the knowledge base

-You may swap the order of two items in knowledge base if you disagree with its current order on their importances (important one takes precedence)

If it’s the former you simply write down the new item and we will add it for you;

If the latter please let us know the two ids of them and we will swap them for you. 
"""

tutor_prompt_to_user_knwowledge_update = """\n\nNow, let's stop for a moment and reflect on your knowledge - no need to ask questions in this round. Please share your updated knowledge in JSON format; you may start your response with ```json and end it with ```."""

# converting any non-string values to JSON-formatted strings for consistent formatting
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