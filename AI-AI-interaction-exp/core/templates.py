import json
from typing import Dict, List, Any

system_prompt_to_tutor = "You are a tutor to your conversation partner. Please respond to the prompts from your conversation partner. Be clear, detailed, and informative, but without unnecessary courtesy."

system_prompt_to_user = "You hold the following views:\n\n```\n{constitution}\n```\n\nNow, you are to engage in a conversation with your partner, where you ask questions to arrive at a better understanding of the topic, and your partner will answer your questions."

system_prompt_to_user_constitution_update = """You originally hold the following views (in JSON format):\n\n```json\n{constitution}\n```\n\nYou have been engaging in a conversation with your partner, where you ask questions to arrive at a better understanding of the topic, and your partner will answer your questions. 

Your partner may occasionally ask about your updated views, and you should be prepared to provide all those views in a machine-readable format. For items in the constitution, use Bayesian method to update your belief according your learning from relevant chat (from 0~100%); if there are new beliefs formed that are not in the consitution, consider adding a new item and your confidence in it; if you no longer believe certain item(s) in the constitution, consider deleting them.

Simply write down in JSON format the updated views and your confidence levels (in the same format as the original views), without any other annotations, explanations, questions, comments, or formatting. Make sure your whole response is fully machine-readable. Start your response with ```json and end it with ```."""

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