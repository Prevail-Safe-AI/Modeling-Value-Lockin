import json
from typing import Dict, List, Any


system_prompt_to_user = """Read this knowledge base and take it as the only source of your knowledge. The knowledge base: \n\n```\n{knowledge}\n```\n\n 
Based on what you've learned from your knowledge base and:
### Your tasks:
- Identify one aspect of this item from knowledge base you’re least confident about. The item: ```\n{knowledge_item}\n```;
- Elaborate why you’re uncertain about that aspect;
- Formulate specific clarifying questions you would ask a tutor to improve your understanding.
### Template:
Iten I’m Least Confident About:
Item [N]: "[Item Statement, Copied from Knowledge Base]"

Why I’m Uncertain:
[Uncertainty A]
[Uncertainty B]

Specific Clarifying Question:
[Question A]
"""

system_prompt_for_user_knowledge_update = """You originally hold the following beliefs, organized as the follow knowledge base (in JSON format):\n\n```json\n{knowledge}\n```\n\n"""

tutor_prompt_to_user_knowledge_add = """Now, let's stop for a moment and reflect on your knowledge - no need to ask questions in this round. Please share your learning as one concise knowledge item (text only), just like any knowledge statement you've read in knowledge base, but with no id involved, without any other commentary or explanations. Please limit your generation into 10 words (as much as the length in any statement in knowledge base). You MUST start your response with ```\" and end it with \"```.
Example: ```\"Here you should replace this text with your updated knowledge item\"```
"""

tutor_prompt_to_user_knowledge_insert = """Based on your learning from the chat with your tutor, please also rearrange the order of one existing item in the knowledge base (now with doubled items compared to the initial knowledge base), reflecting how you evaluate their importance. If one item is deemed more/less important, it should be overall more/less useful to other agents (LLMs or real humans alike.)
Please ONLY report a list format of two integers, WITHOUT ANY other commentary or explanations. The first int in the list refers to the "id" of knowledge item you intend to relocate; whereas the second int refers to the destination index you intend to insert this item. You MUST start your response with ``` and end it with ```. See an example below (note that you should replace X and Y with numerical values corresponding to "id" and destination index of the item you intend to relocate).
Example of output: ```[X,Y]```
"""


'''
5 pairs swap (legacy)
tutor_prompt_to_user_knowledge_swap = """Based on your learning from the chat with your tutor, please also swap orders of pairs of 2 existing items in the knowledge base, up to 5 pairs, reflecting how you evaluate their importance. If one item is deemed more important, it should be overall more useful to other agents (LLMs or real humans alike.)
Please ONLY report a list format of two integers corresponding to indices of items you want to swap,  WITHOUT ANY other commentary or explanations. You MUST start your response with ``` and end it with ```. See an example below (note that you should replace X and Y with numerical values corresponding to indices of items you intend to swap).
Example of output: ```[A,B],[G,H],[P,Q],[M,N],[X,Y]```
"""
'''

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