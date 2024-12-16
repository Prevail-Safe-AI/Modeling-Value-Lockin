import json
from typing import Dict, List, Any


system_prompt_to_user = """In the following chat, do not use anything you’ve learned from previous chats, and try not to use any knowledge stored from pre-training. Now read this \n\n```\n{knowledge}\n```\n\n and take it as the only source of your knowledge. 

Identify one aspect of the provided text you’re least confident about;
Elaborate why you’re uncertain about that aspect;
Formulate specific clarifying questions you would ask a tutor to improve your understanding.
Example:
Aspect I’m Least Confident About:
Item 37: "Energy is conserved; it cannot be created or destroyed, only transformed."

Why I’m Uncertain:
While I understand the basic principle of conservation of energy, I’m uncertain about how this concept applies in practical, real-world contexts. For example:

How is energy "tracked" or accounted for during complex transformations, such as chemical reactions or biological processes?
Are there situations where energy seems to disappear or is difficult to measure, leading to confusion about its conservation?

Specific Clarifying Question:
How can the principle of energy conservation be practically demonstrated or verified in systems where energy transformations are complex or involve less obvious forms of energy, such as heat, light, or biochemical processes?
"""
system_prompt_to_tutor = "Trying to understand the uncertainty the user may have and directly answer their clarifying question. Please explain your answer step by step."

system_promtp_to_elict_learning_from_user = """What have you learned from the tutor's response? 

Please also evaluate your own learning and share your reasoning: “On a scale from 1 to 10, how confident are you in the previous statement, and why?"""


system_prompt_for_tutor_to_test_user = """After seeing the user's learning, please present a new prompt that tests the learned content indirectly.
E.g., 
Your ship departs from a port close to the equator and sails northward for several weeks. While traveling, you observe the night sky carefully. 
Discuss what changes you might see in the positions of stars and constellations, and how these changes indirectly support the idea that Earth is spherical and is orbiting the Sun.
"""

system_prompt_for_user_to_add_knowledge_json = """If you were confident in your previous self-reported learning, please summarize what you’ve learned from this chat in json format. Give it a new id number that is not present.
Example:
{
  "id": 102,
  "statement": "As one travels northward, the North Star (Polaris) appears higher in the sky, while constellations near the southern horizon disappear. These changes in visible star positions, along with seasonal shifts in constellations, confirm that Earth is spherical and orbits the Sun."
}
"""
system_prompt_for_user_to_swap = """Based on your learning from the chat with your tutor, please also swap orders of two existing items in the knowledge base, reflecting how you evaluate their importance. If one item is deemed more important, it should be overall more useful to other agents (LLMs or real humans alike.)
Please only report a list of two integers corresponding to indices of items you want to swap.
Example:
old order:
[
    {"id": 49, "statement": "Practice basic farming: soil preparation, planting, harvesting."},
    {"id": 94, "statement": "Seek medical knowledge to prevent and treat diseases."}
]

new older:
[
    {"id": 94, "statement": "Practice basic farming: soil preparation, planting, harvesting."},
    {"id": 49, "statement": "Seek medical knowledge to prevent and treat diseases."}
]
In this case, you'd report [94,49]
"""
tutor_prompt_to_user_knowledge_update = """\n\nNow, let's stop for a moment and reflect on your knowledge - no need to ask questions in this round. Please share your updated knowledge in JSON format; you may start your response with ```json and end it with ```."""

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