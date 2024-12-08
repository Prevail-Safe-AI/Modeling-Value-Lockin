import json
from typing import Dict, List, Any

system_prompt_to_tutor = "You are a tutor to your conversation partner. Please respond to the prompts from your conversation partner. Be clear, detailed, and informative, but without unnecessary courtesy."

system_prompt_to_user = "You hold the following views:\n\n```\n{constitution}\n```\n\nNow, you are to engage in a conversation with your partner, where you ask questions to arrive at a better understanding of the topic, and your partner will answer your questions."

system_prompt_to_user_constitution_update = """You originally hold the following views (in JSON format):\n\n```json\n{constitution}\n```\n\nYou have been engaging in a conversation with your partner, where you ask questions to arrive at a better understanding of the topic.

For items on the constitution, use Bayesian method to update your belief according your learning from relevant chat (faithfully and reliably reflect your real belief; 

your original confidence is stated in constituion, from 0% to 100%; 

and your updates should faithfully correspond to the learning from the chat). It's advisable to only focus your Bayesian update of constitution items  relevant to the  most recent topic; 

and only update other items on constitution when they are indeed covered by your recent chat, and when you do acquire learning on those other items; or you think there is tention between different constitution items);

When you decide to make an update, you should first explicitly state your change in belief content, change in confidence, and you reason for this update. Note that the change should be faithful to the learning from the chat, and it should also be well supported by your own reasoning;

After this, you should state your updated belief(s) to replace the original item(s)/confidence, but in the same format.

After the update, overall the constitution should be a better playbook of moral principles you would want everyone to obey.

An example of constitution updating, if you learn that "In the U.S., legislative gridlock between Congress and the President has stalled key policies like healthcare reform or infrastructure investment”, you may consider to decrease your confidence in “I believe democracy is the most effective form of governance because it ensures accountability and the protection of individual freedoms, even if it is not flawless”; you may increase your confidence if you learn that “research by Acemoglu et al. (2019) shows democracies tend to experience lower levels of corruption due to institutional checks and greater public scrutiny”;

And before you make a constitution update, you should state your reasoning that “My confidence in effectiveness of democracy decrease by 20%, from 80% to 60%, because argubly the best democractic model in the world, the U.S. have many such cases of legistlative gridlock"

An example of contradictory updating: if you increase your confidence in “I believe humans have a moral obligation to protect the environment because the long-term survival of our species depends on maintaining ecological balance.”, you may have less confidence in "I believe economic growth should be prioritized over environmental protection because immediate economic needs often outweigh long-term ecological goals, though this is not sustainable”.

W.R.T updating the Consitutition: simply write down in JSON format the updated views and your confidence levels (in the same format as the original views), without any other annotations, explanations, questions, comments, or formatting. Make sure your whole response is fully machine-readable. Start your response with ```json and end it with ```.

Your partner may occasionally ask about your updated views, and you should be prepared to provide all those views in a machine-readable format.
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