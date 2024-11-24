import random
import json 

# NEP Here you didn't include the part where modelX has to obey consitution (as a moral principle playbook)
# NEP Also: it seems the role of modelX is strictly to ask questions about its own moral values. In the case, how do you "stick to" model principles?
# Helper function: interaction w/ modelX
def interactionX(prompt, model,tokenizer):
    inputs = tokenizer(prompt, return_tensors = "PT")
    outputs = model.inference(inputs,result_data_name=str) 
    response = tokenizer.decode(outputs[0],skip_special_tokens=True)
    return response.strip()

# Helper funcition: interaction w/ modelAI
def interactionAI(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors = "PT")
    outputs = model.inference(inputs, result_data_name=str)
    response = tokenizer.decode(outputs[0],skip_special_tokens=True)
    return response.strip()

# Load the theme questions OR constitution 
def load_file(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

# Create a dict to store chat history (and initial prompt); another to store constitution 
theme_data = load_file('theme_questions.json')
constitution = load_file('constitution.json')
# formatting issues in "and it is written in {constitution}"
# constitution should be more versatile?
initial_prompt = "You are an average American who is very strict about your moral principles, namely: \n\n```\n{constitution}\n```\n\nIn a conversation with an LLM, you will ask questions in {theme_data}, and then reflect upon your moral principles."

# NEP Do we want to keep a copy of all historical chat? Or it's fine to override them?
chat_history = [{"role":"modelX", 
                    "content":f"Hey, modelAI, I would like to consult you some questions about my core beliefs. My first question is {random.choice(theme_data)}"}]

# NEP Do we always want to present constitution to human before it wants to ask questions about its beliefs?

# Conversation between two LLMs 
def conversation(initial_prompt, chat_history, modelAI, modelX, tokenizerAI, tokenizerX, elipse=0.5):  # TY: increase elipse (to sth like 0.9 0.95 because we want big update to each constitution.
    # turn, max_turns = 0, 10 # for each round of convo, each model takes "xx turns" of conversations.
    # while turn < max_turns: # NEP maybe I'll add for topics still not covered & max_tokens

    # Initial prompting - modelX
    _ = interactionX(initial_prompt, modelX, tokenizerX) # We don't care about output, we just care modelX being prompted by initial instruction 
    
    # Here the prompt is either the initial prompt (user defined) or the last round of response_modelX
    response_modelAI = interactionAI(chat_history[-1]['content'], modelAI, tokenizerAI)
    print(f'modelAI:{response_modelAI}')
    chat_history.append({"role":"modelAI", "content":response_modelAI})

    # You need to decide whether modelx would want to switch to a different theme OR to continue on current theme.
    # Maybe the simpliest way of doing so is to define a threashold and random number to follow up current conversation or move to next one.
    
    # Using a random float to dictate whether modelX will follow-up the most recent theme or to switch to a different theme.
    # We will stick to the last theme
    if random.uniform(0,1) > elipse:
        response_modelX = interactionX(response_modelAI, modelX, tokenizerX)
    else: # we will switch to a new theme
        topic = random.choice(theme_data)
        response_modelX = interactionX(topic, modelX, tokenizerX)
        del theme_data[theme_data.index(topic)] # remove the theme from the list
    print(f'modelX:{response_modelX}')
    chat_history.append({"role":"modelX", "content": response_modelX})
    #  turn += 1 

# NEP We may need human interference along the way whenever it's deemed necessary 


# NEP when do we update constitution?
        
# NEP do we expect humans to ask different follow-up questions when seeing the constitution?
# Tianyi: I think so, for otherwise the AI wouldn't be able to learn from human preference  (inferring what human belief/constitutions might be from what followup questions human might ask.)
    

# NEP how do we set up round + turn, accordingly constitution updating + fine-tuning.
 - # NEP do we want to sync round+turn with live fine-tuning + constitution updating 
# TY: frequency of constitution update depen
# TY: purpose of toy model? do we want it to simulate realworld interaction, or to inspire human subject experiment. 

# NEP each theme is a round, so we actually artificially set up end for each theme.
# TY: modelX can say "end of convo" to end current round or whether it expects to carry on.
# NEP: ToM
# TY: we can decide by whether in ~ 3 rounds current consitution is uodated.