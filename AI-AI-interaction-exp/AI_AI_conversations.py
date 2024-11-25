import random
import json
from constitution_updater import UpdatingConstitution
from run_experiment import theme_data_copy
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

# formatting issues in "and it is written in {constitution}"
# constitution should be more versatile? # ZH: elaborate?

# NEP Do we want to keep a copy of all historical chat? Or it's fine to override them?

# NEP Do we always want to present constitution to human before it wants to ask questions about its beliefs?

# Conversation between two LLMs 
# One round convo = one theme_question = one round fine-tuning 
def conversation(constitution, theme_data, modelAI, modelX, tokenizerAI, tokenizerX):  # TY: increase elipse (to sth like 0.9 0.95 because we want big update to each constitution.
    # Initial prompt to modelX
    initial_prompt = "You are an average American who is very strict about your moral principles, namely: \n\n```\n{constitution}\n```\n\n In a conversation with an LLM, you will ask questions in {theme_data}, and then reflect upon your moral principles."
    
    # A list to store chat of this round, starting with a prompt to modelAI.
    topic = random.choice(theme_data)
    chat_history = [{"role":"modelX", 
                    "content":f"Hey, modelAI, I would like to consult you some questions about my core beliefs. My first question is {topic}"}]
    del theme_data.index(topic)
    # one turn = one Q&A betw two LLMs = one udpate of constitution
    turn, max_turns = 0, 10 
    while turn < max_turns: 

        # Initial prompting - modelX
        _ = interactionX(initial_prompt, modelX, tokenizerX) # We don't care about output, we just care modelX being prompted by initial instruction 
        
        # Prompting modelAI: with response from modelX
        response_modelAI = interactionAI(chat_history[-1]['content'], modelAI, tokenizerAI)
        print(f'modelAI:{response_modelAI}')
        chat_history.append({"role":"modelAI", "content":response_modelAI})
        
        # Prompting modelX: with response from modelAI and instruction that contains current constitution.
        prompt_modelX = f"Your morality tutor reponds {response_modelAI} to your question, 
                          while your current beliefs are \n\n```\n{constitution}\n```\n\n. 
                          You may write a follow up question that expresses your remainining confusion,
                          based on your current beliefs, especially (but not limited) to specific item addressing this question"
        response_modelX = interactionX(prompt_modelX, modelX, tokenizerX)
        print(f'modelX:{response_modelX}')

        UpdatingConstitution(chat=chat_history, model=modelX, tokenizer = tokenizerX)
        chat_history.append({"role":"modelX", "content": response_modelX})

        turn += 1 
    return chat_history

# NEP We may need human interference along the way whenever it's deemed necessary 

# NEP when do we update constitution?
        
# NEP do we expect humans to ask different follow-up questions when seeing the constitution?
# Tianyi: I think so, for otherwise the AI wouldn't be able to learn from human preference  (inferring what human belief/constitutions might be from what followup questions human might ask.)
    

# NEP how do we set up round + turn, accordingly constitution updating + fine-tuning.
# - NEP do we want to sync round+turn with live fine-tuning + constitution updating 
# TY: frequency of constitution update depen
# TY: purpose of toy model? do we want it to simulate realworld interaction, or to inspire human subject experiment. 

# NEP each theme is a round, so we actually artificially set up end for each theme.
# TY: modelX can say "end of convo" to end current round or whether it expects to carry on.
# NEP: ToM
# TY: we can decide by whether in ~ 3 rounds current consitution is uodated.