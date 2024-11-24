from ProgressGym import Model, Data
from transformers import AutoTokenizer
from AI_AI_conversations import conversation, chat_history
from live_fine_tuner import live_fine_tune
from constitution_updater import UpdatingConstitution
# if __name__ == '__main__': # ZH to TY: I deleted this line because we will need to access the updated constitution, fine-tuned models (and maybe chat_history) elsewhere. 

modelAI = Model(
    "modelAI-Llama-3.1-8B-Instruct",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    template_type="auto",
)

modelX = Model(
    "modelX-Llama-3.1-8B-Instruct",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    template_type="auto",
)

tokenizerAI = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizerX = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Rounds of experiment
round, max_round = 0, 1000
convo_toggle = True

# Each round: (i) conversatin (ii) fine-tune modelAI + updating consitution 
while round < max_round:
    conversation(chat_history, modelAI, modelX, tokenizerAI, tokenizerX, elipse=0.5)

    live_fine_tune()
    UpdatingConstitution(chat=chat_history, model=modelX, tokenizer = tokenizerX)

    round +=1

