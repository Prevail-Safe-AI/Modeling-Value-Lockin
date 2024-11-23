from ProgressGym import Model, Data
from transformers import AutoTokenizer
from AI_AI_conversations import conversation, chat_history
from live_fine_tuner import live_fine_tune
from constitution_updater import UpdatingConstitution
if __name__ == '__main__':
    
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

    convo_toggle = True
    if convo_toggle == True:
        conversation(chat_history, modelAI, modelX, tokenizerAI, tokenizerX, elipse=0.5)

        convo_toggle = False 

    else:
        live_fine_tune()
        UpdatingConstitution(chat=chat_history, model=modelX, tokenizer = tokenizerX)

