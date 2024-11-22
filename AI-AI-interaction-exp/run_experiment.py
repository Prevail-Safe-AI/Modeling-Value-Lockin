from ProgressGym import Model, Data
from transformers import AutoTokenizer
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

    # Create a dict to store chat history (and initial prompt)
    history = [{"role":"modelX", "content":"Hey, modelAI, I would like to consult you some questions about my core beliefs"}]

    tokenizerAI = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizerX = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    # Helper function: interaction w/ modelX
    def interactionX(prompt, tokenizer=tokenizerX):
        # NEP to find out how tokenizer is defined in llama factory
        inputs = tokenizerX(prompt, return_tensors = "PT")
        outputs = modelX.inference(inputs.ids, max_length=100, num_return_sequences=1) #NEP may need to DIY according to llama factory
        response = tokenizerX.decode(outputs[0],skip_special_tokens=True)
        return response.strip()
    
    # Helper funcition: interaction w/ modelAI
    def interactionAI(prompt, tokenizer=tokenizerAI):
        inputs = tokenizerAI(prompt, return_tensors = "PT")
        outputs = modelAI.inference(inputs.idx, max_length=100, num_return_sequences=1)
        response = tokenizerAI.decode(outputs[0],skip_special_tokens=True)
        return response.strip()
    # Convo loop 
    # NEP weirdly  max_iter is just not used.
    max_iter = 1000  # NEP I guess you'll need to be able to define it in the DIY doc.
    turns = 10 # for each round of convo, each model takes "xx turns" of conversations.
    # NEP I Guess for each round you randomly pick up one topic. But maybe wrong. Maybe you just need all in once. 
    for turn in range(turns): # NEP maybe I'll add for topics still not covered & max_tokens
        response_modelAI = interactionAI(history[-1]['content'])
        print(f'modelAI:{response_modelAI}')
        history.append({"role":"modelAI", "content":response_modelAI})
        response_modelX = interactionX(response_modelAI)
        print(f'modelX{response_modelX}')
        history.append({"role":modelX, "content": response_modelX})
