from ProgressGym import Model, Data

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
    
    