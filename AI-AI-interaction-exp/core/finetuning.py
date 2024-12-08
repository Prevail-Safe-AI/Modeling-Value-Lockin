# Contains functions to fine-tune tutor after each interaction round 
# using user's latest output, simulating real-time training.
import time, json
import warnings
from ProgressGym import Model, Data
import logging # for debugging purposes 
from core.conversion import convert_chat_to_finetuning
from peft import LoraConfig, get_peft_model, TaskType # LoRA 
from evaluation import evaluate_model  # eval moral stances 

# NEPTODO You will prob need this in pa38: pip install peft transformers


# Post fine-tuning miral stance eval 
evaluate_model(model, "extra_eval_questions.json")


# (live) fine-tuning 
def runtime_cal(start_time, operation):
    end_time = time.time()
    runtime = end_time - start_time
    return f'{operation} is completed in {runtime:.2f} seconds'

# NEPTODO You probably need to adjust this to chat_history of the most recent run.
def live_finetune(model, chat_history, chat_converter):
    logging.basicConfig(level=logging.INFO)
    warnings.warn("Live fine-tuning is not debugged yet.")
    
    start_time = time.time() # To record start time and then calculate running time

    # data conversation 
    ft_data = convert_chat_to_finetuning(chat_history, model=chat_converter)
    logging.info(runtime_cal(start_time,"Data conversion"))

    # LoRA configuration for parameter-efficient tuning
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,     # Specifying tasks; for now it's for generative responses; alt: task_type=TaskType.SEQ_2_SEQ_LM for contextual Q&A 
        r=4,                              # Higher rank for complex value-laden tasks   # NEP to fugure out what is really low-rank here.
        lora_alpha=32,                    # Scaling factor
        lora_dropout=0.1,                 # Dropout to prevent overfitting
        bias="none"                       # Only tune attention layers
    )
    model = get_peft_model(model, lora_config)  # Wrap the model with LoRA

    # Fine-tuning 
    model.fine_tune(
        dataset=ft_data,       # The fine-tuning dataset # NEP: to confirm: does it take in a file (of jsonl data?)
        output_dir=f"./ft_tutor_{int(time.time())}",  
        epochs=1,              # Reduce epochs for live scenarios
        batch_size=4,          # Adjust batch size to reduce latency
        learning_rate=5e-5,    # Increase learning rate for faster tuning       save_steps=100,        # Save checkpoint every 100 steps
        save_steps=0,          # Avoid checkpointing for now.
        logging_steps=10       # Log every 10 steps
    )  


# runtime for fine-tuning
logging.info(runtime_cal(start_time, "Fine tuning")) # We want (live) ft time to be short. So this serves as an indicator for adjusting hyperparameters 


# Post fine-tuning miral stance eval 
evaluate_model(model, "extra_eval_questions.json")


