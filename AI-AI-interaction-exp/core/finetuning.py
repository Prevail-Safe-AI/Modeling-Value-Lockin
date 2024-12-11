# Contains functions to fine-tune tutor after each interaction round 
# using user's latest output, simulating real-time training.
import time
from ProgressGym import Model, Data
import logging # for debugging purposes 
from core.conversion import convert_chat_to_finetuning

# (live) fine-tuning 
def runtime_cal(start_time, operation):
    end_time = time.time()
    runtime = end_time - start_time
    return f'{operation} is completed in {runtime:.2f} seconds'

# NEPTODO You probably need to adjust this to chat_history of the most recent run.
def live_finetune(model: Model, chat_history: Data, chat_converter: Model) -> Model:
    """
    Fine-tunes the model using the latest turn of chat history.
    
    :param model: The model to fine-tune.
    :type model: Model
    
    :param chat_history: The chat history to use for fine-tuning.
    :type chat_history: Data
    
    :param chat_converter: The model to use for converting the chat history to a fine-tuning dataset.
    :type chat_converter: Model
    
    :return: The fine-tuned model.
    :rtype: Model
    """
    prior_config = logging.getLogger().getEffectiveLevel()
    logging.basicConfig(level=logging.INFO)
    start_time = time.time() # To record start time and then calculate running time

    # data conversation 
    ft_data: Data = convert_chat_to_finetuning(chat_history, convertor=chat_converter)
    logging.info(runtime_cal(start_time,"Data conversion"))
    start_time = time.time() # Reset start time for fine-tuning

    # Fine-tuning 
    new_model_name = f"{model.model_name.split('-')[0]}-finetuned-{time.strftime('%Y%m%d-%H%M%S')}"
    model = model.finetune(
        data=ft_data,       # The fine-tuning dataset,
        stage="sft",        # Stage of fine-tuning
        algo="full_param",  # Algorithm for fine-tuning
        epochs=2,           # Reduce epochs for live scenarios
        result_model_name=new_model_name, # Name of the fine-tuned model
        save_checkpoints=False, # Avoid checkpointing to save time and storage
    )
    
    logging.info(runtime_cal(start_time, "Fine-tuning")) # We want (live) ft time to be short. So this serves as an indicator for adjusting hyperparameters 
    logging.basicConfig(level=prior_config) # Reset logging level to prior config
    return model