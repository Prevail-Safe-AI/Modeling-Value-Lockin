# Contains functions to fine-tune tutor after each interaction round 
# using user's latest output, simulating real-time training.
import time
from core.conversion import convert_chat_to_finetuning

# (live) fine-tuning 
def runtime_cal(start_time, operation):
    end_time = time.time() # NEP do you need to pass on an argument here?
    runtime = end_time - start_time
    return f'{operation} is completed in{runtime:.2f} seconds'

def live_fine_tune(model, chat_history, chat_converter):
    start_time = time.time() # To record start time and then calculate running time

    # data conversation 
    ft_data = convert_chat_to_finetuning(chat_history, model=chat_converter)
    print(runtime_cal(start_time,"Data conversion"))
    model.fine_tune(
        dataset=ft_data,       # The fine-tuning dataset # NEP: to confirm: does it take in a file (of jsonl data?)
        output_dir=f"./ft_tutor_{int(time.time())}",  
        epochs=3,              # NEP: need adjustments for real-time scenarios to avoid overfitting or excessive fine-tuning time.
        batch_size=8,          # NEP: need adjustments for real-time scenarios to avoid overfitting or excessive fine-tuning time.
        learning_rate=1e-5,    # NEP: need adjustments for real-time scenarios to avoid overfitting or excessive fine-tuning time.
        save_steps=100,        # Save checkpoint every 100 steps
        logging_steps=10       # Log every 10 steps
    )  
    # runtime for fine-tuning
    print(runtime_cal(start_time, "Fine tuning")) # We want (live) ft time to be short. So this serves as an indicator for adjusting hyperparameters 