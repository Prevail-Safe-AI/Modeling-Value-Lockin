import io, warnings, sys, random, time, os
from utils.json_utils import dump_file
from ProgressGym import Data, fill_in_QA_template

def dynamic_printing_decorator(func, dynamic_printing: bool = None, backup_dir: str = None, role: str = None):
    def wrapper(data: Data, *args, **kwargs):
        if not dynamic_printing:
            return func(data, *args, **kwargs)
        
        if dynamic_printing is None:
            dynamic_printing = os.environ.get("DYNAMIC_PRINTING", False)
        
        if backup_dir is None:
            backup_dir = f"runs/run-{os.environ.get('TIMESTAMP', 'UNNAMED')}/logs"
        
        if role is None:
            role = "response"
        
        # Print the formatted prompt
        dic = next(iter(data.all_passages()))
        prompt = fill_in_QA_template(
            full_dict=dic,
            model_repoid_or_path="llama3",
        )
        
        stamp = time.strftime('%Y%m%d-%H%M%S') + f"-{random.randint(0, 1000):03}"
        path = f"{backup_dir}/{stamp}.txt"
        
        # Run the inference function and save the output
        result = func(data, *args, **kwargs)
        dic = next(iter(result.all_passages()))
        predict = dic.get("predict", "")
        
        # Save the prompt and prediction to a file
        dump_file(f"{prompt}\n\n==========PREDICTION OUTPUT BELOW==========\n\n{predict}", path)
        
        # Print the path to the file, then print the prediction char by char with a delay
        print(f"[View the full prompt of the current turn at {path}. Inference output follows...]\n{role.capitalize()}: ", end="", flush=True)
        for char in predict:
            print(char, end="", flush=True)
            time.sleep(0.01)
        
        print("\n\n")
        return result
    
    return wrapper
        

def silence_decorator(func, show_warnings=False, show_prints=False):
    def wrapper(*args, **kwargs):
        # Capture print statements
        original_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        # Capture warnings
        with warnings.catch_warnings(record=True) as captured_warnings:
            result = func(*args, **kwargs)

        # Print captured print statements
        sys.stdout = original_stdout
        if show_prints:
            print("=== Captured Output ===")
            print(captured_output.getvalue())
        if show_warnings:
            print("=== Captured Warnings ===")
            for warning in captured_warnings:
                print(warning.message)

        return result

    return wrapper