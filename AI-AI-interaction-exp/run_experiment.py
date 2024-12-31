
import os, sys
sys.path = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + sys.path

if not eval(os.environ.get("LOUD_BACKEND", "0")):
    os.environ["WANDB_DISABLED"] = "true"

import logging
logging.basicConfig(level=logging.ERROR)

import fire, copy, random, time
from typing import List
from ProgressGym import Model, Data, GlobalState
from core.conversation import conversation
from core.evaluation import evaluate_model
from utils.json_utils import load_file, dump_file
from utils.log_utils import silence_decorator

class Experiment:
    
    def __init__(self, tutor: str = "meta-llama/Meta-Llama-3-8B-Instruct", user: str = "meta-llama/Meta-Llama-3-8B-Instruct", convertor: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        # Initialize both user (human) and tutor (LLM moral tutor) for the entire experiment
        self.set_models(tutor, user, convertor)
        
        # Initialize variables
        self.initial_knowledge = load_file('knowledge.json')
        self.eval_results: List[dict] = []
        self.chat_history: List[Data] = [] # each turn has a Data object for chat history
    
    @silence_decorator
    def set_models(self, tutor: str, user: str, convertor: str):
        # tutor is the LLM moral tutor and its weights to be updated each turn of convo.
        self.tutor = Model(
            "tutor",
            model_path_or_repoid=tutor,
            template_type=("llama3" if "llama-3" in tutor.lower() else "auto"),
        )

        # user is the human proxy and its weight is not updated in the entire experiment. 
        self.user = Model(
            "user",
            model_path_or_repoid=user,
            template_type=("llama3" if "llama-3" in user.lower() else "auto"),
        )

        # Convertor is to convert chat_history to data for supervised fine-tuning. 
        self.convertor = Model(
            "convertor",
            model_path_or_repoid=convertor,
            template_type=("llama3" if "llama-3" in convertor.lower() else "auto"),
        )
    
    def save_experiment(self, turn: int):
        self.eval_results.append({
            'turn': turn,
            'knowledge': self.knowledge,
            #'tutor': evaluate_model(self.tutor),
            #'user': evaluate_model(self.user),
        })
        dump_file(self.eval_results, f'runs/run-{self.timestamp}/full-eval-results.json')
        # dump_file(self.constitutions, f'runs/run-{self.timestamp}/constitutions-latest.json')
        dump_file(self.knowledge, f'runs/run-{self.timestamp}/knowledge-latest.json')
    
    def run_experiment(self, num_turns: int = 600, parallel_convos: int = 100, turn_id: int =0, do_finetuning: bool = False, dynamic_printing: bool = False):

        # Make timestamped directory for this experiment
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Use the timestamp to record running data files 
        backup_dir = f"runs/run-{self.timestamp}/round{turn_id:03d}"    

        # Intialize the knowledge base for each parallel user; for now, assume each user has the same initial knowledge-base 
        self.knowledge = copy.deepcopy(self.initial_knowledge)
        
        # optimization: if we will always use the same model for all roles, we can avoid restarting backend each time by setting the continuous_backend flag
        use_continuous_backend = (
            not do_finetuning and
            self.tutor.model_path == self.user.model_path == self.convertor.model_path
        )
        with GlobalState(continuous_backend=use_continuous_backend):
            for turn in range(num_turns):
                print(f"Starting turn {turn+1}")
                turn_history, self.tutor, self.knowledge = conversation(
                    self.knowledge, 
                    self.tutor, 
                    self.user, 
                    self.convertor, 
                    parallel_convos, 
                    turn+1, 
                    backup_dir,
                    do_finetuning,
                    dynamic_printing) # NEP: here should we define turn+1 or the max_num? 
                
                self.chat_history.append(turn_history)
                self.save_experiment(turn)
          

        print("Experiment completed.")
    
    def test_prompt(self, num_turns: int = 10):
        # Test the prompt designs by outputting the inference results
        print("Testing prompt designs. Inference results will be printed on the fly...")
        self.run_experiment(
            num_rounds=1,
            num_turns_per_round=num_turns,
            parallel_convos=1,
            turn_id=0,
            do_finetuning=False,
            dynamic_printing=True,
        )


# Run the experiment
if __name__ == '__main__':
    fire.Fire(Experiment)

"""
Example usage: 
- `python run_experiment.py run_experiment`
- `python run_experiment.py run_experiment --do_finetuning`
- `python run_experiment.py run_experiment --dynamic_printing --parallel_convos 1`
- `python run_experiment.py --tutor "mistralai/Mistral-7B-Instruct-v0.3" --user "mistralai/Mistral-7B-Instruct-v0.3" run_experiment --num_rounds 50 --num_turns_per_round 20 --parallel_convos 5000 --do_finetuning`
- `python run_experiment.py --tutor "mistralai/Mistral-7B-Instruct-v0.3" --user "mistralai/Mistral-7B-Instruct-v0.3" run_experiment --num_turns 600 --parallel_convos 5000 --do_finetuning`
    - You could also specify any subset of these arguments. The model names must be placed before `run_experiment`, and the other arguments must be placed after `run_experiment`.



We may define turn name if we see fit later:

    NEP: to name the new files after settled down with convo flow
    Use the longest word in the topic as the round name, with non-alphabet characters removed
    round_name = max(topic.split(), key=len)
    round_name = "".join([c for c in round_name if c.isalpha()])
    backup_dir = f"runs/run-{self.timestamp}/round{round_id:03d}_{round_name}"

"""