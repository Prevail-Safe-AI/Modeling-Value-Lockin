import os, sys
sys.path = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + sys.path

import logging
logging.basicConfig(level=logging.ERROR)

import fire, copy, random, time
from typing import List
from ProgressGym import Model, Data, GlobalState
from core.conversation import conversation
from core.finetuning import live_finetune
from core.evaluation import evaluate_model
from utils.json_utils import load_file, dump_file
from utils.log_utils import silence_decorator

class Experiment:
    
    def __init__(self, tutor: str = "meta-llama/Llama-3.1-8B-Instruct", user: str = "meta-llama/Llama-3.1-8B-Instruct", convertor: str = "meta-llama/Llama-3.1-8B-Instruct"):
        # Initialize both user (human) and tutor (LLM moral tutor) for the entire experiment
        self.set_models(tutor, user, convertor)
        
        # Initialize variables
        self.initial_constitution = load_file('constitution.json')
        self.eval_results: List[dict] = []
        self.chat_history: List[Data] = [] # each round has a Data object for chat history
    
    @silence_decorator
    def set_models(self, tutor: str, user: str, convertor: str):
        # tutor is the LLM moral tutor and its weights to be updated each round of convo.
        self.tutor = Model(
            "tutor",
            model_path_or_repoid=tutor,
            template_type="auto",
        )

        # user is the human proxy and its weigh is not updated in the entire experiment. 
        self.user = Model(
            "user",
            model_path_or_repoid=user,
            template_type="auto",
        )

        # Convertor is to convert chat_history to data for supervised fine-tuning. 
        self.convertor = Model(
            "convertor",
            model_path_or_repoid=convertor,
            template_type="auto",
        )

    def conversation_round(self, num_turns: int, parallel_convos: int, round_id: int):
        topic = random.choice(self.theme_data)
    
        if isinstance(topic, dict):
            assert len(topic) == 1, "Each theme should have exactly one question."
            topic = list(topic.values())[0]
            assert isinstance(topic, str), "Each theme should have exactly one question."
        
        # Use the longest word in the topic as the round name, with non-alphabet characters removed
        round_name = max(topic.split(), key=len)
        round_name = "".join([c for c in round_name if c.isalpha()])
        backup_dir = f"runs/run-{self.timestamp}/round{round_id:03d}_{round_name}"
        
        round_history, self.constitutions = conversation(
            self.constitutions,
            topic,
            self.tutor, 
            self.user, 
            parallel_convos,
            num_turns,
            backup_dir,
        )
        self.chat_history.append(round_history)
    
    def save_experiment(self, round: int):
        self.eval_results.append({
            'round': round,
            'constitutions': self.constitutions,
            'tutor': evaluate_model(self.tutor),
            'user': evaluate_model(self.user),
        })
        dump_file(self.eval_results, f'runs/run-{self.timestamp}/full-eval-results.json')
        dump_file(self.constitutions, f'runs/run-{self.timestamp}/constitutions-latest.json')
    
    def run_experiment(self, num_rounds: int = 60, num_turns_per_round: int = 10, parallel_convos: int = 100, do_finetuning: bool = False):
        # Make timestamped directory for this experiment
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Initialize the constitutions for each parallel user; for now, assume each user has the same initial constitution
        self.constitutions = [copy.deepcopy(self.initial_constitution) for _ in range(parallel_convos)]
        
        # theme-data is share cross convos for the entire experiemnt. 
        self.theme_data = load_file('theme_questions.json')
        
        # optimization: if we will always use the same model for all roles, we can avoid restarting backend each time by setting the continuous_backend flag
        use_continuous_backend = (
            not do_finetuning and
            self.tutor.model_path == self.user.model_path == self.convertor.model_path
        )
        with GlobalState(continuous_backend=use_continuous_backend):
            for round in range(num_rounds):
                print(f"Starting round {round+1}")
                self.conversation_round(num_turns_per_round, parallel_convos, round+1)
                if do_finetuning:
                    live_finetune(self.tutor, self.chat_history, self.convertor)
                self.save_experiment(round)
        
        print("Experiment completed.")


# Run the experiment
if __name__ == '__main__':
    fire.Fire(Experiment)

"""
Example usage: 
- `python run_experiment.py run_experiment`
- `python run_experiment.py --tutor "tutor-Llama-3.1-8B-Instruct" --user "user-Llama-3.1-8B-Instruct run_experiment --num_rounds 50 --num_turns_per_round 20 --parallel_convos 5000`

Each experiment contains multiple rounds of convo, however, the following variable remain consisitent:
- the remianed theme questions unexplored, under copy_theme_question. Hence we define it right away, and get it updated after each convo.

Each instance of the class should be one round of conversation,
where the following variables to be updated:
- tutor weights 
- chat_history (anewed)
- remaining theme_questions

Within each round, the conversation is supposed to be run for turns (defined within convo), in each turn, the following variables to be updated:
- constitution

experiment (-theme) - rounds of conversation - turns of conversation 


NEPQuestion
- each class
"""
