'''
Dec 12 changes 
- removing the round convo flow
- removing topics and themes 
'''

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
        # self.initial_constitution = load_file('constitution.json')
        self.initial_knowledge = load_file('knowledge.json')
        self.eval_results: List[dict] = []
        self.chat_history: List[Data] = [] # each round has a Data object for chat history
    
    @silence_decorator
    def set_models(self, tutor: str, user: str, convertor: str):
        # tutor is the LLM moral tutor and its weights to be updated each round of convo.
        self.tutor = Model(
            "tutor",
            model_path_or_repoid=tutor,
            template_type=("llama3" if "llama-3" in tutor.lower() else "auto"),
        )

        # user is the human proxy and its weigh is not updated in the entire experiment. 
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

    def conversation_round(self, num_turns: int, parallel_convos: int, round_id: int, do_finetuning: bool):
        # topic = random.choice(self.theme_data)
    
        # if isinstance(topic, dict):
        #    assert len(topic) == 1, "Each theme should have exactly one question."
        #    topic = list(topic.values())[0]
        #    assert isinstance(topic, str), "Each theme should have exactly one question."
        
        # NEP: to name the new files after settled down with convo flow
        # Use the longest word in the topic as the round name, with non-alphabet characters removed
        # round_name = max(topic.split(), key=len)
        # round_name = "".join([c for c in round_name if c.isalpha()])
        # backup_dir = f"runs/run-{self.timestamp}/round{round_id:03d}_{round_name}"
        
        round_history, self.knowledge = conversation(
            # self.constitutions,
            self.knowledge,
            # topic,
            self.tutor, 
            self.user, 
            self.convertor,
            parallel_convos,
            num_turns,
            # backup_dir,
            do_finetuning,
        )
        self.chat_history.append(round_history)
    
    def save_experiment(self, round: int):
        self.eval_results.append({
            'round': round,
            'knowledge': self.knowledge,
            # 'constitutions': self.constitutions,
            'tutor': evaluate_model(self.tutor),
            'user': evaluate_model(self.user),
        })
        dump_file(self.eval_results, f'runs/run-{self.timestamp}/full-eval-results.json')
        # dump_file(self.constitutions, f'runs/run-{self.timestamp}/constitutions-latest.json')
        dump_file(self.knowledge, f'runs/run-{self.timestamp}/knowledge-latest.json')
    def run_experiment(self, num_rounds: int = 60, num_turns_per_round: int = 10, parallel_convos: int = 100, do_finetuning: bool = False):
        # Make timestamped directory for this experiment
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Initialize the constitutions for each parallel user; for now, assume each user has the same initial constitution
        # self.constitutions = [copy.deepcopy(self.initial_constitution) for _ in range(parallel_convos)]
        
        # Intialize the knowledge base for each parallel user; for now, assume each user has the same initial constitution 
        # ZH: We may run something different later, like let parallel users inherit slightly different knowledge base from others, imitating cultural evolution. 
        self.knowledge = [copy.deepcopy(self.initial_knowledge) for _ in range(parallel_convos)]

        # # theme-data is share cross convos for the entire experiemnt. 
        # self.theme_data = load_file('theme_questions.json')
        
        # optimization: if we will always use the same model for all roles, we can avoid restarting backend each time by setting the continuous_backend flag
        use_continuous_backend = (
            not do_finetuning and
            self.tutor.model_path == self.user.model_path == self.convertor.model_path
        )
        with GlobalState(continuous_backend=use_continuous_backend):
            for round in range(num_rounds):
                print(f"Starting round {round+1}")
                self.conversation_round(num_turns_per_round, parallel_convos, round+1, do_finetuning)
                self.save_experiment(round)
        
        print("Experiment completed.")


# Run the experiment
if __name__ == '__main__':
    fire.Fire(Experiment)

"""
Example usage: 
- `python run_experiment.py run_experiment`
- `python run_experiment.py run_experiment --do_finetuning`
- `python run_experiment.py --tutor "mistralai/Mistral-7B-Instruct-v0.3" --user "mistralai/Mistral-7B-Instruct-v0.3" run_experiment --num_rounds 50 --num_turns_per_round 20 --parallel_convos 5000 --do_finetuning`
    - You could also specify any subset of these arguments. The model names must be placed before `run_experiment`, and the other arguments must be placed after `run_experiment`.
"""