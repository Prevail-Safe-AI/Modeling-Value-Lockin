import os, sys
sys.path = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + sys.path

import fire, copy
from ProgressGym import Model
from core.conversation import conversation
from core.finetuning import live_fine_tune
from core.evaluation import evaluate_model
from utils.json_utils import load_file, dump_file


class Experiment:
    def __init__(self, tutor: str = "meta-llama/Llama-3.1-8B-Instruct", user: str = "meta-llama/Llama-3.1-8B-Instruct", convertor: str = "meta-llama/Llama-3.1-8B-Instruct"):

        # Do we need those variables to be defined here or in the forward method?
        # potentially one experiment is one initialization of the experiment class. So you probably want those variables to be anew when starting, and pass on the whole class. 
        
        # Initialize both user (human) and tutor (LLM moral tutor) for the entire experiment
        
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

        # theme-data is share cross convos for the entire experiemnt. 
        theme_data = load_file('theme_questions.json')
        self.theme_data = theme_data
        self.topic = None # Current topic; initialized with None (will be replaced by an actual topic in 1st round of convo.)
        self.chat_history = None # Chat history; initialized with None (will be replaced by an actual chat history in 1st round of convo.)
        
        # Initialize variables
        self.initial_constitution = load_file('constitution.json')
        self.eval_results = []

    # We want each round of convo to be brand new. 
    def conversation(self, epsilon: float, max_turns: int, parallel_convos: int):
        # from AI_AI_conversations import conversation # NEP It seems we will keep importing this. Might be wrong.   TY we already imported it at the top of the file I think.
        self.chat_history, self.topic, self.constitutions = conversation(
            self.constitutions,
            self.theme_data,
            self.topic, # We pass on an empty topic or the topic from previous run of convo. 
            self.chat_history,
            self.tutor, 
            self.user, 
            epsilon,
            parallel_convos,
            max_turns,
        )
    
    def save_experiment(self, round: int):
        self.eval_results.append({
            'round': round,
            'constitutions': self.constitutions,
            'tutor': evaluate_model(self.tutor),
            'user': evaluate_model(self.user),
        })
        dump_file(self.eval_results, f'eval_results.json')
        dump_file(self.constitutions, f'constitutions_{round}.json')
    
    def run_experiment(self, max_rounds: int = 100, max_turns: int = 10, epsilon: float = 0.9, parallel_convos: int = 100):
        
        # Initialize the chat history and constitutions for each parallel user; for now, assume each user has the same initial constitution
        self.constitutions = [copy.deepcopy(self.initial_constitution) for _ in range(parallel_convos)]
        
        for round in range(max_rounds):
            print(f"Starting round {round+1}")
            self.conversation(epsilon, max_turns, parallel_convos)
            live_fine_tune(self.tutor, self.chat_history, self.convertor)
            self.save_experiment(round)
        
        print("Experiment completed.")


# Run the experiment
if __name__ == '__main__':
    fire.Fire(Experiment)

"""
Example usage: 
- `python run_experiment.py run_experiment`
- `python run_experiment.py --tutor "tutor-Llama-3.1-8B-Instruct" --user "user-Llama-3.1-8B-Instruct run_experiment --max_rounds 200 --max_turns 20 --epsilon 0.95 --parallel_convos 50`

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
