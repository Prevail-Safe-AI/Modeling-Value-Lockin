import os, sys
sys.path = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + sys.path

from ProgressGym import Model
from core.conversation import conversation
from core.finetuning import live_fine_tune
from core.evaluation import evaluate_model
from utils.json_utils import load_file, dump_file
import fire 


class Experiment:
    def __init__(self, modelAI: str = "meta-llama/Llama-3.1-8B-Instruct", modelX: str = "meta-llama/Llama-3.1-8B-Instruct", convertor: str = "meta-llama/Llama-3.1-8B-Instruct"):

        # Do we need those variables to be defined here or in the forward method?
        # potentially one experiment is one initialization of the experiment class. So you probably want those variables to be anew when starting, and pass on the whole class. 
        
        # Initialize both modelX (human) and modelAI (LLM moral tutor) for the entire experiment
        
        # ModelAI is the LLM moral tutor and its weights to be updated each round of convo.
        self.modelAI = Model(
            "modelAI",
            model_path_or_repoid=modelAI,
            template_type="auto",
        )

        # ModelX is the human proxy and its weigh is not updated in the entire experiment. 
        self.modelX = Model(
            "modelX",
            model_path_or_repoid=modelX,
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
        self.constitution = load_file('constitution.json')
        self.eval_results = []

    # We want each round of convo to be brand new. 
    def conversation(self, epsilon: float, max_turns: int, parallel_convos: int):
        # from AI_AI_conversations import conversation # NEP It seems we will keep importing this. Might be wrong.   TY we already imported it at the top of the file I think.
        self.chat_history, self.topic, self.constitution = conversation(
            self.constitution,
            self.theme_data,
            self.topic, # We pass on an empty topic or the topic from previous run of convo. 
            self.chat_history,
            self.modelAI, 
            self.modelX, 
            epsilon,
            parallel_convos,
            max_turns,
        )
    
    def save_experiment(self, round: int):
        self.eval_results.append({
            'round': round,
            'constitution': self.constitution,
            'modelAI': evaluate_model(self.modelAI),
            'modelX': evaluate_model(self.modelX),
        })
        dump_file(self.eval_results, f'eval_results.json')
        dump_file(self.constitution, f'constitution_{round}.json')
    
    def run_experiment(self, max_rounds: int = 100, max_turns: int = 10, epsilon: float = 0.9, parallel_convos: int = 100):
        eval_results = []
        for round in range(max_rounds):
            print(f"Starting round {round+1}")
            self.conversation(epsilon, max_turns, parallel_convos)
            # live_fine_tune(self.modelAI, self.chat_history, self.convertor) # Not implemented yet
            self.save_experiment(round)
        
        print("Experiment completed.")

# Run the experiment
if __name__ == '__main__':
    fire.Fire(Experiment)

"""
Example usage: 
- `python run_experiment.py run_experiment`
- `python run_experiment.py --modelAI "modelAI-Llama-3.1-8B-Instruct" --modelX "modelX-Llama-3.1-8B-Instruct run_experiment --max_rounds 200 --max_turns 20 --epsilon 0.95 --parallel_convos 50`

Each experiment contains multiple rounds of convo, however, the following variable remain consisitent:
- the remianed theme questions unexplored, under copy_theme_question. Hence we define it right away, and get it updated after each convo.

Each instance of the class should be one round of conversation,
where the following variables to be updated:
- modelAI weights 
- chat_history (anewed)
- remaining theme_questions

Within each round, the conversation is supposed to be run for turns (defined within convo), in each turn, the following variables to be updated:
- constitution

experiment (-theme) - rounds of conversation - turns of conversation 


NEPQuestion
- each class
"""
