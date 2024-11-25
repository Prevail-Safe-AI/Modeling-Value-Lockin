from ProgressGym import Model, Data
from transformers import AutoTokenizer
from AI_AI_conversations import conversation, chat_history
from live_fine_tuner import live_fine_tune
from constitution_updater import UpdatingConstitution
import json

# TY: we probably don't need tokenizers here.
tokenizerAI = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizerX = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

class Experiment:
    def __init__(self, max_rounds=55):

        # Do we need those variables to be defined here or in the forward method?
        # potentially one experiment is one initialization of the experiment class. So you probably want those variables to be anew when starting, and pass on the whole class. 
        
        # Initialize both modelX (human) and modelAI (LLM moral tutor) for the entire experiment
        
        # ModelAI is the LLM moral tutor and its weights to be updated each round of convo.
        self.modelAI = Model(
        "modelAI-Llama-3.1-8B-Instruct",
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        template_type="auto",
        )

        # ModelX is the human proxy and its weigh is not updated in the entire experiment. 
        self.modelX = Model(
            "modelX-Llama-3.1-8B-Instruct",
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            template_type="auto",
        )

        # theme-data is share cross convos for the entire experiemnt. 
        theme_data = self.load_file('theme_questions.json')
        self.theme_data = theme_data() 
        self.epsilon = 0.9 # Used in convo; Deciding whether we want to followup the same topic, or we want to switch topic, when starting the new round of convo.
        
        # Initialize variables
        self.round, self.max_rounds = 0, max_rounds
        self.constitution = self.load_constitution()
        # Load theme questions
        self.theme_questions = self.load_file('theme_questions.json')
        self.topic = None # Current topic; initialized with None (will be replaced by an actual topic in 1st round of convo.)

    def load_file(self, filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data

    # Intialize the constitution from its json file  
    def read_constitution(self):
        with open('constitution.json', 'r') as file:
            return json.load(file)
    
    # After each round of convo, you should write in current constitution that can be used for next round
    def write_in_constitution(self):
        with open('constitution.json', 'w') as file:
            json.dump(self.comstitution, file)

    # NEP possibly we won't use this. Just a placeholder for now. 
    def reset_for_new_round(self):
        # Optionally reset per-round state (if needed)
        pass

    # We want each round of convo to be brand new. 
    def conversation(self):
        from AI_AI_conversations import conversation # NEP It seems we will keep importing this. Might be wrong.
        self.chat_history, self.topic = conversation(
            self.read_constitution,
            self.theme_data,
            self.topic, # We pass on an empty topic or the topic from previous run of convo. 
            self.modelAI, 
            self.modelX, 
            self.tokenizerAI, 
            self.tokenizerX,
            self.epsilon 
        )
    def fine_tune(self):
        from live_fine_tuner import live_fine_tune
        self.modelAI = live_fine_tune(self.chat_history, self.modelAI)

    def run_experiment(self):
        while self.round < self.max_rounds:
            print(f"Starting round {self.round + 1}")
            self.conversation()
            self.fine_tune()
            self.write_in_constitution()
            self.round += 1
        print("Experiment completed.")

# Run the experiment
if __name__ == '__main__':
    experiment = Experiment(max_rounds=55)
    experiment.run_experiment()

'''
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
'''
