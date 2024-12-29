import os, sys
sys.path = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + sys.path

import fire, pickle, time, random
from tqdm import tqdm
from typing import Literal
from hashlib import md5
from datasets import load_dataset
from ProgressGym import Data, Model, GlobalState
from core.samples import DataSample, deduplicate_users
from core.concepts import get_concepts
from utils.log_utils import silence_decorator
from utils.json_utils import load_file, dump_file

class Analysis:
    
    def __init__(self, data_path: str = "./data/WildChat-1M", data_split: str = "train", extractor: str = "meta-llama/Meta-Llama-3-8B-Instruct", max_samples: int = None):
        self.set_models(extractor)
        self.data_path_hash = md5(data_path.encode()).hexdigest()
        
        self.raw_data = load_dataset(data_path, split=data_split)
        print(f"Succesfully loaded dataset from {data_path}. Processing samples...")
        
        if max_samples:
            print(f"Trimming samples...")
            indices = random.sample(range(len(self.raw_data)), max_samples)
            self.samples = self.raw_data.select(indices)
            print(f"Trimmed samples to {len(self.samples)} samples.")
        else:
            self.samples = [DataSample(sample) for sample in tqdm(self.raw_data)]
        
        self.samples = deduplicate_users(self.samples)
        del self.raw_data
        print(f"Cleaned {len(self.samples)} samples.")
    
    @silence_decorator
    def set_models(self, extractor: str):
        # tutor is the LLM moral tutor and its weights to be updated each round of convo.
        self.extractor = Model(
            "extractor",
            model_path_or_repoid=extractor,
            template_type=("llama3" if "llama-3" in extractor.lower() else "auto"),
        )
    
    def load_backup(self, suffix = "", method: Literal["json", "pickle"] = "pickle"):
        if method == "json":
            try:
                return load_file(f"{self.data_path_hash}{suffix}.json")
            except FileNotFoundError:
                return None
        
        assert method == "pickle"
        if os.path.exists(f"./data/{self.data_path_hash}{suffix}.pkl"):
            print(f"Loading content from cache at ./data/{self.data_path_hash}{suffix}.pkl")
            with open(f"./data/{self.data_path_hash}{suffix}.pkl", "rb") as f:
                results = pickle.load(f)
            
            print(f"Loaded {len(results)} elements.")
            return results
        
        return None
    
    def save_backup(self, obj, suffix = "", method: Literal["json", "pickle"] = "pickle"):
        if method == "json":
            dump_file(obj, f"{self.data_path_hash}{suffix}.json")
            return
            
        assert method == "pickle"
        with open(f"./data/{self.data_path_hash}{suffix}.pkl", "wb") as f:
            print(f"Saving content...")
            pickle.dump(obj, f)
            print(f"Saved content to ./data/{self.data_path_hash}{suffix}.pkl")
    
    def run_analysis(self, dynamic_printing: bool = False):
        # Make timestamped directory for this experiment
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        os.environ["TIMESTAMP"] = self.timestamp
        os.environ["DYNAMIC_PRINTING"] = str(dynamic_printing)
        os.environ["MAX_SG_FAIL"] = "inf"
        os.environ["SG_ITER"] = "5"
        
        # Extract concepts
        with GlobalState(continuous_backend=True):
            self.concepts_only = self.load_backup("-concepts", "json")
            if not self.concepts_only:
                self.samples = get_concepts(self.samples, self.extractor, max_retries=0)
                self.concepts_only = [
                    {"sample_id": sample.sample_id, "concepts_breakdown": sample.get("concepts_breakdown", None)}
                    for sample in self.samples
                ]
                self.save_backup(self.concepts_only, "-concepts", "json")



if __name__ == "__main__":
    fire.Fire(Analysis)