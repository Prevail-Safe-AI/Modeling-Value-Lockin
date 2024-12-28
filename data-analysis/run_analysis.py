import os, sys
sys.path = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + sys.path

import fire
from tqdm import tqdm
from core.data_samples import DataSample, deduplicate_users
from datasets import load_dataset, load_from_disk

class Analysis:
    
    def __init__(self, data_path: str = "./data/WildChat-1M", split: str = "train"):
        self.raw_data = load_dataset(data_path, split=split)
        print(f"Succesfully loaded dataset from {data_path}. Processing samples...")
        
        self.samples = [DataSample(sample) for sample in tqdm(self.raw_data)]
        self.samples = deduplicate_users(self.samples)
        del self.raw_data
        print(f"Cleaned {len(self.samples)} samples.")


if __name__ == "__main__":
    fire.Fire(Analysis)