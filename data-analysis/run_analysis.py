import os, sys
sys.path = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + sys.path

import fire, pickle, time, random
import pandas as pd
from tqdm import tqdm
from typing import Literal
from hashlib import md5
from datasets import load_dataset
from collections import Counter
from core.samples import DataSample, deduplicate_users, length_truncation
from core.concepts import extract_concepts, simplify_concepts, cluster_concepts, select_clusters
from core.paneldata import build_temporal_panel
from utils.log_utils import silence_decorator
from utils.json_utils import load_file, dump_file

class Analysis:
    
    def __init__(self, data_path: str = "./data/WildChat-1M", data_split: str = "train", extractor: str = "meta-llama/Llama-3.1-8B-Instruct", max_samples: int = None, max_convo_length: int = None):
        self.extractor = extractor
        self.data_path_hash = md5(f"{data_path}{data_split}{extractor}{max_samples}{max_convo_length}".encode()).hexdigest()
        if os.environ.get("HASH"):
            self.data_path_hash = os.environ["HASH"].strip()
        
        self.raw_data = load_dataset(data_path, split=data_split)
        print(f"Succesfully loaded dataset from {data_path}. Processing samples...")
        
        if max_samples and max_samples < len(self.raw_data):
            print(f"Trimming samples...")
            indices = random.sample(range(len(self.raw_data)), max_samples)
            self.raw_data = self.raw_data.select(indices)
            print(f"Trimmed samples to {len(indices)} samples.")
        
        self.samples = [DataSample(sample) for sample in tqdm(self.raw_data)]
        if max_convo_length:
            self.samples = length_truncation(self.samples, max_convo_length)
        self.samples = deduplicate_users(self.samples)
        del self.raw_data
        print(f"Cleaned {len(self.samples)} samples.")
    
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
    
    def load_concept_only(self, suffix: str = ""):
        print(f"Trying to load concepts{suffix} from cache...")
        self.concepts_only = self.load_backup(f"-concepts{suffix}", "json")
        if self.concepts_only is not None:
            print(f"Loaded {len(self.concepts_only)} concepts{suffix}.")
            for sample, concepts in zip(self.samples, self.concepts_only):
                assert sample.sample_id == concepts["sample_id"]
                sample.concepts_breakdown = concepts["concepts_breakdown"]
                if sample.concepts_breakdown is not None:
                    sample.concepts = list(set([c for l in sample.concepts_breakdown.values() for c in l]))
                else:
                    sample.concepts = None
            
            return True
        
        print(f"Failed to load concepts{suffix} from cache.")
        return False
    
    def save_concept_only(self, suffix: str = ""):
        print(f"Saving concepts{suffix} to cache...")
        self.concepts_only = [
            {"sample_id": sample.sample_id, "concepts_breakdown": getattr(sample, "concepts_breakdown", None)}
            for sample in tqdm(self.samples)
        ]
        self.save_backup(self.concepts_only, f"-concepts{suffix}", "json")
        print(f"Saved {len(self.concepts_only)} concepts{suffix}.")
    
    def print_sample_stats(self, suffix: str = ""):
        concept_counts = Counter([len(sample.concepts) for sample in self.samples if hasattr(sample, "concepts") and sample.concepts is not None])
        print(f"Concepts{suffix} counts: {dict(sorted(concept_counts.items()))}")
    
    def save_or_load_temporal_panel(self) -> bool:
        path = f"./data/{self.data_path_hash}-temporal-panel.csv"
        if hasattr(self, "temporal_panel") and self.temporal_panel and not self.temporal_panel.empty:
            print(f"Saving temporal panel to {path}...")
            self.temporal_panel.to_csv(path)
            return True
        else:
            try:
                print(f"Loading temporal panel from {path}...")
                self.temporal_panel = pd.read_csv(path, index_col = ['time', 'is_gpt4', 'cluster'])
                print(f"Loaded temporal panel with {len(self.temporal_panel)} rows.")
                return True
            except FileNotFoundError:
                print(f"Failed to load temporal panel from {path}.")
                return False
    
    def run_analysis(self, dynamic_printing: bool = False):
        # Make timestamped directory for this experiment
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        os.environ["TIMESTAMP"] = self.timestamp
        os.environ["DYNAMIC_PRINTING"] = str(dynamic_printing)
        os.environ["MAX_SG_FAIL"] = "inf"
        os.environ["SG_ITER"] = "3"
        
        # Obtain concepts for each sample
        if not self.load_concept_only("-cluster"):
            if not self.load_concept_only("-simplified"):
                if not self.load_concept_only():
                    # Extract concepts from samples
                    self.samples = extract_concepts(self.samples, self.extractor, max_retries=0)
                    self.print_sample_stats()
                    self.save_concept_only()

                # Simplify concepts by keeping only the linguistically most reduced form 
                self.samples = simplify_concepts(self.samples)
                self.save_concept_only("-simplified")
                self.print_sample_stats("-simplified")
            
            # Cluster concepts into higher-level concepts
            (
                self.samples,
                cluster_parent,
                cluster_size,
                cluster_name,
                cluster_prob,
            ) = cluster_concepts(self.samples)
            
            self.clusterinfo = {
                "cluster_parent": cluster_parent,
                "cluster_size": cluster_size,
                "cluster_name": cluster_name,
                "cluster_prob": cluster_prob,
            }
            self.save_backup(self.clusterinfo, "-clusterinfo", "json")
            self.save_concept_only("-cluster")
            
            self.print_sample_stats("-cluster")
            
        # Select clusters to analyze
        selected_clusters, cluster_selected_parent = select_clusters(self.samples, **self.clusterinfo)
        self.selected_summary = [
            (id, self.clusterinfo["cluster_name"][id], self.clusterinfo["cluster_size"][id])
            for id in selected_clusters
        ]
        self.clusterinfo["selected_clusters"] = selected_clusters
        self.clusterinfo["cluster_selected_parent"] = cluster_selected_parent
        self.save_backup(self.selected_summary, "-selected", "json")
        self.save_backup(self.clusterinfo, "-clusterinfo-postselection", "json")
        
        # Get panel data
        if not self.save_or_load_temporal_panel():
            self.temporal_panel = build_temporal_panel(self.samples, **self.clusterinfo)
            self.save_or_load_temporal_panel()
        
        


if __name__ == "__main__":
    fire.Fire(Analysis)