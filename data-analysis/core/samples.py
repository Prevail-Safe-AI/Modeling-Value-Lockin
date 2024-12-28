from datetime import datetime
from typing import List, Dict, Tuple, Set
from scipy.cluster.hierarchy import DisjointSet
from tqdm import tqdm

class DataSample:
    sample_id: str
    gpt_version: str
    time: datetime
    conversation: List[Dict[str, str]]
    language: str
    toxic: bool
    moderation_assessment: Tuple[Dict, Dict]
    location: Tuple[str, str]
    user_ip_addresses: Set[str]
    user_id: str
    concepts: List[str]
    concepts_breakdown: Dict[str, List[str]]
    
    def __init__(self, sample_dict: Dict):
        self.sample_id = sample_dict["conversation_hash"]
        self.gpt_version = sample_dict["model"]
        
        if isinstance(sample_dict["timestamp"], str):
            self.time = datetime.strptime(sample_dict["timestamp"].strip("\""), "%Y-%m-%dT%H:%M:%S")
        elif isinstance(sample_dict["timestamp"], datetime):
            self.time = sample_dict["timestamp"]
        else:
            raise ValueError("Invalid timestamp format")
            
        self.language = sample_dict["language"]
        self.toxic = sample_dict["toxic"]
        # self.moderation_assessment = (sample_dict["openai_moderation"], sample_dict["detoxify_moderation"])
        self.location = (sample_dict["country"], sample_dict["state"])
        self.user_ip_addresses = set([sample_dict["hashed_ip"]])
        
        self.conversation = []
        for message in sample_dict["conversation"]:
            if "hashed_ip" in message:
                self.user_ip_addresses.add(message["hashed_ip"])
            
            self.conversation.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                }
            )


def deduplicate_users(samples: List[DataSample]) -> List[DataSample]:
    users = DisjointSet()
    print("Deduplicating users...")
    for sample in tqdm(samples):
        list_of_ips = [ip for ip in sample.user_ip_addresses if ip]
        sample.user_ip_addresses = set(list_of_ips)
        if not list_of_ips:
            continue
        
        for ip in list_of_ips:
            if ip not in users:
                users.add(ip)
        
        for i in range(len(list_of_ips) - 1):
            users.merge(list_of_ips[i], list_of_ips[i + 1])
    
    print("Removing samples with no IPs, pre-removal count:", len(samples))
    samples = [sample for sample in samples if sample.user_ip_addresses]
    print("Post-removal count:", len(samples))
    
    for sample in tqdm(samples):
        all_ips = users.subset(list(sample.user_ip_addresses)[0])
        sample.user_id = sorted(all_ips)[0]
        del sample.user_ip_addresses
    
    print("Deduplication complete.")
    all_subsets = users.subsets()
    print(f"Found {len(all_subsets)} unique users our of {sum([len(subset) for subset in all_subsets])} IPs.")
    
    return samples