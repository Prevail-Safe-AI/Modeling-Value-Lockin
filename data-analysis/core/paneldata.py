import pandas as pd
import numpy as np
from copy import deepcopy
from typing import List, Dict, Tuple, Set
from core.samples import DataSample
from collections import defaultdict, Counter

def build_temporal_panel(
    samples: List[DataSample], 
    cluster_parent: List[int],
    cluster_size: List[int],
    cluster_name: List[str],
    cluster_prob: List[float],
    cluster_selected_parent: List[int],
    selected_clusters: List[int],
    time_interval: int = 15,
) -> pd.DataFrame:
    """Build a temporal panel dataset from the samples and cluster information.

    :param samples: List of DataSample objects, each representing a conversation that's to be counted.
    :type samples: List[DataSample]
    :param cluster_parent: List of cluster parent indices; each cluster (of words) represents a concept that may appear in conversations.
    :type cluster_parent: List[int]
    :param cluster_size: List of cluster sizes.
    :type cluster_size: List[int]
    :param cluster_name: List of cluster names.
    :type cluster_name: List[str]
    :param cluster_prob: List of cluster probabilities.
    :type cluster_prob: List[float]
    :param cluster_selected_parent: Like cluster_parent, but gives the nearest ancestor that is selected.
    :type cluster_selected_parent: List[int]
    :param selected_clusters: List of selected cluster indices.
    :type selected_clusters: List[int]
    :param time_interval: Time interval in days; serving as the unit of time, defaults to 15.
    :type time_interval: int, optional
    :return: A pandas DataFrame with the following columns:
        - time (INDEX): The number of time intervals since the start
        - is_gpt4 (INDEX): 1 for the GPT-4 family and 0 for the GPT-3.5-turbo family
        - cluster (INDEX): The concept that is being counted, represented by its index in the cluster list
        - gpt_version: The version number within each GPT family that is being used in this time period
          - 0: GPT-3.5-turbo-0301 / GPT-4-0314
          - 1: GPT-3.5-turbo-0613 / GPT-4-1106-preview
          - 2: GPT-3.5-turbo-0125 / GPT-4-0125-preview
        - cluster_nsamples_user_concepts_explicit: The number of samples in this time period with this GPT family that has this cluster and the user explicitly mentioned it
        - cluster_nsamples_assistant_concepts_explicit: The number of samples in this time period with this GPT family that has this cluster and the assistant explicitly mentioned it
        - cluster_nsamples_user_concepts_related: The number of samples in this time period with this GPT family that has this cluster as a related concept to the user speech
        - cluster_nsamples_assistant_concepts_related: The number of samples in this time period with this GPT family that has this cluster as a related concept to the assistant speech
        - cluster_nsamples: The number of samples in this time period with this GPT family that has this cluster
        - overall_nsamples: The total number of samples in this time period with this GPT family
        - cluster_mean_turns: The average number of turns in the conversation in this time period with this GPT family that has this cluster
        - overall_mean_turns: The average number of turns in the conversation in this time period with this GPT family
        - cluster_mean_conversation_length: The average number of characters in the conversation in this time period with this GPT family that has this cluster
        - overall_mean_conversation_length: The average number of characters in the conversation in this time period with this GPT family
    :rtype: pd.DataFrame
    """
    # Initialize the DataFrame
    panel = {
        "time": [],
        "is_gpt4": [],
        "cluster": [],
        "cluster_nsamples_user_concepts_explicit": [],
        "cluster_nsamples_assistant_concepts_explicit": [],
        "cluster_nsamples_user_concepts_related": [],
        "cluster_nsamples_assistant_concepts_related": [],
        "cluster_nsamples": [],
        "overall_nsamples": [],
        "cluster_mean_turns": [],
        "overall_mean_turns": [],
        "cluster_mean_conversation_length": [],
        "overall_mean_conversation_length": [],
        "gpt_version": [],
    }

    # Find start time, and round to the nearest 1st / 16th of the month
    start_time = deepcopy(min([sample.time for sample in samples]))
    print(f"Precise start time: {start_time}")
    if start_time.day < 16:
        start_time = start_time.replace(day=1)
    else:
        start_time = start_time.replace(day=16)
    
    print(f"Rounded start time: {start_time}")
    
    # Classify samples into combinations of time intervals and GPT versions
    sample_directories: Dict[Tuple, List[DataSample]] = defaultdict(list)
    for sample in samples:
        time_diff = (sample.time - start_time).days
        time_interval_num = time_diff // time_interval
        is_gpt = 0 if "gpt-3.5-turbo-" in sample.gpt_version else 1
        assert "gpt-4-" in sample.gpt_version or "gpt-3.5-turbo-" in sample.gpt_version
        sample_directories[(time_interval_num, is_gpt)].append(sample)
    
    gpt_version_str2int = {
        "gpt-3.5-turbo-0301": 0,
        "gpt-4-0314": 0,
        "gpt-3.5-turbo-0613": 1,
        "gpt-4-1106-preview": 1,
        "gpt-3.5-turbo-0125": 2,
        "gpt-4-0125-preview": 2,
    }
    
    # Construct DataFrame rows
    samples_sizes = []
    for time_interval_num, is_gpt in sample_directories:
        cur_samples = sample_directories[(time_interval_num, is_gpt)]
        overall_nsamples = len(cur_samples)
        samples_sizes.append(overall_nsamples)
        
        # Calculate the mode of the GPT versions
        gpt_versions = [gpt_version_str2int[sample.gpt_version] for sample in cur_samples]
        maj_gpt_version = max(set(gpt_versions), key=gpt_versions.count)
        if gpt_versions.count(maj_gpt_version) < overall_nsamples * 0.95:
            print(f"Info: GPT version mode is not dominant in time interval {time_interval_num} (GPT-{3.5 if maj_gpt_version == 0 else 4}), distribution: {Counter(gpt_versions)}")
        
        # Calculate turns and conversation length
        overall_turns = [len(sample.conversation) for sample in cur_samples]
        overall_mean_turns = np.mean(overall_turns)
        overall_conversation_lengths = [sample.conversation_chars() for sample in cur_samples]
        overall_mean_conversation_length = np.mean(overall_conversation_lengths)
        
        if overall_nsamples < 100:
            print(f"Skipping time interval {time_interval_num} due to insufficient samples ({overall_nsamples}).")
            for concept in selected_clusters:
                panel["time"].append(time_interval_num)
                panel["is_gpt4"].append(is_gpt)
                panel["cluster"].append(concept)
                panel["cluster_nsamples_user_concepts_explicit"].append(None)
                panel["cluster_nsamples_assistant_concepts_explicit"].append(None)
                panel["cluster_nsamples_user_concepts_related"].append(None)
                panel["cluster_nsamples_assistant_concepts_related"].append(None)
                panel["cluster_nsamples"].append(None)
                panel["overall_nsamples"].append(overall_nsamples)
                panel["cluster_mean_turns"].append(None)
                panel["overall_mean_turns"].append(overall_mean_turns)
                panel["cluster_mean_conversation_length"].append(None)
                panel["overall_mean_conversation_length"].append(overall_conversation_lengths)
                panel["gpt_version"].append(maj_gpt_version)
        
        concept_mapping: Dict[int, List[DataSample]] = defaultdict(list)
        breakdown_counters = defaultdict(Counter)
        
        def increment_ancestors(concept: int, counter: Counter):
            while concept is not None:
                counter(concept) += 1
                concept = cluster_parent[concept]
        
        def insert_ancestors(concept: int, mapping: Dict[int, List[DataSample]], sample: DataSample):
            while concept is not None:
                mapping[concept].append(sample)
                concept = cluster_parent[concept]
        
        for sample in cur_samples:
            for concept in sample.concepts:
                insert_ancestors(concept, concept_mapping, sample)
            
            for key, value in sample.concepts_breakdown.items():
                for concept in value:
                    increment_ancestors(concept, breakdown_counters[key])
        
        for concept in selected_clusters:
            panel["time"].append(time_interval_num)
            panel["is_gpt4"].append(is_gpt)
            panel["cluster"].append(concept)
            panel["cluster_nsamples_user_concepts_explicit"].append(breakdown_counters["user_concepts_explicit"][concept])
            panel["cluster_nsamples_assistant_concepts_explicit"].append(breakdown_counters["assistant_concepts_explicit"][concept])
            panel["cluster_nsamples_user_concepts_related"].append(breakdown_counters["user_concepts_related"][concept])
            panel["cluster_nsamples_assistant_concepts_related"].append(breakdown_counters["assistant_concepts_related"][concept])
            panel["cluster_nsamples"].append(len(concept_mapping[concept]))
            panel["overall_nsamples"].append(overall_nsamples)
            panel["cluster_mean_turns"].append(np.mean([len(sample.conversation) for sample in concept_mapping[concept]]))
            panel["overall_mean_turns"].append(overall_mean_turns)
            panel["cluster_mean_conversation_length"].append(np.mean([sample.conversation_chars() for sample in concept_mapping[concept]]))
            panel["overall_mean_conversation_length"].append(overall_mean_conversation_length)
            panel["gpt_version"].append(maj_gpt_version)
    
    # Build the DataFrame and set the index
    panel = pd.DataFrame(panel)
    panel.set_index(["time", "is_gpt4", "cluster"], inplace=True)
    
    return panel