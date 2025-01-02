import pandas as pd
import numpy as np
from datetime import datetime
from copy import deepcopy
from typing import List, Dict, Tuple, Set
from core.samples import DataSample
from core.concepts import CLUSTER_MIN_SIZE, CLUSTER_STEP_MULTIPLIER_LOG2
from collections import defaultdict, Counter


gpt_version_str2int = {
    "gpt-3.5-turbo-0301": 0,
    "gpt-4-0314": 0,
    "gpt-3.5-turbo-0613": 1,
    "gpt-4-1106-preview": 1,
    "gpt-3.5-turbo-0125": 2,
    "gpt-4-0125-preview": 2,
}


TIME_INTERVAL_DAYS = 15
MAX_TIME_INTERVALS = 24
START_TIME = datetime(2023, 4, 1, 0, 0, 0)
def get_time_interval(time: datetime) -> int:
    res = (time - START_TIME).days // TIME_INTERVAL_DAYS
    assert 0 <= res <= MAX_TIME_INTERVALS
    return min(res, MAX_TIME_INTERVALS - 1)


def calculate_diversity(
    concepts_present: List[int], 
    cluster_parent: List[int],
    cluster_size: List[int],
    cluster_name: List[str],
    cluster_prob: List[float],
    cluster_selected_parent: List[int],
    selected_clusters: List[int],
    root: int,
) -> float:
    depth: Dict[int, int] = {}
    
    def get_depth(concept: int) -> int:
        if concept in depth:
            return depth[concept]
        
        if cluster_selected_parent[concept] is None:
            depth[concept] = 0
            return 0
        
        depth[concept] = get_depth(cluster_selected_parent[concept]) + 1
    
    subtree_counts: Dict[int, int] = defaultdict(int)
    for concept in concepts_present:
        concept = cluster_selected_parent[concept]
        while concept is not None:
            subtree_counts[concept] += 1
            concept = cluster_selected_parent[concept]
    
    def get_weight(concept: int) -> float:
        return np.log2(cluster_size[concept]) / CLUSTER_STEP_MULTIPLIER_LOG2 - get_depth(concept)
    
    nsamples = len(concepts_present)
    diversity = get_weight(root) * nsamples * (nsamples - 1)
    
    for concept, count in subtree_counts.items():
        if concept == root:
            continue
        
        diversity += (get_weight(concept) - get_weight(cluster_selected_parent[concept])) * count * (count - 1)
    
    return diversity / (nsamples * (nsamples - 1))


def build_user_panel(
    samples: List[DataSample],
    cluster_parent: List[int],
    cluster_size: List[int],
    cluster_name: List[str],
    cluster_prob: List[float],
    cluster_selected_parent: List[int],
    selected_clusters: List[int],
    root: int,
) -> pd.DataFrame:
    """Build a panel dataset with each row representing a user.

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
    :param root: The root cluster index.
    :type root: int
    :return: Pandas DataFrame with the following columns:
        - user (INDEX): user ID
        - language: typical language the user speaks
        - location: typical location the user is from. It is a tuple of two optional strings, representing the country and state respectively.
        - nsamples: total number of conversations the user had
        - nsamples_temporal_composition: number of conversations the user had during each time period respectivly. It is a tuple of 24 ints.
        - nsamples_version_composition: number of conversations the user had with each GPT version. It is a tuple of six ints, representing the following versions respectively:
            - GPT-3.5-turbo-0301 
            - GPT-3.5-turbo-0613
            - GPT-3.5-turbo-0125
            - GPT-4-0314
            - GPT-4-1106-preview
            - GPT-4-0125-preview
        - temporal_extension: standard deviation of the time interval index of the conversations the user had
        - version_diversity: square sum of the proportion of conversations the user had with each GPT version
        - mean_turns: average number of turns in the conversations the user had
        - mean_conversation_length: average number of characters in the conversations the user had
        - mean_prompt_length: average number of characters in the prompts the user gave
        - concept_diversity: variance of the concept distribution that the user's conversations contain
        - concept_diversity_user_concepts_explicit: variance of the concept distribution that the user explicitly mentioned
        - concept_diversity_assistant_concepts_explicit: variance of the concept distribution that the assistant explicitly mentioned
        - concept_diversity_user_concepts_related: variance of the concept distribution that the user related to
        - concept_diversity_assistant_concepts_related: variance of the concept distribution that the assistant related to
        - concept_diversity_assistant_across_time: variance of the concept distribution in assistant speech during each time period respectivly. It is a tuple of 24 floats.
        - concept_diversity_user_across_time: variance of the concept distribution in user speech during each time period respectivly. It is a tuple of 24 floats.
        - concept_diversity_across_time: variance of the concept distribution that appeared during each time period respectivly. It is a tuple of 24 floats.
        
    :rtype: pd.DataFrame
    """
    # Initialize the DataFrame
    panel = {
        "user": [],
        "language": [],
        "location": [],
        "nsamples": [],
        "nsamples_temporal_composition": [],
        "nsamples_version_composition": [],
        "temporal_extension": [],
        "version_diversity": [],
        "mean_turns": [],
        "mean_conversation_length": [],
        "mean_prompt_length": [],
        "concept_diversity": [],
        "concept_diversity_user_concepts_explicit": [],
        "concept_diversity_assistant_concepts_explicit": [],
        "concept_diversity_user_concepts_related": [],
        "concept_diversity_assistant_concepts_related": [],
        "concept_diversity_user_across_time": [],
        "concept_diversity_assistant_across_time": [],
        "concept_diversity_across_time": [],    
    }
    
    # Classify samples by users
    user_directories: Dict[str, List[DataSample]] = defaultdict(list)
    for sample in samples:
        user_directories[sample.user_id].append(sample)
    
    # Construct DataFrame rows
    for user_id in user_directories:
        cur_samples = user_directories[user_id]
        nsamples = len(cur_samples)
        
        # Calculate the mode of language and location
        languages = [sample.language for sample in cur_samples]
        locations = [sample.location for sample in cur_samples]
        maj_language = max(set(languages), key=languages.count)
        maj_location = max(set(locations), key=locations.count)
        
        # Classify samples by time intervals and GPT versions respectively
        time_directory: List[List[DataSample]] = [[] for _ in range(MAX_TIME_INTERVALS)]
        version_directory: Dict[int, List[DataSample]] = defaultdict(list)
        for sample in cur_samples:
            time_interval_num = get_time_interval(sample.time)
            time_directory[time_interval_num].append(sample)
            
            version_num = gpt_version_str2int[sample.gpt_version] + 3 * int("gpt-4-" in sample.gpt_version)
            version_directory[version_num].append(sample)
        
        # Calculate temporal extension
        temporal_weights = [len(time_directory[i]) for i in range(MAX_TIME_INTERVALS)]
        mean_time_interval = np.average(list(range(MAX_TIME_INTERVALS)), weights=temporal_weights)
        temporal_extension = float(np.sqrt(np.average([(i - mean_time_interval) ** 2 for i in range(MAX_TIME_INTERVALS)], weights=temporal_weights)))

        # Calculate version diversity
        version_weights = np.array([len(version_directory[i]) for i in range(6)])
        version_diversity = np.sum((version_weights / nsamples) ** 2)
        
        # Calculate mean turns and conversation length
        mean_turns = np.mean([len(sample.conversation) for sample in cur_samples])
        mean_conversation_length = np.mean([sample.conversation_chars() for sample in cur_samples])
        mean_prompt_length = np.mean([sample.role_chars("user") for sample in cur_samples])
        
        def get_concept_list(samples: List[DataSample], breakdown_key: str = None) -> List[int]:
            concept_list = []
            for sample in samples:
                if breakdown_key is None:
                    concept_list.extend(sample.concepts if sample.concepts is not None else [])
                else:
                    concept_list.extend(
                        sample.concepts_breakdown[breakdown_key]
                        if sample.concepts_breakdown is not None and breakdown_key in sample.concepts_breakdown
                        else []
                    )
            
            return concept_list
        
        generic_arguments = {
            "cluster_parent": cluster_parent,
            "cluster_size": cluster_size,
            "cluster_name": cluster_name,
            "cluster_prob": cluster_prob,
            "cluster_selected_parent": cluster_selected_parent,
            "selected_clusters": selected_clusters,
            "root": root,
        }
        
        # Calculate concept diversity
        concept_diversity = calculate_diversity(
            get_concept_list(cur_samples),
            **generic_arguments
        )
        concept_diversity_user_concepts_explicit = calculate_diversity(
            get_concept_list(cur_samples, "user_concepts_explicit"),
            **generic_arguments
        )
        concept_diversity_assistant_concepts_explicit = calculate_diversity(
            get_concept_list(cur_samples, "assistant_concepts_explicit"),
            **generic_arguments
        )
        concept_diversity_user_concepts_related = calculate_diversity(
            get_concept_list(cur_samples, "user_concepts_related"),
            **generic_arguments
        )
        concept_diversity_assistant_concepts_related = calculate_diversity(
            get_concept_list(cur_samples, "assistant_concepts_related"),
            **generic_arguments
        )
        concept_diversity_user_across_time = [
            calculate_diversity(
                get_concept_list(time_directory[i], "user_concepts_explicit") +
                get_concept_list(time_directory[i], "user_concepts_related"),
                **generic_arguments
            )
            for i in range(MAX_TIME_INTERVALS)
        ]
        concept_diversity_assistant_across_time = [
            calculate_diversity(
                get_concept_list(time_directory[i], "assistant_concepts_explicit") +
                get_concept_list(time_directory[i], "assistant_concepts_related"),
                **generic_arguments
            )
            for i in range(MAX_TIME_INTERVALS)
        ]
        concept_diversity_across_time = [
            calculate_diversity(
                get_concept_list(time_directory[i]),
                **generic_arguments
            )
            for i in range(MAX_TIME_INTERVALS)
        ]
        
        # Append the row to the DataFrame
        panel["user"].append(user_id)
        panel["language"].append(maj_language)
        panel["location"].append(maj_location)
        panel["nsamples"].append(nsamples)
        panel["nsamples_temporal_composition"].append(tuple([len(time_directory[i]) for i in range(MAX_TIME_INTERVALS)]))
        panel["nsamples_version_composition"].append(tuple([len(version_directory[i]) for i in range(6)]))
        panel["temporal_extension"].append(temporal_extension)
        panel["version_diversity"].append(version_diversity)
        panel["mean_turns"].append(mean_turns)
        panel["mean_conversation_length"].append(mean_conversation_length)
        panel["mean_prompt_length"].append(mean_prompt_length)
        panel["concept_diversity"].append(concept_diversity)
        panel["concept_diversity_user_concepts_explicit"].append(concept_diversity_user_concepts_explicit)
        panel["concept_diversity_assistant_concepts_explicit"].append(concept_diversity_assistant_concepts_explicit)
        panel["concept_diversity_user_concepts_related"].append(concept_diversity_user_concepts_related)
        panel["concept_diversity_assistant_concepts_related"].append(concept_diversity_assistant_concepts_related)
        panel["concept_diversity_user_across_time"].append(tuple(concept_diversity_user_across_time))
        panel["concept_diversity_assistant_across_time"].append(tuple(concept_diversity_assistant_across_time))
        panel["concept_diversity_across_time"].append(tuple(concept_diversity_across_time))
    
    # Build the DataFrame and set the index
    panel = pd.DataFrame(panel)
    panel.set_index("user", inplace=True)
    
    return panel
        

def build_temporal_panel(
    samples: List[DataSample], 
    cluster_parent: List[int],
    cluster_size: List[int],
    cluster_name: List[str],
    cluster_prob: List[float],
    cluster_selected_parent: List[int],
    selected_clusters: List[int],
    root: int,
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
    :param root: The root cluster index.
    :type root: int
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

    # Classify samples into combinations of time intervals and GPT versions
    sample_directories: Dict[Tuple, List[DataSample]] = defaultdict(list)
    for sample in samples:
        time_interval_num = get_time_interval(sample.time)
        is_gpt = 0 if "gpt-3.5-turbo-" in sample.gpt_version else 1
        assert "gpt-4-" in sample.gpt_version or "gpt-3.5-turbo-" in sample.gpt_version
        sample_directories[(time_interval_num, is_gpt)].append(sample)
    
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