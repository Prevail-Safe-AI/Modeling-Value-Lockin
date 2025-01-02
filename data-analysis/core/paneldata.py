import pandas as pd
import numpy as np
import time
from tqdm import tqdm
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
START_TIME = datetime(2023, 4, 1, 0, 0, 0).replace(tzinfo=None)
def get_time_interval(time: datetime) -> int:
    res = (time.replace(tzinfo=None) - START_TIME).days // TIME_INTERVAL_DAYS
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
    if not concepts_present:
        return np.nan
    
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


panel_index_cols = {
    "user_panel": ["user"],
    "temporal_panel1": ["time", "is_gpt4"],
    "temporal_panel2": ["time", "is_gpt4", "cluster"],
}


def create_panel_and_backup(data: List[Dict], panel_name: str) -> pd.DataFrame:
    """Create a panel from the data and save it as a backup.

    :param data: List of dictionaries, each representing a row of the panel.
    :type data: List[Dict]
    :param panel_name: Name of the panel.
    :type panel_name: str
    :param index_cols: List of column names to be used as the index.
    :type index_cols: List[str]
    :return: The panel DataFrame.
    :rtype: pd.DataFrame
    """
    print(f"Creating panel {panel_name}...")
    panel = pd.DataFrame(data)
    panel.set_index(panel_index_cols[panel_name], inplace=True)
    
    print(f"Saving panel {panel_name}...")
    panel.to_csv(f"./data/{panel_name}-{time.strftime('%Y%m%d-%H%M%S')}.csv")
    
    print(f"Panel {panel_name} created and saved.")
    print(f"Panel {panel_name} shape: {panel.shape}")
    print(f"Panel {panel_name} head:\n{panel.head()}")
    print(f"Panel {panel_name} info:\n{panel.info()}")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(f"Panel {panel_name} summary stats:\n{panel.describe()}")
    
    return panel


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
    
    generic_arguments = {
        "cluster_parent": cluster_parent,
        "cluster_size": cluster_size,
        "cluster_name": cluster_name,
        "cluster_prob": cluster_prob,
        "cluster_selected_parent": cluster_selected_parent,
        "selected_clusters": selected_clusters,
        "root": root,
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
    panel = create_panel_and_backup(panel, "user_panel")
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    :return: Two pandas DataFrames,
      The first one (stats about each time period) with the following columns:
        - time (INDEX): The number of time intervals since the start
        - is_gpt4 (INDEX): 1 for the GPT-4 family and 0 for the GPT-3.5-turbo family
        - gpt_version: The version number within each GPT family that is being used in this time period
          - 0: GPT-3.5-turbo-0301 / GPT-4-0314
          - 1: GPT-3.5-turbo-0613 / GPT-4-1106-preview
          - 2: GPT-3.5-turbo-0125 / GPT-4-0125-preview
        - overall_nsamples: The total number of samples in this time period with this GPT family
        - overall_mean_turns: The average number of turns in the conversation in this time period with this GPT family
        - overall_mean_conversation_length: The average number of characters in the conversation in this time period with this GPT family
        - overall_mean_prompt_length: The average number of characters in the prompts in this time period with this GPT family
        - concept_diversity: The variance of the concept distribution that appeared in this time period
        - concept_diversity_user_concepts_explicit: The variance of the concept distribution in user speech (explicit) that appeared in this time period
        - concept_diversity_assistant_concepts_explicit: The variance of the concept distribution in assistant speech (explicit) that appeared in this time period
        - concept_diversity_user_concepts_related: The variance of the concept distribution in user speech (related) that appeared in this time period
        - concept_diversity_assistant_concepts_related: The variance of the concept distribution in assistant speech (related) that appeared in this time period
        - concept_diversity_user: The variance of the concept distribution in user speech (explicit or related) that appeared in this time period
        - concept_diversity_assistant: The variance of the concept distribution in assistant speech (explicit or related) that appeared in this time period
      The second one (stats about each concept in each time period) with the following columns:
        - time (INDEX): The number of time intervals since the start
        - is_gpt4 (INDEX): 1 for the GPT-4 family and 0 for the GPT-3.5-turbo family
        - cluster (INDEX): The concept that is being counted, represented by its index in the cluster list
            - If a certain triplet of indices (time, is_gpt4, cluster) is not in the DataFrame, it means that the concept is not present in the samples of that time period and GPT family
        - cluster_nsamples_user_concepts_explicit: The number of samples in this time period with this GPT family that has this cluster and the user explicitly mentioned it
        - cluster_nsamples_assistant_concepts_explicit: The number of samples in this time period with this GPT family that has this cluster and the assistant explicitly mentioned it
        - cluster_nsamples_user_concepts_related: The number of samples in this time period with this GPT family that has this cluster as a related concept to the user speech
        - cluster_nsamples_assistant_concepts_related: The number of samples in this time period with this GPT family that has this cluster as a related concept to the assistant speech
        - cluster_nsamples: The number of samples in this time period with this GPT family that has this cluster
        - cluster_mean_turns: The average number of turns in the conversation in this time period with this GPT family that has this cluster
        - cluster_mean_conversation_length: The average number of characters in the conversation in this time period with this GPT family that has this cluster
        - cluster_mean_prompt_length: The average number of characters in the prompts in this time period with this GPT family that has this cluster
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
    # Initialize the DataFrame
    panel1 = {
        "time": [],
        "is_gpt4": [],
        "gpt_version": [],
        "overall_nsamples": [],
        "overall_mean_turns": [],
        "overall_mean_conversation_length": [],
        "overall_mean_prompt_length": [],
        "concept_diversity": [],
        "concept_diversity_user_concepts_explicit": [],
        "concept_diversity_assistant_concepts_explicit": [],
        "concept_diversity_user_concepts_related": [],
        "concept_diversity_assistant_concepts_related": [],
        "concept_diversity_user": [],
        "concept_diversity_assistant": [],
    }
    panel2 = {
        "time": [],
        "is_gpt4": [],
        "cluster": [],
        "cluster_nsamples_user_concepts_explicit": [],
        "cluster_nsamples_assistant_concepts_explicit": [],
        "cluster_nsamples_user_concepts_related": [],
        "cluster_nsamples_assistant_concepts_related": [],
        "cluster_nsamples": [],
        "cluster_mean_turns": [],
        "cluster_mean_conversation_length": [],
        "cluster_mean_prompt_length": [],
    }
    
    generic_arguments = {
        "cluster_parent": cluster_parent,
        "cluster_size": cluster_size,
        "cluster_name": cluster_name,
        "cluster_prob": cluster_prob,
        "cluster_selected_parent": cluster_selected_parent,
        "selected_clusters": selected_clusters,
        "root": root,
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
    for time_interval_num, is_gpt in tqdm(sample_directories):
        cur_samples = sample_directories[(time_interval_num, is_gpt)]
        overall_nsamples = len(cur_samples)
        samples_sizes.append(overall_nsamples)
        
        # Calculate the mode of the GPT versions
        gpt_versions = [gpt_version_str2int[sample.gpt_version] for sample in cur_samples]
        maj_gpt_version = max(set(gpt_versions), key=gpt_versions.count) if overall_nsamples > 0 else None
        if gpt_versions.count(maj_gpt_version) < overall_nsamples * 0.95:
            print(f"Info: GPT version mode is not dominant in time interval {time_interval_num} (GPT-{3.5 if maj_gpt_version == 0 else 4}), distribution: {Counter(gpt_versions)}")
        
        # Calculate turns and conversation length
        overall_turns = [len(sample.conversation) for sample in cur_samples]
        overall_mean_turns = float(np.mean(overall_turns))
        overall_conversation_lengths = [sample.conversation_chars() for sample in cur_samples]
        overall_mean_conversation_length = float(np.mean(overall_conversation_lengths))
        overall_mean_prompt_length = float(np.mean([sample.role_chars("user") for sample in cur_samples]))
        
        if overall_nsamples < 100:
            print(f"Skipping time interval {time_interval_num} due to insufficient samples ({overall_nsamples}).")
            
            panel1["time"].append(time_interval_num)
            panel1["is_gpt4"].append(is_gpt)
            panel1["gpt_version"].append(maj_gpt_version)
            panel1["overall_nsamples"].append(overall_nsamples)
            panel1["overall_mean_turns"].append(overall_mean_turns)
            panel1["overall_mean_conversation_length"].append(overall_conversation_lengths)
            panel1["overall_mean_prompt_length"].append(overall_mean_prompt_length)
            panel1["concept_diversity"].append(None)
            panel1["concept_diversity_user_concepts_explicit"].append(None)
            panel1["concept_diversity_assistant_concepts_explicit"].append(None)
            panel1["concept_diversity_user_concepts_related"].append(None)
            panel1["concept_diversity_assistant_concepts_related"].append(None)
            panel1["concept_diversity_user"].append(None)
            panel1["concept_diversity_assistant"].append(None)
            
            ## Skip adding row to the second panel since there are no samples
            # for concept in selected_clusters:
            #     panel2["time"].append(time_interval_num)
            #     panel2["is_gpt4"].append(is_gpt)
            #     panel2["cluster"].append(concept)
            #     panel2["cluster_nsamples_user_concepts_explicit"].append(None)
            #     panel2["cluster_nsamples_assistant_concepts_explicit"].append(None)
            #     panel2["cluster_nsamples_user_concepts_related"].append(None)
            #     panel2["cluster_nsamples_assistant_concepts_related"].append(None)
            #     panel2["cluster_nsamples"].append(None)
            #     panel2["cluster_mean_turns"].append(None)
            #     panel2["cluster_mean_conversation_length"].append(None)
            #     panel2["cluster_mean_prompt_length"].append(None)
        
        # Calculate concept diversity for panel1
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
        concept_diversity_user = calculate_diversity(
            get_concept_list(cur_samples, "user_concepts_explicit") + get_concept_list(cur_samples, "user_concepts_related"),
            **generic_arguments
        )
        concept_diversity_assistant = calculate_diversity(
            get_concept_list(cur_samples, "assistant_concepts_explicit") + get_concept_list(cur_samples, "assistant_concepts_related"),
            **generic_arguments
        )
        
        # Append the row to panel1
        panel1["time"].append(time_interval_num)
        panel1["is_gpt4"].append(is_gpt)
        panel1["gpt_version"].append(maj_gpt_version)
        panel1["overall_nsamples"].append(overall_nsamples)
        panel1["overall_mean_turns"].append(overall_mean_turns)
        panel1["overall_mean_conversation_length"].append(overall_mean_conversation_length)
        panel1["overall_mean_prompt_length"].append(overall_mean_prompt_length)
        panel1["concept_diversity"].append(concept_diversity)
        panel1["concept_diversity_user_concepts_explicit"].append(concept_diversity_user_concepts_explicit)
        panel1["concept_diversity_assistant_concepts_explicit"].append(concept_diversity_assistant_concepts_explicit)
        panel1["concept_diversity_user_concepts_related"].append(concept_diversity_user_concepts_related)
        panel1["concept_diversity_assistant_concepts_related"].append(concept_diversity_assistant_concepts_related)
        panel1["concept_diversity_user"].append(concept_diversity_user)
        panel1["concept_diversity_assistant"].append(concept_diversity_assistant)
        
        concept_mapping: Dict[int, List[DataSample]] = defaultdict(list)
        breakdown_counters = defaultdict(Counter)
        
        def increment_ancestors(concept: int, counter: Counter):
            while concept is not None:
                counter[concept] += 1
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
        
        # for concept in selected_clusters:
        for concept in concept_mapping:
            panel2["time"].append(time_interval_num)
            panel2["is_gpt4"].append(is_gpt)
            panel2["cluster"].append(concept)
            panel2["cluster_nsamples_user_concepts_explicit"].append(breakdown_counters["user_concepts_explicit"][concept])
            panel2["cluster_nsamples_assistant_concepts_explicit"].append(breakdown_counters["assistant_concepts_explicit"][concept])
            panel2["cluster_nsamples_user_concepts_related"].append(breakdown_counters["user_concepts_related"][concept])
            panel2["cluster_nsamples_assistant_concepts_related"].append(breakdown_counters["assistant_concepts_related"][concept])
            panel2["cluster_nsamples"].append(len(concept_mapping[concept]))
            panel2["cluster_mean_turns"].append(np.mean([len(sample.conversation) for sample in concept_mapping[concept]]))
            panel2["cluster_mean_conversation_length"].append(np.mean([sample.conversation_chars() for sample in concept_mapping[concept]]))
            panel2["cluster_mean_prompt_length"].append(np.mean([sample.role_chars("user") for sample in concept_mapping[concept]]))
    
    # Build the DataFrame and set the index
    panel1 = create_panel_and_backup(panel1, "temporal_panel1")
    panel2 = create_panel_and_backup(panel2, "temporal_panel2")
    return panel1, panel2


def build_all_panels(existing_dict, *args, **kwargs) -> Dict[str, pd.DataFrame]:
    """Build all panels from the samples and cluster information.

    :return: Three pandas DataFrames in a dictionary,
      The first one (user_panel) with the following columns:
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
      The second one (temporal_panel1) with the following columns:
        - time (INDEX): The number of time intervals since the start
        - is_gpt4 (INDEX): 1 for the GPT-4 family and 0 for the GPT-3.5-turbo family
        - gpt_version: The version number within each GPT family that is being used in this time period
          - 0: GPT-3.5-turbo-0301 / GPT-4-0314
          - 1: GPT-3.5-turbo-0613 / GPT-4-1106-preview
          - 2: GPT-3.5-turbo-0125 / GPT-4-0125-preview
        - overall_nsamples: The total number of samples in this time period with this GPT family
        - overall_mean_turns: The average number of turns in the conversation in this time period with this GPT family
        - overall_mean_conversation_length: The average number of characters in the conversation in this time period with this GPT family
        - overall_mean_prompt_length: The average number of characters in the prompts in this time period with this GPT family
        - concept_diversity: The variance of the concept distribution that appeared in this time period
        - concept_diversity_user_concepts_explicit: The variance of the concept distribution in user speech (explicit) that appeared in this time period
        - concept_diversity_assistant_concepts_explicit: The variance of the concept distribution in assistant speech (explicit) that appeared in this time period
        - concept_diversity_user_concepts_related: The variance of the concept distribution in user speech (related) that appeared in this time period
        - concept_diversity_assistant_concepts_related: The variance of the concept distribution in assistant speech (related) that appeared in this time period
        - concept_diversity_user: The variance of the concept distribution in user speech (explicit or related) that appeared in this time period
        - concept_diversity_assistant: The variance of the concept distribution in assistant speech (explicit or related) that appeared in this time period
      The third one (temporal_panel2) with the following columns:
        - time (INDEX): The number of time intervals since the start
        - is_gpt4 (INDEX): 1 for the GPT-4 family and 0 for the GPT-3.5-turbo family
        - cluster (INDEX): The concept that is being counted, represented by its index in the cluster list
            - If a certain triplet of indices (time, is_gpt4, cluster) is not in the DataFrame, it means that the concept is not present in the samples of that time period and GPT family
        - cluster_nsamples_user_concepts_explicit: The number of samples in this time period with this GPT family that has this cluster and the user explicitly mentioned it
        - cluster_nsamples_assistant_concepts_explicit: The number of samples in this time period with this GPT family that has this cluster and the assistant explicitly mentioned it
        - cluster_nsamples_user_concepts_related: The number of samples in this time period with this GPT family that has this cluster as a related concept to the user speech
        - cluster_nsamples_assistant_concepts_related: The number of samples in this time period with this GPT family that has this cluster as a related concept to the assistant speech
        - cluster_nsamples: The number of samples in this time period with this GPT family that has this cluster
        - cluster_mean_turns: The average number of turns in the conversation in this time period with this GPT family that has this cluster
        - cluster_mean_conversation_length: The average number of characters in the conversation in this time period with this GPT family that has this cluster
        - cluster_mean_prompt_length: The average number of characters in the prompts in this time period with this GPT family that has this cluster
    :rtype: Dict[str, pd.DataFrame]
    """
    print(f"Building all panels... (existing_dict: {existing_dict.keys()})")
    
    if "user_panel" not in existing_dict or not isinstance(existing_dict["user_panel"], pd.DataFrame):
        print("Building user panel...")
        existing_dict["user_panel"] = build_user_panel(*args, **kwargs)
    
    if "temporal_panel1" not in existing_dict or not isinstance(existing_dict["temporal_panel1"], pd.DataFrame) or \
       "temporal_panel2" not in existing_dict or not isinstance(existing_dict["temporal_panel2"], pd.DataFrame):
        print("Building temporal panels...")
        existing_dict["temporal_panel1"], existing_dict["temporal_panel2"] = build_temporal_panel(*args, **kwargs)
    
    return existing_dict