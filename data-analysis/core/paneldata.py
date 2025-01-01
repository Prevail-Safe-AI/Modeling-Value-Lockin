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
    :param time_interval: Time interval in days; serving as the unit of time, defaults to 15.
    :type time_interval: int, optional
    :return: A pandas DataFrame with the following columns:
        - time (INDEX): The number of time intervals since the start
        - is_gpt4 (INDEX): 1 for the GPT-4 family and 0 for the GPT-3.5-turbo family
        - cluster (INDEX): The concept that is being counted, represented by its index in the cluster list
        - freq: The number of samples that has this cluster in this time period with this GPT family
        - total_samples: The total number of samples in this time period with this GPT family
        - gpt_version: The version number within each GPT family that is being used in this time period
          - 0: GPT-3.5-turbo-0301 / GPT-4-0314
          - 1: GPT-3.5-turbo-0613 / GPT-4-1106-preview
          - 2: GPT-3.5-turbo-0125 / GPT-4-0125-preview
    :rtype: pd.DataFrame
    """
    # Initialize the DataFrame
    panel = pd.DataFrame(
        {
            "time": [],
            "is_gpt4": [],
            "cluster": [],
            "freq": [],
            "total_samples": [],
            "gpt_version": [],
        }
    )

    # Find start time, and round to the nearest 1st / 16th of the month
    start_time = deepcopy(min([sample.time for sample in samples]))
    print(f"Precise start time: {start_time}")
    if start_time.day < 16:
        start_time = start_time.replace(day=1)
    else:
        start_time = start_time.replace(day=16)
    
    print(f"Rounded start time: {start_time}")
    
    # Classify samples into combinations of time intervals and GPT versions
    sample_directories = defaultdict(list)
    for sample in samples:
        time_diff = (sample.time - start_time).days
        time_interval_num = time_diff // time_interval
        gpt_version = 0 if "gpt-3.5-turbo-" in sample.gpt_version else 1
        assert "gpt-4-" in sample.gpt_version or "gpt-3.5-turbo-" in sample.gpt_version
        sample_directories[(time_interval_num, gpt_version)].append(sample)
    
    # Build the panel
    for time_interval_num, gpt_version in sample_directories:
        time_interval_samples = sample_directories[(time_interval_num, gpt_version)]
        total_samples = len(time_interval_samples)
        pass