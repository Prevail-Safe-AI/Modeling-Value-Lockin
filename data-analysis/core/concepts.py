import json, warnings, random, time, os
import voyageai
import hdbscan
from tqdm import tqdm
from typing import List, Dict, Tuple, Mapping
from typeguard import check_type
from copy import deepcopy
import voyageai.client
from core.samples import DataSample
from core.templates import (
    concept_extraction_template,
)
from utils.log_utils import silence_decorator, dynamic_printing_decorator, dump_file
from utils.json_utils import extract_json_from_str
from utils.nlp_utils import simplify_text
import numpy as np
import pandas as pd
import concurrent.futures as cf
import threading as th

CLUSTER_MIN_SIZE = 4
CLUSTER_STEP_MULTIPLIER_LOG2 = 2


def extract_concepts(samples: List[DataSample], extractor: str, max_retries: int = 5) -> List[DataSample]:
    """Extract concepts (list of strings) from the conversation content of each sample.

    :param samples: List of DataSample objects.
    :type samples: List[DataSample]
    
    :param extractor: LLM used for extracting concepts. Specify the path or repo ID.
    :type extractor: str
    
    :param max_retries: Number of retries allowed for failed extraction, possibly due to LLM mishandling formatting.
    :type max_retries: int
    
    :return: List of DataSample objects with the extracted concepts added to the `concepts` attribute. Operation may be done in place.
    :rtype: List[DataSample]
    """
    from ProgressGym import Data, Model, GlobalState
    with GlobalState(continuous_backend=True):
        extractor = Model(
            "extractor",
            model_path_or_repoid=extractor,
            template_type=("llama3" if "llama-3" in extractor.lower() else "auto"),
        )
        extraction_input = Data(
            "concept_extraction_input",
            data_content = [
                {
                    "instruction": concept_extraction_template.replace(
                        "{conversation}", json.dumps(sample.conversation, indent=2, sort_keys=True)
                    )
                }
                for sample in tqdm(samples)
            ]
        )
        extraction_output: Data = dynamic_printing_decorator(extractor.inference)(
            extraction_input,
            "concept_extraction_output",
            max_tokens=2048,
        )
        error_count = 0
        successful_count = 0
        for sample, dialogue in tqdm(zip(samples, extraction_output.all_passages())):
            sample: DataSample
            try:
                sample.concepts_breakdown = check_type(
                    extract_json_from_str(dialogue["predict"]),
                    Dict[str, List[str]],
                )
                sample.concepts = list(set(
                    sample.concepts_breakdown["user_concepts_explicit"] +
                    sample.concepts_breakdown["user_concepts_related"] +
                    sample.concepts_breakdown["assistant_concepts_explicit"] +
                    sample.concepts_breakdown["assistant_concepts_related"]
                ))
                successful_count += 1
            except:
                error_count += 1
                if not hasattr(sample, "concepts_breakdown"):
                    sample.concepts_breakdown = None
                if not hasattr(sample, "concepts"):
                    sample.concepts = None
        
        if error_count + successful_count != len(samples):
            warnings.warn(f"Counts of successful and failed extractions do not add up to the total number of samples: {successful_count} successful, {error_count} failed, expected {len(samples)} total.")
        
        print(f"Failed to extract concepts from {error_count} out of {len(samples)} conversations.")
        if error_count and max_retries:
            print(f"Retrying... ({max_retries} retries remain)")
            failed_indices = [
                i for i, sample in tqdm(enumerate(samples)) if not hasattr(sample.concepts)
            ]
            assert len(failed_indices) == error_count
            
            retry_results = extract_concepts(
                [samples[i] for i in failed_indices],
                extractor, 
                max_retries - 1,
            )
            assert len(retry_results) == error_count
            
            for retry_id, raw_id in enumerate(failed_indices):
                samples[raw_id] = retry_results[retry_id]
    
    return samples

def simplify_concepts(samples: List[DataSample]) -> List[DataSample]:
    for sample in tqdm(samples):
        if not sample.concepts:
            continue
        
        sample.concepts = [simplify_text(concept) for concept in sample.concepts]
        for key in sample.concepts_breakdown:
            sample.concepts_breakdown[key] = [simplify_text(concept) for concept in sample.concepts_breakdown[key]]
    
    return samples

PARALLEL_THREADS = 64
MAX_RPS = 1900 / 60
NUM_RETRIES = 500
EMB_DIM = 256

def fill_in_embeddings(vo: voyageai.Client, embeddings: np.array, strings: List[str], start_index: int, end_index: int, pbar: tqdm, lock: th.Lock) -> None:
    for retry_count in range(NUM_RETRIES):
        time.sleep(PARALLEL_THREADS/MAX_RPS)
        cur_emb = vo.embed(
            strings[start_index : end_index],
            model="voyage-3-large",
            output_dimension=EMB_DIM,
        ).embeddings
        
        if not isinstance(cur_emb, list) or \
            len(cur_emb) != len(strings[start_index : end_index]) or \
            np.isnan(cur_emb).any() or \
            np.isinf(cur_emb).any():
            
            if retry_count == NUM_RETRIES-1:
                print(f"{start_index}-{end_index} Failed to embed strings after {retry_count} retries. Dumping incomplete embeddings ({len(embeddings)} out of {len(strings)})...")
                with lock:
                    dump_file(list(zip(strings, embeddings.tolist())), f"embeddings-incomplete-{time.strftime('%Y%m%d-%H%M%S')}.json")
                
                raise ValueError("Failed to embed strings after 500 retries.")
            
            print(f"{start_index}-{end_index} Retrying for the {retry_count}-th time...")
            time.sleep(8 * PARALLEL_THREADS/MAX_RPS)
            continue
        
        cur_emb = np.array(cur_emb)
        assert cur_emb.shape == (end_index - start_index, EMB_DIM)
        
        with lock:
            embeddings[start_index : end_index] = cur_emb
            pbar.update(end_index - start_index)    
            
        break

def cluster_strs(strings: List[str]) -> Tuple[List[int], List[int], List[str], List[float], int]:
    """Hierarchically cluster strings to identify common themes and topics. Each node or cluster has a unique integer ID and a string summary, with the cluster summary being a few randomly selected strings from the cluster, weighted by probability.

    :param strings: List of strings to cluster.
    :type strings: List[str]
    :return: List of integers representing the parent ID of each string or cluster, followed by their sizes, followed by a list of strings representing the summary of each cluster or the content of each string, followed by a list of floats representing the probability of membership to its assigned cluster / robustness of the cluster itself, followed by the id of the root cluster.
    :rtype: Tuple[List[int], List[int], List[str], List[float], int]
    """
    print(f"Sorting {len(strings)} strings...")
    strings = sorted(strings)
    
    if os.environ.get("EMBEDDINGS", None) is not None:
        print("Loading embeddings...")
        with open(os.environ["EMBEDDINGS"], "r") as f:
            loaded_embeddings = json.load(f)
        
        print("Sorting loaded embeddings...")
        loaded_embeddings = sorted(loaded_embeddings, key=lambda x: x[0])
        
        embeddings = [embedding for _, embedding in tqdm(loaded_embeddings)]
        print("Embeddings loaded. Verifying...")
        
        if "dummy" not in os.environ.get("EMBEDDINGS", ""):
            for emb_combo, string in zip(loaded_embeddings, strings):
                s, emb = emb_combo
                assert isinstance(s, str) and isinstance(emb, list) and len(emb) == EMB_DIM
                assert not np.isnan(emb).any() and not np.isinf(emb).any()
                assert not np.all(emb == 0)
                assert s == string
        else:
            print("Dummy embeddings detected. Skipping verification and truncating to match embedding length.")
            strings = strings[:len(embeddings)]
        
        del loaded_embeddings
    
    else:
        print(f"Embedding strings ({len(strings)} total)...")
        vo = voyageai.Client()
        batch_size = 128
        embeddings = np.zeros((len(strings), EMB_DIM))
        
        with tqdm(total=len(strings)) as pbar:
            lock = th.Lock()
            with cf.ThreadPoolExecutor(max_workers=PARALLEL_THREADS) as executor:
                for i in range(0, len(strings), batch_size):
                    time.sleep(1/MAX_RPS)
                    executor.submit(fill_in_embeddings, vo, embeddings, strings, i, min(i + batch_size, len(strings)), pbar, lock)
                
                executor.shutdown(wait=True)
    
        embeddings = embeddings.tolist()
    
        # Backup embedding data
        print("Backing up embeddings...")
        dump_file(list(zip(strings, embeddings)), f"embeddings-{time.strftime('%Y%m%d-%H%M%S')}.json")
        print("Embeddings backed up.")
    
    # Cluster the embeddings
    data = np.array(embeddings)
    del embeddings
    print(f"Data shape: {data.shape}")
    print(f"Data head: {data[:5, :10]}")
    print("Clustering...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=CLUSTER_MIN_SIZE).fit(data)
    print("Clustering complete.")
    
    # Identify parent clusters of each string and each cluster
    clusters_pd = clusterer.condensed_tree_.to_pandas()
    min_id = int(min(clusters_pd.child.min(), clusters_pd.parent.min()) + 0.1)
    max_id = int(max(clusters_pd.child.max(), clusters_pd.parent.max()) + 0.1)
    assert min_id == 0 and len(clusters_pd) == max_id # Ensure that the IDs are contiguous and that it is a single tree without any disconnected nodes
    parent_mapping = [None] * (max_id + 1)
    for i, row in tqdm(clusters_pd.iterrows()):
        parent_mapping[int(row.child + 0.1)] = int(row.parent + 0.1)
    
    # Get the probabilities of each string and robustness of each cluster
    weights = [None] * min_id + [float(clusterer.probabilities_[i]) for i in range(len(strings))]
    weights += [None] * (max_id + 1 - len(weights))
    
    # Create summary mapping for strings
    summaries = [None] * min_id + strings + [None] * (max_id + 1 - len(strings))
    assert len(parent_mapping) == len(summaries) == len(weights)
    
    # Create strings mapping for lowest-level clusters
    sons: List[List[int]] = [[] for _ in range(max_id + 1)]
    cluster_sizes = [None for _ in range(max_id + 1)]
    queue, ptr = [], 0
    for i, string in tqdm(enumerate(strings)):
        label = parent_mapping[i + min_id]
        if cluster_sizes[label] is None:
            queue.append(label)
            cluster_sizes[label] = 0
        
        cluster_sizes[label] += 1
        cluster_sizes[i + min_id] = 1
        sons[label].append(i + min_id)
    
    # Create strings mapping for higher-level clusters with bottom-up BFS
    with tqdm(total=max_id - len(strings)) as pbar:
        while ptr < len(queue):
            pbar.update(1)
            label = queue[ptr]
            ptr += 1
            if len(sons[label]) > 5:
                cluster_element_probs = np.array([weights[i] for i in sons[label]])
                if not np.isnan(cluster_element_probs).all() and not np.isinf(cluster_element_probs).all():
                    cluster_element_probs = np.ones_like(cluster_element_probs)
                cluster_element_probs /= np.sum(cluster_element_probs)
                sons[label] = np.random.choice(
                    sons[label],
                    5,
                    replace=False,
                    p=cluster_element_probs,
                ).tolist()
            
            cluster_name = f"CLUSTER {label} ({cluster_sizes[label]}): {', '.join([summaries[i] for i in sons[label]])}"
            summaries[label] = cluster_name
            
            if parent_mapping[label] is not None:
                parent = parent_mapping[label]
                if cluster_sizes[parent] is None:
                    queue.append(parent)
                    cluster_sizes[parent] = 0
                
                cluster_sizes[parent] += cluster_sizes[label]
                sons[parent_mapping[label]].extend(sons[label])
                sons[label] = []
    
    # Backup cluster data
    print("Backing up clusters...")
    dump_file(
        [
            parent_mapping,
            cluster_sizes,
            summaries,
            weights,
        ],
        f"clusters-{time.strftime('%Y%m%d-%H%M%S')}.json",
    )
    
    assert all([cluster_sizes[i] is not None for i in range(max_id + 1)])
    assert all([summaries[i] is not None for i in range(max_id + 1)])
    assert cluster_sizes[len(strings)] == len(strings)
    return parent_mapping, cluster_sizes, summaries, weights, len(strings)

def cluster_concepts(samples: List[DataSample]) -> Tuple[List[DataSample], List[int], List[int], List[str], List[float], int]:
    """Hierarchically cluster concepts to identify common themes and topics.

    :param samples: List of DataSample objects.
    :type samples: List[DataSample]
    :return: Tuple of the samples with the `concepts` attribute replaced with the `concepts_breakdown` attribute, followed by the parent ID of each concept or cluster, followed by their sizes, followed by a list of strings representing the summary of each cluster or the content of each concept, followed by a list of floats representing the probability of membership to its assigned cluster / robustness of the cluster itself, followed by the id of the root cluster.
    :rtype: Tuple[List[DataSample], List[int], List[int], List[str], List[float], int]:
    """
    all_concepts = set()
    for sample in samples:
        if not sample.concepts:
            continue
        all_concepts.update([x for x in sample.concepts if isinstance(x, str) and x])
    
    all_concepts = list(all_concepts)
    parent_mapping, cluster_sizes, summaries, weights, root = cluster_strs(all_concepts)
    
    inv_mapping = {summary: i for i, summary in enumerate(summaries)}
    
    def is_legit(concept):
        if "dummy" in os.environ.get("EMBEDDINGS", ""):
            return isinstance(concept, str) and concept and concept in inv_mapping

        return isinstance(concept, str) and concept
        
    for sample in samples:
        if not sample.concepts:
            continue
        sample.concepts = [inv_mapping[concept] for concept in sample.concepts if is_legit(concept)]
        for key in sample.concepts_breakdown:
            sample.concepts_breakdown[key] = [inv_mapping[concept] for concept in sample.concepts_breakdown[key] if is_legit(concept)]
    
    return samples, parent_mapping, cluster_sizes, summaries, weights, root

def select_clusters(
    samples: List[DataSample], 
    cluster_parent: List[int],
    cluster_size: List[int],
    cluster_name: List[str],
    cluster_prob: List[float],
    root: int,
    min_size: int = CLUSTER_MIN_SIZE,
    step_multiplier: int = CLUSTER_STEP_MULTIPLIER_LOG2,
    **kwargs,
) -> Tuple[List[int], List[int]]:
    max_subtree_size = [0] * len(cluster_parent)
    for i, parent in tqdm(enumerate(cluster_parent)):
        if parent is not None:
            max_subtree_size[parent] = max(max_subtree_size[parent], cluster_size[i])
    
    def exp_order(n, multiplier=step_multiplier):
        return (int.bit_length(n) - 1) // multiplier
    
    selected_clusters = []
    sizes_counts = [0] * 32
    for i, sizes in tqdm(enumerate(zip(cluster_size, max_subtree_size))):
        self_size, child_size = sizes
        if self_size > min_size and exp_order(self_size) > exp_order(child_size):
            selected_clusters.append(i)
            sizes_counts[exp_order(self_size, 1)] += 1
    
    print(f"Selected {len(selected_clusters)} clusters out of {len(cluster_size)} total.")
    print(f"Cluster size distribution: {sizes_counts}")
    
    cluster_selected_parent = deepcopy(cluster_parent)
    is_selected = [False] * len(cluster_parent)
    for i in selected_clusters:
        is_selected[i] = True
    
    def get_nearest_selected_parent(i):
        if cluster_selected_parent[i] is not None and not is_selected[cluster_selected_parent[i]]:
            cluster_selected_parent[i] = get_nearest_selected_parent(cluster_selected_parent[i])
        return cluster_selected_parent[i]
    
    for i in range(len(cluster_selected_parent)):
        get_nearest_selected_parent(i)
    
    return selected_clusters, cluster_selected_parent

if __name__ == "__main__":
    print(cluster_strs(
        [
            "computational linguistics",
            "natural language processing",
            "machine learning",
            "deep learning",
            "artificial intelligence",
            "neural networks",
            "computer vision",
            "cat",
            "dog",
            "bird",
            "fish",
            "reptile",
            "mammal",
            "insect",
            "United States",
            "Canada",
            "Mexico",
            "United Nations",
            "European Union",
            "World Health Organization",
            "World Trade Organization",
            "World Bank",
            "Dogecoin",
            "Douglas Adams",
            "Isaac Asimov",
            "Ray Bradbury",
            "Arthur C. Clarke",
            "social media",
            "social network",
            "fake news",
            "misinformation",
            "disinformation",
            "a lie",
            "nice little lie",
        ]
    ))