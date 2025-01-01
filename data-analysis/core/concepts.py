import json, warnings, random, time
import voyageai
import hdbscan
from tqdm import tqdm
from typing import List, Dict, Tuple, Mapping
from typeguard import check_type
from core.samples import DataSample
from core.templates import (
    concept_extraction_template,
)
from utils.log_utils import silence_decorator, dynamic_printing_decorator, dump_file
from utils.json_utils import extract_json_from_str
from utils.nlp_utils import simplify_text
import numpy as np
import pandas as pd


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

def cluster_strs(strings: List[str]) -> Tuple[List[int], List[int], List[str], List[float]]:
    """Hierarchically cluster strings to identify common themes and topics. Each node or cluster has a unique integer ID and a string summary, with the cluster summary being a few randomly selected strings from the cluster, weighted by probability.

    :param strings: List of strings to cluster.
    :type strings: List[str]
    :return: List of integers representing the parent ID of each string or cluster, followed by their sizes, followed by a list of strings representing the summary of each cluster or the content of each string, followed by a list of floats representing the probability of membership to its assigned cluster / robustness of the cluster itself.
    :rtype: Tuple[List[int], List[int], List[str], List[float]]
    """
    print(f"Embedding strings ({len(strings)} total)...")
    vo = voyageai.Client()
    batch_size = 128
    embeddings = []
    for i in tqdm(range(0, len(strings), batch_size)):
        for retry_count in range(500):
            time.sleep(1/1900)
            cur_emb = vo.embed(
                strings[i : i + batch_size],
                model="voyage-3-large",
                output_dimension=256,
            ).embeddings
            
            if not isinstance(cur_emb, list) or \
                len(cur_emb) != len(strings[i : i + batch_size]) or \
                np.isnan(cur_emb).any() or \
                np.isinf(cur_emb).any():
                
                if retry_count == 499:
                    print(f"Failed to embed strings after {retry_count} retries. Dumping incomplete embeddings ({len(embeddings)} out of {len(strings)})...")
                    dump_file(list(zip(strings, embeddings)), f"embeddings-incomplete-{time.strftime('%Y%m%d-%H%M%S')}.json")
                    raise ValueError("Failed to embed strings after 500 retries.")
                
                print(f"Retrying for the {retry_count}-th time...")
                time.sleep(1/100)
                continue
            
            embeddings.extend(cur_emb)
            break
    
    # Backup embedding data
    print("Backing up embeddings...")
    dump_file(list(zip(strings, embeddings)), f"embeddings-{time.strftime('%Y%m%d-%H%M%S')}.json")
    print("Embeddings backed up.")
    
    # Cluster the embeddings
    data = np.array(embeddings)
    print(f"Data shape: {data.shape}")
    print(f"Data head: {data[:5, :10]}")
    print("Clustering...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5).fit(data)
    print("Clustering complete.")
    
    # Identify parent clusters of each string and each cluster
    clusters_pd = clusterer.condensed_tree_.to_pandas()
    min_id = int(min(clusters_pd.child.min(), clusters_pd.parent.min()) + 0.1)
    max_id = int(max(clusters_pd.child.max(), clusters_pd.parent.max()) + 0.1)
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
    return parent_mapping, cluster_sizes, summaries, weights

def cluster_concepts(samples: List[DataSample]) -> Tuple[List[DataSample], List[int], List[int], List[str], List[float]]:
    """Hierarchically cluster concepts to identify common themes and topics.

    :param samples: List of DataSample objects.
    :type samples: List[DataSample]
    :return: Tuple of the samples with the `concepts` attribute replaced with the `concepts_breakdown` attribute, followed by the parent ID of each concept or cluster, followed by their sizes, followed by a list of strings representing the summary of each cluster or the content of each concept, followed by a list of floats representing the probability of membership to its assigned cluster / robustness of the cluster itself.
    :rtype: Tuple[List[DataSample], List[int], List[int], List[str], List[float]]:
    """
    all_concepts = set()
    for sample in samples:
        if not sample.concepts:
            continue
        all_concepts.update(sample.concepts)
    
    all_concepts = list(all_concepts)
    parent_mapping, cluster_sizes, summaries, weights = cluster_strs(all_concepts)
    
    inv_mapping = {summary: i for i, summary in enumerate(summaries)}
    for sample in samples:
        if not sample.concepts:
            continue
        sample.concepts = [inv_mapping[concept] for concept in sample.concepts]
        for key in sample.concepts_breakdown:
            sample.concepts_breakdown[key] = [inv_mapping[concept] for concept in sample.concepts_breakdown[key]]
    
    return samples, parent_mapping, cluster_sizes, summaries, weights
    


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