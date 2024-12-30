import json, warnings
from tqdm import tqdm
from typing import List, Dict
from typeguard import check_type
from core.samples import DataSample
from ProgressGym import Data, Model
from core.templates import (
    concept_extraction_template,
)

from utils.log_utils import silence_decorator, dynamic_printing_decorator
from utils.json_utils import extract_json_from_str

def extract_concepts(samples: List[DataSample], extractor: Model, max_retries: int = 5) -> List[DataSample]:
    """Extract concepts (list of strings) from the conversation content of each sample.

    :param samples: List of DataSample objects.
    :type samples: List[DataSample]
    
    :param extractor: LLM used for extracting concepts.
    :type extractor: Model
    
    :param max_retries: Number of retries allowed for failed extraction, possibly due to LLM mishandling formatting.
    :type max_retries: int
    
    :return: List of DataSample objects with the extracted concepts added to the `concepts` attribute. Operation may be done in place.
    :rtype: List[DataSample]
    """
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
    for sample, dialogue in zip(samples, extraction_output.all_passages()):
        sample: DataSample
        try:
            sample.concepts_breakdown = check_type(
                extract_json_from_str(dialogue["predict"]),
                Dict[str, List[str]],
            )
            sample.concepts = list(set(
                sample.concepts_breakdown["user_concepts_explicit"],
                sample.concepts_breakdown["user_concepts_related"],
                sample.concepts_breakdown["assistant_concepts_explicit"],
                sample.concepts_breakdown["assistant_concepts_related"],
            ))
            successful_count += 1
        except:
            error_count += 1
    
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


def get_concepts(samples: List[DataSample], extractor: Model, max_retries: int = 5) -> List[DataSample]:
    samples = extract_concepts(samples, extractor, max_retries)
    return samples