"""Synthetic coding data generator for pretraining, SFT, and agent trajectory fine-tuning."""

import random
from typing import Dict, List


ALGORITHMS = [
    ("quicksort", "Sorts an array using divide-and-conquer partitioning with an in-place pivot."),
    ("merge_sort", "Sorts a sequence stably by recursively splitting and merging sorted halves."),
    ("dijkstra", "Finds the shortest path from a starting vertex to all other vertices in a weighted graph."),
    ("breadth_first_search", "Traverses a graph or tree level by level using a FIFO queue."),
    ("depth_first_search", "Explores as deep as possible along each branch before backtracking."),
    ("trie_autocomplete", "Prefix tree data structure for high-performance string lookup and auto-completion."),
    ("lru_cache", "Fixed-capacity cache evicting the least recently accessed item via doubly linked list and hashmap."),
    ("bloom_filter", "Space-efficient probabilistic data structure testing set membership with zero false negatives."),
    ("topological_sort", "Linear ordering of vertices in a Directed Acyclic Graph (DAG) using Kahn's algorithm."),
    ("rabin_karp", "String searching algorithm using rolling polynomial hashing for substring matching."),
]

LANGUAGES = ["python", "typescript", "rust", "go", "cpp", "sql"]


class SyntheticDataGenerator:
    """Generates synthetic instruction-following, debugging, and agent training samples."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_algorithm_prompt(self, algo_name: str, description: str, language: str) -> Dict[str, str]:
        prompt = f"Implement {algo_name} in {language.capitalize()}.\nSpecification: {description}\nEnsure optimal time complexity, idiomatic syntax, and clear docstrings."
        
        # Skeleton completion
        code = f"""// {algo_name.upper()} Implementation in {language.capitalize()}
// {description}
"""
        return {
            "type": "algorithm",
            "language": language,
            "prompt": prompt,
            "completion": code,
        }

    def generate_batch(self, count: int = 50) -> List[Dict[str, str]]:
        """Generates a batch of synthetic coding training samples."""
        batch = []
        for i in range(count):
            algo_name, desc = self.rng.choice(ALGORITHMS)
            lang = self.rng.choice(LANGUAGES)
            sample = self.generate_algorithm_prompt(algo_name, desc, lang)
            sample["id"] = f"synth-{i:05d}"
            batch.append(sample)
        return batch
