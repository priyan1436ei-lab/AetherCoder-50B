"""Unit tests for the Multilingual Dataset Pipeline and Synthetic Generator."""

import os
import tempfile
import pytest

from data_pipeline.multilingual_code_dataset import CURATED_DATASET, save_curated_dataset_jsonl
from data_pipeline.synthetic_generator import SyntheticDataGenerator
from data_pipeline.dataset_sharder import BinaryDatasetReader
from scripts.prepare_dataset import prepare_training_dataset


def test_curated_dataset_covers_all_domains():
    domains = {sample.domain for sample in CURATED_DATASET}
    required_domains = {
        "general",
        "instruction_following",
        "debugging",
        "security",
        "agent_trajectory",
        "multi_file_reasoning",
        "unit_test",
        "code_completion",
    }
    for req in required_domains:
        assert req in domains, f"Missing domain {req} in curated dataset."
    assert len(CURATED_DATASET) >= 18, "Expected at least 18 curated samples"



def test_synthetic_generator_batch():
    gen = SyntheticDataGenerator(seed=123)
    batch = gen.generate_batch(count=10)
    assert len(batch) == 10
    for sample in batch:
        assert "prompt" in sample
        assert "completion" in sample
        assert "language" in sample


def test_dataset_preparation_pipeline_run():
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir = os.path.join(tmp_dir, "raw")
        shards_dir = os.path.join(tmp_dir, "shards")

        prepare_training_dataset(output_dir=shards_dir, raw_dir=raw_dir)

        # Check raw JSONL exists
        assert os.path.exists(os.path.join(raw_dir, "curated_multilingual.jsonl"))

        # Check binary shard exists
        bin_files = [f for f in os.listdir(shards_dir) if f.endswith(".bin")]
        assert len(bin_files) > 0

        # Verify reader can iterate blocks
        reader = BinaryDatasetReader(shard_paths=[os.path.join(shards_dir, bin_files[0])])
        blocks = list(reader.iterate_blocks(block_size=32))
        assert len(blocks) > 0
        assert len(blocks[0]) == 32
