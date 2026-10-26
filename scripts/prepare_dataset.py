"""Dataset preparation, sanitization, tokenization, and binary sharding pipeline."""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from data_pipeline.license_filter import LicenseFilter
from data_pipeline.secret_scrubber import SecretScrubber
from data_pipeline.fim_transform import FIMTransform
from data_pipeline.dataset_sharder import BinaryDatasetWriter
from data_pipeline.multilingual_code_dataset import CURATED_DATASET, save_curated_dataset_jsonl
from data_pipeline.large_scale_corpus import MULTILINGUAL_CORPUS
from data_pipeline.synthetic_generator import SyntheticDataGenerator
from data_pipeline.dataset_builder import build_all_stage_datasets
from tokenizer.code_tokenizer import create_default_code_tokenizer


def prepare_training_dataset(output_dir: str = "./data/shards", raw_dir: str = "./data/raw"):
    print("=" * 70)
    print("CODING LLM: COMPREHENSIVE DATASET PREPARATION & SHARDING ENGINE")
    print("=" * 70)

    # 1. Export Raw Curated & Multilingual Corpus
    print("[1/6] Exporting curated multilingual datasets across 16 languages...")
    raw_curated_path = os.path.join(raw_dir, "curated_multilingual.jsonl")
    curated_count = save_curated_dataset_jsonl(raw_curated_path)
    print(f"      Saved {curated_count} curated samples to {raw_curated_path}")

    # 2. Build Stage-Specific Datasets (SFT, DPO, Agent, FIM)
    print("[2/6] Generating dedicated stage datasets (SFT, DPO, Agent Trajectories, FIM)...")
    stage_metrics = build_all_stage_datasets(base_dir="./data")
    print(f"      SFT Instructions:    {stage_metrics['sft_samples']} pairs")
    print(f"      DPO Preferences:     {stage_metrics['dpo_samples']} pairs")
    print(f"      Agent Trajectories:  {stage_metrics['agent_trajectories']} traces")
    print(f"      FIM Completions:     {stage_metrics['fim_samples']} snippets")

    # 3. Generate Synthetic Coding Batches
    print("[3/6] Synthesizing algorithms and data structure training samples...")
    synth_gen = SyntheticDataGenerator(seed=2026)
    synth_samples = synth_gen.generate_batch(count=150)
    print(f"      Synthesized {len(synth_samples)} samples.")

    # 4. Initialize Sanitization & Sharding
    print("[4/6] Initializing Secret Scrubber, License Filter & FIM Transformer...")
    scrubber = SecretScrubber()
    lic_filter = LicenseFilter()
    tokenizer = create_default_code_tokenizer()
    fim = FIMTransform(
        fim_rate=0.5,
        prefix_token_id=tokenizer.fim_prefix_id or 5,
        suffix_token_id=tokenizer.fim_suffix_id or 6,
        middle_token_id=tokenizer.fim_middle_id or 7,
    )

    writer = BinaryDatasetWriter(output_dir=output_dir, shard_size_tokens=100_000)

    # 5. Process and Tokenize Multilingual Corpus
    print("[5/6] Scrubbing, FIM transforming, and tokenizing full 16-language corpus...")
    total_tokens = 0
    clean_sample_count = 0

    # Process 16-language corpus
    for item in MULTILINGUAL_CORPUS:
        clean_code, _ = scrubber.scrub(item["code"])
        text = f"// Language: {item['language']} | Domain: {item['domain']}\n// Task: {item['prompt']}\n{clean_code}"
        tokens = tokenizer.encode(text, add_special_tokens=True)
        transformed_tokens, _ = fim.transform_tokens(tokens)
        writer.add_tokens(transformed_tokens)
        total_tokens += len(transformed_tokens)
        clean_sample_count += 1

    # Process Curated Dataset
    for sample in CURATED_DATASET:
        clean_code, findings = scrubber.scrub(sample.code)
        full_text = f"// Language: {sample.language} | Domain: {sample.domain}\n"
        if sample.prompt:
            full_text += f"// Task: {sample.prompt}\n"
        full_text += clean_code

        if sample.target_fix:
            clean_fix, _ = scrubber.scrub(sample.target_fix)
            full_text += f"\n\n// Solution & Repair:\n{clean_fix}"

        if sample.tests:
            clean_tests, _ = scrubber.scrub(sample.tests)
            full_text += f"\n\n// Verified Test Suite:\n{clean_tests}"

        tokens = tokenizer.encode(full_text, add_special_tokens=True)
        transformed_tokens, _ = fim.transform_tokens(tokens)
        writer.add_tokens(transformed_tokens)
        total_tokens += len(transformed_tokens)
        clean_sample_count += 1

    # Process Synthetic Samples
    for sample in synth_samples:
        clean_code, _ = scrubber.scrub(sample["completion"])
        text = f"// Language: {sample['language']}\n// Task: {sample['prompt']}\n{clean_code}"
        tokens = tokenizer.encode(text, add_special_tokens=True)
        transformed_tokens, _ = fim.transform_tokens(tokens)
        writer.add_tokens(transformed_tokens)
        total_tokens += len(transformed_tokens)
        clean_sample_count += 1

    # 6. Flush Final Shard
    writer.close()
    print("[6/6] Finalizing binary memory-mapped shards...")
    print("=" * 70)
    print(f"[SUCCESS] Dataset preparation complete!")
    print(f"          Processed Samples: {clean_sample_count}")
    print(f"          Total Tokens:      {total_tokens:,}")
    print(f"          Shards Location:   {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    prepare_training_dataset()
