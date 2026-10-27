"""Dataset Builder creating structured artifacts for all 8 training stages."""

import json
import os
from typing import Dict, List
from data_pipeline.large_scale_corpus import MULTILINGUAL_CORPUS
from data_pipeline.multilingual_code_dataset import CURATED_DATASET


def build_all_stage_datasets(base_dir: str = "./data"):
    """Creates dedicated dataset directories and JSONL files for each LLM training stage."""
    sft_dir = os.path.join(base_dir, "sft")
    dpo_dir = os.path.join(base_dir, "dpo")
    agent_dir = os.path.join(base_dir, "agent")
    fim_dir = os.path.join(base_dir, "fim")
    multi_file_dir = os.path.join(base_dir, "multi_file")
    unit_test_dir = os.path.join(base_dir, "unit_test")


    for d in [sft_dir, dpo_dir, agent_dir, fim_dir, multi_file_dir, unit_test_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Build SFT Instruction-Following Dataset
    sft_file = os.path.join(sft_dir, "sft_instructions.jsonl")
    sft_count = 0
    with open(sft_file, "w", encoding="utf-8") as f:
        for item in MULTILINGUAL_CORPUS:
            record = {
                "id": f"sft-{item['language']}-{sft_count:04d}",
                "language": item["language"],
                "prompt": item["prompt"],
                "completion": item["code"],
                "domain": item["domain"],
            }
            f.write(json.dumps(record) + "\n")
            sft_count += 1
        for item in CURATED_DATASET:
            if item.prompt:
                record = {
                    "id": f"sft-{item.sample_id}",
                    "language": item.language,
                    "prompt": item.prompt,
                    "completion": item.code,
                    "domain": item.domain,
                }
                f.write(json.dumps(record) + "\n")
                sft_count += 1

    # 2. Build DPO Preference Dataset (Chosen vs Rejected)
    dpo_file = os.path.join(dpo_dir, "dpo_pairs.jsonl")
    dpo_count = 0
    with open(dpo_file, "w", encoding="utf-8") as f:
        # Debugging and security pairs are natural chosen/rejected pairs
        for item in CURATED_DATASET:
            if item.target_fix:
                record = {
                    "id": f"dpo-{item.sample_id}",
                    "language": item.language,
                    "prompt": item.prompt or "Fix this code.",
                    "chosen": item.target_fix,
                    "rejected": item.code,
                    "reasoning": f"Remediates vulnerability/bug in {item.sample_id}",
                }
                f.write(json.dumps(record) + "\n")
                dpo_count += 1

    # 3. Build Agent Trajectory Dataset
    agent_file = os.path.join(agent_dir, "agent_trajectories.jsonl")
    agent_count = 0
    with open(agent_file, "w", encoding="utf-8") as f:
        for item in CURATED_DATASET:
            if item.domain == "agent_trajectory":
                record = {
                    "id": f"agent-{item.sample_id}",
                    "trajectory": item.code,
                }
                f.write(json.dumps(record) + "\n")
                agent_count += 1

    # 4. Build FIM (Fill-in-the-Middle) Dataset
    fim_file = os.path.join(fim_dir, "fim_pairs.jsonl")
    fim_count = 0
    with open(fim_file, "w", encoding="utf-8") as f:
        for item in MULTILINGUAL_CORPUS:
            lines = item["code"].splitlines(keepends=True)
            if len(lines) >= 6:
                mid_idx = len(lines) // 2
                prefix = "".join(lines[:mid_idx - 1])
                middle = "".join(lines[mid_idx - 1:mid_idx + 2])
                suffix = "".join(lines[mid_idx + 2:])
                record = {
                    "id": f"fim-{item['language']}-{fim_count:04d}",
                    "language": item["language"],
                    "prefix": prefix,
                    "middle": middle,
                    "suffix": suffix,
                }
                f.write(json.dumps(record) + "\n")
                fim_count += 1

    # 5. Build Multi-File Repository Reasoning Dataset
    multi_file_path = os.path.join(multi_file_dir, "multi_file_reasoning.jsonl")
    multi_file_count = 0
    with open(multi_file_path, "w", encoding="utf-8") as f:
        for item in CURATED_DATASET:
            if item.domain == "multi_file_reasoning":
                record = {
                    "id": f"multifile-{item.sample_id}",
                    "language": item.language,
                    "prompt": item.prompt,
                    "code": item.code,
                }
                f.write(json.dumps(record) + "\n")
                multi_file_count += 1

    # 6. Build Unit Test Generation Dataset
    unit_test_path = os.path.join(unit_test_dir, "unit_test_suites.jsonl")
    unit_test_count = 0
    with open(unit_test_path, "w", encoding="utf-8") as f:
        for item in CURATED_DATASET:
            if item.domain == "unit_test" or item.tests:
                record = {
                    "id": f"test-{item.sample_id}",
                    "language": item.language,
                    "prompt": f"Write unit tests for: {item.prompt or item.sample_id}",
                    "implementation": item.code,
                    "tests": item.tests or item.code,
                }
                f.write(json.dumps(record) + "\n")
                unit_test_count += 1

    return {
        "sft_samples": sft_count,
        "dpo_samples": dpo_count,
        "agent_trajectories": agent_count,
        "fim_samples": fim_count,
        "multi_file_samples": multi_file_count,
        "unit_test_samples": unit_test_count,
    }

