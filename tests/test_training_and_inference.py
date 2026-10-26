"""Integration tests for Training Engine, SFT, DPO, Inference Generator, and Benchmarks."""

import os
import pytest
import tempfile
import torch

from model.config import ModelConfig
from model.model import CodingCausalLM
from tokenizer.code_tokenizer import create_default_code_tokenizer
from training.pretraining.train_engine import PretrainingEngine
from training.sft.train_sft import SFTTrainer, SFTDatasetPreparer
from training.preference_optimization.train_dpo import DPOTrainer
from inference.generator import CodeGenerator
from benchmarks.humaneval_runner import HumanEvalRunner, EvalProblem


@pytest.fixture
def mini_model():
    cfg = ModelConfig(
        model_type="moe_causal_lm",
        vocab_size=500,
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=256,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
    )
    return CodingCausalLM(cfg)


def test_pretraining_engine_step_and_checkpoint(mini_model):
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = PretrainingEngine(
            model=mini_model,
            learning_rate=1e-3,
            checkpoint_dir=tmp_dir,
            max_steps=10,
        )

        input_ids = torch.randint(0, 500, (2, 8))
        labels = input_ids.clone()

        step_res = engine.train_step(input_ids, labels)
        assert "loss" in step_res
        assert step_res["loss"] > 0.0

        grad_norm, lr = engine.optimizer_step()
        assert grad_norm >= 0.0
        assert lr > 0.0

        # Test Checkpoint Save and Recovery
        ckpt_path = engine.save_checkpoint(step=1)
        assert os.path.exists(ckpt_path)

        restored_step = engine.load_checkpoint(ckpt_path)
        assert restored_step == 1


def test_sft_dataset_masking_and_trainer(mini_model):
    tok = create_default_code_tokenizer()
    preparer = SFTDatasetPreparer(tokenizer=tok, max_seq_len=128)

    sample = preparer.format_and_tokenize(
        prompt="Write a function that adds two numbers.",
        completion="def add(a, b):\n    return a + b",
    )

    # Prompt portion must be masked with -100
    labels = sample["labels"].tolist()
    assert -100 in labels
    assert labels[0] == -100

    trainer = SFTTrainer(model=mini_model, learning_rate=1e-3)
    loss = trainer.train_step(
        sample["input_ids"][:16].unsqueeze(0) % 500,
        sample["labels"][:16].unsqueeze(0) % 500,
    )
    assert loss > 0.0


def test_dpo_trainer_step(mini_model):
    ref_model = CodingCausalLM(mini_model.config)
    trainer = DPOTrainer(policy_model=mini_model, reference_model=ref_model, beta=0.1)

    chosen_ids = torch.randint(0, 500, (2, 8))
    chosen_labels = chosen_ids.clone()
    rejected_ids = torch.randint(0, 500, (2, 8))
    rejected_labels = rejected_ids.clone()

    loss, ch_r, rej_r = trainer.train_step(chosen_ids, chosen_labels, rejected_ids, rejected_labels)
    assert not torch.isnan(torch.tensor(loss))


def test_inference_generator_stream(mini_model):
    tok = create_default_code_tokenizer()
    generator = CodeGenerator(model=mini_model, tokenizer=tok)

    stream_chunks = list(generator.generate_stream(prompt="def main():", max_new_tokens=4))
    assert len(stream_chunks) > 0


def test_humaneval_runner():
    runner = HumanEvalRunner(timeout_seconds=2)
    problem = EvalProblem(
        task_id="test/0",
        prompt="def return_forty_two():",
        canonical_solution="    return 42",
        test="def check(candidate):\n    assert candidate() == 42",
        entry_point="return_forty_two",
    )

    # Correct solution
    correct_res = runner.check_solution(problem, "    return 42")
    assert correct_res.passed is True

    # Incorrect solution
    wrong_res = runner.check_solution(problem, "    return 0")
    assert wrong_res.passed is False
