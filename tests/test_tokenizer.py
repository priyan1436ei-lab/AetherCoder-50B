"""Unit tests for CodeTokenizer and Data Pipeline."""

import os
import tempfile
import numpy as np
import pytest
from tokenizer.code_tokenizer import CodeTokenizer, create_default_code_tokenizer
from tokenizer.special_tokens import FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE
from data_pipeline.secret_scrubber import SecretScrubber, REDACTION_TOKEN
from data_pipeline.license_filter import LicenseFilter
from data_pipeline.fim_transform import FIMTransform
from data_pipeline.dataset_sharder import BinaryDatasetWriter, BinaryDatasetReader


def test_tokenizer_preserves_indentation_and_code_syntax():
    tokenizer = create_default_code_tokenizer()
    sample_code = """def fibonacci(n: int) -> int:
    # Check base cases (Tabs & Spaces preserved)
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""
    tokens = tokenizer.encode(sample_code)
    assert len(tokens) > 0

    reconstructed = tokenizer.decode(tokens)
    assert reconstructed == sample_code, "Decoded text must preserve exact indentation and whitespace."


def test_tokenizer_unicode_and_special_characters():
    tokenizer = create_default_code_tokenizer()
    complex_code = """// Unicode Test: こんにちは世界 | 🚀 Antigravity | π ≈ 3.14159
fn calculate_area(r: f64) -> f64 {
    std::f64::consts::PI * r * r
}
"""
    tokens = tokenizer.encode(complex_code)
    # Check bounds
    assert all(0 <= tid < tokenizer.vocab_size for tid in tokens)
    reconstructed = tokenizer.decode(tokens)
    assert reconstructed == complex_code


def test_tokenizer_fim_formatting():
    tokenizer = create_default_code_tokenizer()
    prefix = "def calculate_area(radius):\n    "
    suffix = "\n    return area"
    middle = "area = 3.14159 * radius ** 2"

    fim_text = tokenizer.format_fim(prefix=prefix, suffix=suffix, middle=middle, mode="PSM")
    assert FIM_PREFIX in fim_text
    assert FIM_SUFFIX in fim_text
    assert FIM_MIDDLE in fim_text
    assert middle in fim_text


def test_secret_scrubber_redacts_credentials():
    scrubber = SecretScrubber()
    dirty_code = """
    # Connect to AWS S3
    AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
    GITHUB_TOKEN = "ghp_1234567890abcdefghijklmnopqrstuvwxyzAB"
    
    def connect():
        pass
    """
    cleaned, findings = scrubber.scrub(dirty_code)
    assert "AKIAIOSFODNN7EXAMPLE" not in cleaned
    assert "ghp_1234567890abcdefghijklmnopqrstuvwxyzAB" not in cleaned
    assert REDACTION_TOKEN in cleaned
    assert len(findings) == 2


def test_license_filter_permissibility():
    filter = LicenseFilter()
    assert filter.is_permissible("MIT") is True
    assert filter.is_permissible("Apache-2.0") is True
    assert filter.is_permissible("GPL-3.0") is False

    mit_text = "Permission is hereby granted, free of charge, to any person obtaining a copy..."
    assert filter.is_permissible(license_identifier=None, license_text=mit_text) is True


def test_fim_transform_tokens():
    fim = FIMTransform(fim_rate=1.0, spm_rate=0.5)
    dummy_tokens = list(range(100, 150))
    transformed, was_applied = fim.transform_tokens(dummy_tokens)
    assert was_applied is True
    assert len(transformed) == len(dummy_tokens) + 4


def test_dataset_sharder_overflow_detection():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # uint16 holds up to 65535
        writer = BinaryDatasetWriter(output_dir=tmp_dir, shard_size_tokens=100, dtype=np.uint16)
        valid_tokens = [10, 20, 30, 65535]
        writer.add_tokens(valid_tokens)

        # Attempt to insert 65536 into uint16
        with pytest.raises(ValueError, match="exceeds capacity"):
            writer.add_tokens([65536])
