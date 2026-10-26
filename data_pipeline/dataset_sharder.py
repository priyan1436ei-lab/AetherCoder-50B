"""High-throughput binary memory-mapped dataset writer and streaming shard reader."""

import json
import os
import struct
from typing import Iterator, List, Optional
import numpy as np


class BinaryDatasetWriter:
    """Writes tokenized integer arrays into memory-mapped .bin and .idx shards."""

    def __init__(self, output_dir: str, shard_size_tokens: int = 100_000_000, dtype: np.dtype = np.uint16):
        self.output_dir = output_dir
        self.shard_size_tokens = shard_size_tokens
        self.dtype = dtype
        os.makedirs(output_dir, exist_ok=True)
        self.current_shard_idx = 0
        self.current_tokens: List[int] = []
        self.total_tokens_written = 0

    def add_tokens(self, tokens: List[int]) -> None:
        """Appends tokens to current buffer and flushes shard if threshold reached."""
        self.current_tokens.extend(tokens)
        if len(self.current_tokens) >= self.shard_size_tokens:
            self._flush_shard()

    def _flush_shard(self) -> None:
        if not self.current_tokens:
            return

        shard_path = os.path.join(self.output_dir, f"shard_{self.current_shard_idx:05d}.bin")
        arr = np.array(self.current_tokens, dtype=self.dtype)
        with open(shard_path, "wb") as f:
            f.write(arr.tobytes())

        meta_path = os.path.join(self.output_dir, f"shard_{self.current_shard_idx:05d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "num_tokens": len(self.current_tokens),
                "dtype": str(self.dtype),
                "shard_id": self.current_shard_idx,
            }, f, indent=2)

        self.total_tokens_written += len(self.current_tokens)
        self.current_shard_idx += 1
        self.current_tokens = []

    def close(self) -> None:
        """Flushes remaining tokens to the final shard."""
        self._flush_shard()


class BinaryDatasetReader:
    """Zero-copy memory-mapped streaming dataset reader across binary shards."""

    def __init__(self, shard_paths: List[str], dtype: np.dtype = np.uint16):
        self.shard_paths = shard_paths
        self.dtype = dtype

    def iterate_blocks(self, block_size: int = 2048) -> Iterator[np.ndarray]:
        """Yields fixed-length token sequences for training."""
        for path in self.shard_paths:
            if not os.path.exists(path):
                continue
            mmap_arr = np.memmap(path, dtype=self.dtype, mode="r")
            try:
                num_blocks = len(mmap_arr) // block_size
                for i in range(num_blocks):
                    yield np.array(mmap_arr[i * block_size : (i + 1) * block_size])
            finally:
                if hasattr(mmap_arr, "_mmap") and mmap_arr._mmap is not None:
                    mmap_arr._mmap.close()
                del mmap_arr
