# AetherCoder-50B Dataset & Tokenization Guide

This document describes the data pipeline, tokenization architecture, binary sharding format, and artifact manifest generation.

---

## 1. Tokenizer Architecture

AetherCoder utilizes a Byte-Pair Encoding (BPE) tokenizer optimized for software engineering:
- **Vocabulary Size**: 65,536 tokens.
- **Whitespace & Indentation**: Exact whitespace and indentation sequences are preserved without collapsing tabs or multiple spaces.
- **Unicode & Line Endings**: Preserves UTF-8 code comments, multi-language identifier names, and CRLF / LF line endings.
- **Special Tokens**:
  - `<|bos|>`: Beginning of sequence (`1`)
  - `<|eos|>`: End of sequence (`2`)
  - `<|pad|>`: Padding token (`0`)
  - `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`: Fill-in-the-Middle tokens.
  - `<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`: Chat/agent role delimiters.

---

## 2. Dataset Pipeline & Corpus

The repository includes curated algorithms and full production modules across 16 programming languages:
- **Languages**: Python, TypeScript, JavaScript, Rust, Go, C++, C, Java, Kotlin, Swift, Scala, Ruby, PHP, C#, SQL, Shell.
- **Algorithmic Domains**:
  - Multi-file repository reasoning and import graphs
  - Unit test generation and test harness mocks
  - Cybersecurity defenses (SSRF DNS pinning, SQL injection mitigation)
  - Distributed concurrency (Raft state machines, Go channels, Rust lock-free ring buffers)
  - Data structures (Tries, Segment Trees, Bloom Filters, LSM-tree memtables)
  - Memory management (C slab allocators, arena allocators)

---

## 3. Sharding & Memory-Mapped Storage

To allow large-scale training without loading datasets entirely into RAM:
1. **Binary Storage Format**: Tokens are packed contiguously into binary `.bin` shard files.
2. **Explicit Data Types**: Stored as `uint32` (or `uint16` with overflow detection).
   - Any token ID exceeding the data type maximum triggers an immediate exception, preventing silent token corruption.
3. **Memory-Mapped Reader**: `BinaryDatasetReader` reads shards on-demand via `np.memmap`, keeping memory footprint negligible even with gigabyte-scale corpora.
4. **Content Deduplication**: Shard builders calculate SHA-256 hashes of code snippets to eliminate exact duplicates across repositories.
5. **Data Splits**: Partitions data into 80% train, 10% validation, and 10% test splits without cross-split leakage.

---

## 4. Reproducible Dataset Preparation

To generate binary shards, deduplicate, and export the dataset manifest:

```bash
# Generate 200 diverse samples (~213k tokens) into binary shards
python scripts/prepare_dataset.py \
  --output_dir data/shards \
  --samples 200 \
  --vocab_size 65536 \
  --max_seq_len 1024
```

### Manifest Schema (`data/dataset_manifest.json`)
The preparation script produces a JSON manifest documenting:
- `manifest_version`
- `timestamp`
- `splits`: Record counts and token counts for train, validation, and test.
- `shards`: File paths, split associations, record counts, and token counts.
- `deduplication`: Total inputs, unique kept records, and duplicate records pruned.
- `vocab_size` and `storage_dtype` (`uint32`).
