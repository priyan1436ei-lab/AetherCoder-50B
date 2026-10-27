# AetherCoder-50B Repository Audit Report

**Date of Audit:** October 2026  
**Auditor:** DeepMind Antigravity AI Engineering Suite  
**Repository:** AetherCoder-50B (priyan1436ei-lab/AetherCoder-50B)  
**Target:** Turn repository into a verified, reproducible, secure coding-model research project.

---

## 1. Current Repository Structure

```
AetherCoder-50B/
├── .github/                   # [MISSING] CI/CD workflows
├── api/                       # FastAPI inference server & OpenAI-compatible schemas
│   ├── schemas.py             # Pydantic v2 schemas for chat completions
│   ├── security.py            # API key auth & rate limiting
│   └── server.py              # Application entrypoint & SSE streaming
├── benchmarks/                # Evaluation runners
│   ├── humaneval_runner.py    # Python code execution test harness
│   └── metrics.py             # pass@k calculation functions
├── checkpoints/               # Binary checkpoints (gitignored)
│   └── prototype_150m/        # 150M prototype checkpoint artifacts
├── coding_agent/              # ReAct Coding Agent
│   ├── agent.py               # Autonomous repair loop & error diagnosis
│   ├── repo_indexer.py        # AST and symbol indexer for multi-file context
│   ├── sandbox/               # Sandboxed boundary enforcement (PermissionGuard)
│   └── tools/                 # FileOps, Patcher, and SandboxedTerminal
├── configs/                   # Architecture & Training configs
│   ├── model/                 # 50B dense, 50B MoE, and 150M prototype YAMLs
│   └── training/              # Local prototype and FSDP distributed YAMLs
├── data/                      # Dataset directories
│   ├── agent/                 # Agent trajectory traces
│   ├── dpo/                   # DPO preference pairs (chosen/rejected)
│   ├── fim/                   # Fill-in-the-Middle tokenized pairs
│   ├── multi_file/            # Multi-file repository reasoning sets
│   ├── raw/                   # Raw JSONL datasets
│   ├── sft/                   # Supervised fine-tuning pairs
│   ├── shards/                # Memory-mapped binary shards (.bin)
│   └── unit_test/             # Generated unit test suites
├── data_pipeline/             # Preprocessing & Tokenization pipeline
│   ├── dataset_builder.py     # Stage-specific dataset partitioner
│   ├── dataset_sharder.py     # Binary memory-mapped shard reader/writer
│   ├── fim_transform.py       # Fill-in-the-Middle (PSM/SPM) data augmenter
│   ├── large_scale_corpus.py  # 32 enterprise multilingual implementations (16 langs)
│   ├── license_filter.py      # Permissive license detector (MIT, Apache, BSD)
│   ├── multilingual_code_dataset.py # 18 curated samples across all 8 domains
│   ├── secret_scrubber.py     # Credential and API key scrubber
│   └── synthetic_generator.py # Algorithmic code synthesis engine
├── docs/                      # [NEW] Documentation guides & specifications
├── frontend/                  # Web Developer Studio
│   ├── app.js                 # SSE client, telemetry polling, and chat UI
│   ├── index.html             # Developer Studio interface
│   └── styles.css             # Glassmorphic dark-mode styling
├── inference/                 # KV-cache and autoregressive generation
│   ├── generator.py           # Streaming token generator with nucleus sampling
│   └── kv_cache.py            # Dynamic KV cache manager
├── model/                     # Core PyTorch Transformer Architecture
│   ├── attention/             # Grouped-Query Attention (GQA) & RoPE
│   ├── feedforward/           # SwiGLU & Sparse MoE with shared experts
│   ├── losses/                # Load balancing auxiliary loss & router z-loss
│   ├── norm/                  # RMSNorm with configurable epsilon
│   ├── config.py              # ModelConfig dataclass
│   ├── model.py               # CodingCausalLM decoder container
│   └── transformer_block.py   # Transformer decoder block
├── scripts/                   # CLI runner utilities
│   ├── export_quantized.py    # INT8/INT4 quantization exporter
│   ├── prepare_dataset.py     # Sharding and tokenization pipeline runner
│   ├── run_prototype_train.py # Single-device pretraining runner
│   └── run_sft_train.py       # Supervised fine-tuning runner
├── tests/                     # Pytest automated test suite
├── tokenizer/                 # Byte-Level BPE code tokenizer
│   ├── code_tokenizer.py      # BPE tokenizer with FIM special tokens
│   └── special_tokens.py      # Definition of control & FIM tokens
├── .gitignore                 # Artifact and checkpoint exclusions
├── LICENSE                    # Apache 2.0 License
├── requirements.txt           # [MISSING] Root dependency manifest
└── SECURITY.md                # Security policy and sandbox boundary model
```

---

## 2. Implemented vs. Partially Implemented vs. Missing Features

| Component | Status | Detailed Evaluation |
| :--- | :--- | :--- |
| **MoE Architecture** | **Partially Implemented** | MoE layers, router gating, and shared experts exist, but router gradients when experts receive 0 tokens needed defensive handling, and exact dynamic parameter counting was hardcoded. |
| **KV Cache & RoPE** | **Bugs Identified** | During cached generation in `generator.py`, `position_ids` was not passed, defaulting to `0` for every decoded token. RoPE cache expansion was bounded. Attention mask support was missing from `CodingCausalLM.forward`. |
| **Model Config Validation** | **Partially Implemented** | `ModelConfig` lacked mathematical checks for divisibility of hidden dimensions and attention heads, KV-head ratio, and expert bounds before allocating memory. |
| **Pretraining Engine** | **Partially Implemented** | Checkpointing was non-atomic (interrupted writes corrupted files). Scheduler and RNG states were not preserved in checkpoints. Mixed precision GradScaler was missing. |
| **API Server & Security**| **Partially Implemented** | Permitted unauthenticated requests if `api_key` was unset. CORS was wide open (`*`). Model IDs in requests were unvalidated. Advertised 50B checkpoint even when nonexistent. |
| **Autonomous Coding Agent**| **Partially Implemented** | Basic ReAct flow existed, but lacked multi-language AST indexing, test failure diagnosis, and surgical patch rollback on regression. |
| **HumanEval Benchmark** | **Partially Implemented** | Temporary files were created with `delete=False` and never unlinked, leaking disk space. Process execution was unconfined. Hardcoded SOTA claims in README were unverified. |
| **Dependencies & CI** | **Missing** | No root `requirements.txt`, no `pyproject.toml`, and no `.github/workflows/ci.yml`. |
| **Documentation & Audits**| **Missing** | Missing `docs/REPOSITORY_AUDIT.md`, `docs/ARCHITECTURE.md`, `docs/TRAINING_GUIDE.md`, `docs/DATASET_GUIDE.md`, `docs/API.md`, `docs/EVALUATION.md`. |

---

## 3. Confirmed Bugs with Evidence

### Bug 1: Position ID Reset During Incremental KV-Cache Decoding
* **Location:** [inference/generator.py:L114](file:///g:/My%20Drive/coding%20llm/inference/generator.py#L114) and [model/model.py:L77-L79](file:///g:/My%20Drive/coding%20llm/model/model.py#L77-L79)
* **Evidence:** In `generator.py`, `self.model(input_ids=next_input, kv_caches=kv_caches, use_cache=True)` omits `position_ids`. In `model.py`, `position_ids` defaults to `torch.arange(seq_len)` which is `0` when `seq_len=1`.
* **Impact:** Every incrementally generated token receives RoPE angle for position 0, causing token-by-token cached decoding to mathematically diverge from full-sequence decoding.
* **Remediation:** Calculate `past_length = kv_caches[0][0].shape[2]` and construct `position_ids = past_length + torch.arange(seq_len)`.

### Bug 2: Missing `attention_mask` in Model Container Forward Pass
* **Location:** [model/model.py:L60-L67](file:///g:/My%20Drive/coding%20llm/model/model.py#L60-L67)
* **Evidence:** `CodingCausalLM.forward()` had signature `(input_ids, position_ids, labels, kv_caches, use_cache)`. It completely omitted `attention_mask`. While `TransformerBlock` and `GroupedQueryAttention` accepted `attention_mask`, `CodingCausalLM` had no mechanism to pass it down.
* **Impact:** Batched inference with padding tokens could not pass attention masks, causing queries to attend to padding tokens.
* **Remediation:** Add `attention_mask: Optional[torch.Tensor] = None` to `forward()` and forward it to all layers.

### Bug 3: Temporary File Leak in Benchmark Runner
* **Location:** [benchmarks/humaneval_runner.py:L36-L39](file:///g:/My%20Drive/coding%20llm/benchmarks/humaneval_runner.py#L36-L39)
* **Evidence:** `tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)` was opened and never deleted in a `finally` block or `os.unlink`.
* **Impact:** Running 164 HumanEval tasks leaked 164 Python files in the temporary directory per evaluation pass.
* **Remediation:** Wrap execution in `try ... finally: if os.path.exists(temp_path): os.unlink(temp_path)`.

### Bug 4: Non-Atomic Checkpoint Persistence & Incomplete Resume State
* **Location:** [training/pretraining/train_engine.py:L117-L124](file:///g:/My%20Drive/coding%20llm/training/pretraining/train_engine.py#L117-L124)
* **Evidence:** Checkpoints were written directly to `save_path` using `torch.save()`. If a crash occurred midway, the file became corrupt. Furthermore, RNG states (`torch.get_rng_state()`, `random.getstate()`) and scheduler states were omitted.
* **Impact:** Resuming from interrupted pretraining runs caused silent non-determinism and risk of loading corrupted checkpoints.
* **Remediation:** Write to a `.tmp` file first and atomic rename with `os.replace()`; record RNG states and scheduler parameters.

### Bug 5: Silent Unauthenticated Access in API Server
* **Location:** [api/security.py:L21-L22](file:///g:/My%20Drive/coding%20llm/api/security.py#L21-L22)
* **Evidence:** If `master_key` is None, `verify_auth_header` returns `"anonymous_local"` without requiring any authentication or header validation.
* **Impact:** Any external deployment without explicitly passing `--api-key` exposed all model inference endpoints to unauthenticated callers.
* **Remediation:** Require explicit `AETHERCODER_ALLOW_ANONYMOUS=1` to allow anonymous local access; otherwise require valid `AETHERCODER_API_KEY`.

---

## 4. Hardware & Memory Reality Check

### Can a 50B Model Be Trained or Inferred on Consumer PCs?
* **Development Environment:** ~16 GB System RAM, Consumer GPU (e.g. 6 GB VRAM) or CPU execution.
* **50B FP32 Weight Memory:** 200 GB (Impossible on consumer hardware).
* **50B BF16 / FP16 Weight Memory:** 100 GB (Requires 2× 80GB H100 GPUs minimum just to load).
* **50B INT4 Quantized Memory:** 25 GB (Requires >24GB VRAM GPU such as RTX 3090/4090 or A100).
* **Pretraining 50B with AdamW & Gradients:** Requires at least 800 GB - 1.2 TB VRAM across a distributed cluster (e.g. 16 to 32 H100 80GB nodes via FSDP).
* **Conclusion & Reality:** The nominal 50B model is an architectural target for distributed cloud clusters. **Local execution must strictly default to the 150M prototype.** The repository must never promise local 50B pretraining.

---

## 5. Prioritized Remediation Roadmap

1. **Phase 2:** Fix `ModelConfig` validation, causal masking, position IDs during KV cache decoding, exact parameter counts, and write numerical equivalence tests between cached vs non-cached decoding.
2. **Phase 3:** Complete tokenizer serialization/roundtrip tests, dataset sharding manifest creation, token overflow guards, and streaming support.
3. **Phase 4:** Add atomic checkpointing, RNG preservation, learning rate scheduler resume, and non-finite loss guards in `train_engine.py`.
4. **Phase 5:** Write `scripts/check_hardware.py` and `scripts/validate_config.py` to inspect hardware and validate distributed configurations without allocating 50B parameters.
5. **Phase 6:** Secure API server with explicit auth, CORS restrictions, model ID validation, and real hardware telemetry.
6. **Phase 7:** Strengthen coding agent with isolated subprocessing, symlink escape detection, and automated test-backed rollback.
7. **Phase 8:** Connect Frontend Developer Studio to real API, display actual model, and sanitize markdown.
8. **Phase 9:** Harden `humaneval_runner.py` with tempfile cleanup, machine-readable JSON reports, and honest documentation.
9. **Phase 10:** Add `requirements.txt`, `pyproject.toml`, and `.github/workflows/ci.yml`.
10. **Phase 11 & 12:** Provide complete documentation and execute full test suite.
