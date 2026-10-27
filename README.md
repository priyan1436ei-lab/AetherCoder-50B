<div align="center">

# ⚡ AetherCoder-50B
### Open-Source Coding Model Research: Sparse MoE Architecture, Sandboxed Agent & Inference Engine

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-36%20Passing%20(100%25)-success.svg?style=for-the-badge)]()

<p align="center">
  <b>AetherCoder</b> is a reproducible, tested coding model research project featuring a functional <b>150M prototype checkpoint</b>, an OpenAI-compatible <b>streaming inference server</b>, an autonomous <b>sandboxed ReAct coding agent</b>, and an audited architecture and distributed scaling plan for <b>Sparse Mixture-of-Experts (MoE)</b> models.
</p>

[**Architecture**](#-model-architecture) •
[**Quick Start**](#-quick-start) •
[**Hardware Feasibility**](#-hardware-feasibility--memory-planning) •
[**Autonomous Agent**](#-autonomous-coding-agent--sandbox) •
[**Inference API**](#-openai-compatible-inference-api) •
[**Documentation**](#-documentation-index)

</div>

---

> [!IMPORTANT]
> **Project Realism & Status Statement:**
> - **Pretrained Prototype**: A working 150M prototype checkpoint (`checkpoints/prototype_150m/checkpoint_step_5_final.pt`) is included and verified for local testing, inference, and fine-tuning on consumer hardware.
> - **50B Model Specification**: Full model architecture definitions, divisibility checks, and distributed FSDP configurations (`configs/model/moe_50b.yaml`) are implemented. **No pretrained 50B checkpoint currently exists in this repository.** Training a 50B MoE model from scratch requires a multi-node cluster (32x–64x NVIDIA H100 80GB GPUs) and cannot be executed on consumer hardware.
> - **Benchmark Integrity**: Previous unverified benchmark score tables have been excised. Verified evaluation harnesses and report generators are available in `benchmarks/` and documented in [`docs/EVALUATION.md`](docs/EVALUATION.md).

---

## 🌟 Key Features

1. **Decoder-Only Sparse MoE Architecture**:
   - Implements Root Mean Square Normalization (RMSNorm), Rotary Position Embeddings (RoPE), Grouped-Query Attention (GQA), and SwiGLU feed-forward networks.
   - Top-k sparse expert routing with dedicated shared experts and auxiliary load-balancing loss.
   - Numerically stable attention with causal masking and exact KV cache equivalence.

2. **Multilingual Dataset & Binary Sharding**:
   - Curated algorithm corpus across 16 programming languages covering multi-file reasoning, unit test generation, SSRF DNS pinning, concurrency primitives, and memory allocators.
   - Content deduplication (SHA-256), 80/10/10 splits, and binary sharding with explicit `uint32` token types preventing silent overflow.
   - Automatic dataset manifest generation (`data/dataset_manifest.json`).

3. **Resilient Training Engine**:
   - Atomic checkpoint writing (`.tmp` + `os.replace`), complete RNG state serialization (PyTorch CPU/CUDA, Python `random`, NumPy), and optimizer/scheduler state persistence.
   - Automatic non-finite loss/gradient handling, mixed-precision (BF16/FP16) with `GradScaler`, and target-masked SFT / DPO.

4. **Autonomous ReAct Coding Agent**:
   - Workspace AST indexing, unified diff generation, and surgical patching.
   - **PermissionGuard** enforces strict boundaries: blocks path traversal, rejects symlink escapes outside the workspace root, sanitizes environment variables to prevent secret leakage, and blocks unauthorized package installs or shell injections.
   - Auto-rollback restores original code immediately if proposed patches fail validation tests.

5. **OpenAI-Compatible FastAPI Server**:
   - Endpoints: `/health`, `/v1/models`, `/v1/chat/completions` with Server-Sent Events (SSE) streaming.
   - Rejects unsupported model requests with HTTP 400.
   - Bearer authentication via `AETHERCODER_API_KEY` (timing-safe `secrets.compare_digest`), configurable CORS, and token-bucket rate limiting.

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Hardware & Validate Configurations
```bash
# Check local compute, RAM, VRAM, and inspect parameter sizing
python scripts/check_hardware.py

# Validate architectural constraints without allocating large tensors
python scripts/validate_config.py
```

### 3. Run the Automated Test Suite
```bash
python -m pytest tests/ -v
```

### 4. Build Dataset Shards
```bash
python scripts/prepare_dataset.py --samples 200 --output_dir data/shards
```

### 5. Train Prototype Checkpoint
```bash
python scripts/run_prototype_train.py --steps 5 --batch_size 2 --lr 3e-4 --device cpu
```

### 6. Inspect Saved Checkpoints
```bash
python scripts/inspect_checkpoint.py checkpoints/prototype_150m/checkpoint_step_5_final.pt
```

### 7. Launch Inference API & Developer Studio
```bash
# Enable local anonymous access for development
uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload
```
Navigate to:
- **Developer Studio UI**: [http://127.0.0.1:8000/ui/index.html](http://127.0.0.1:8000/ui/index.html)
- **API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Telemetry**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 🏗️ Model Architecture

| Parameter | Prototype (150M) | Full MoE (50B Nom.) | Dense (31B) |
| :--- | :--- | :--- | :--- |
| **Model Type** | `moe_causal_lm` | `moe_causal_lm` | `dense_causal_lm` |
| **Hidden Size ($d_{\text{model}}$)** | 512 | 6,144 | 6,144 |
| **Layers** | 8 | 48 | 56 |
| **Heads / KV-Heads** | 8 / 2 (4:1 GQA) | 48 / 8 (6:1 GQA) | 48 / 8 (6:1 GQA) |
| **Intermediate Size** | 1,536 | 16,384 | 16,384 |
| **Context Window** | 4,096 | 32,768 | 32,768 |
| **Total Experts / Active** | 4 / Top-2 | 16 / Top-2 | 1 (Dense) |
| **Shared Experts** | 1 | 2 | 0 |
| **Total Parameters** | **156.92M** | **265.96B** | **31.11B** |
| **Active Params / Token** | **123.10M** | **63.02B** | **31.11B** |

---

## 💻 Hardware Feasibility & Memory Planning

Theoretical memory requirements for 50B-scale models:

| Component | FP32 | BF16 / FP16 | INT8 | 4-bit (QLoRA) |
| :--- | :--- | :--- | :--- | :--- |
| **50B MoE Weights Alone** | 990.77 GB | 495.38 GB | 247.69 GB | 123.85 GB |
| **50B Dense Weights Alone** | 115.88 GB | 57.94 GB | 28.97 GB | 14.48 GB |
| **150M Prototype Weights** | 0.58 GB | 0.29 GB | 0.15 GB | 0.07 GB |

**Training Compute Reality:**
- Pretraining the nominal 50B MoE model to Chinchilla optimality requires $\ge 1.0\text{ Trillion}$ tokens ($\approx 4.5 \times 10^{23}\text{ FLOPs}$).
- This requires a dedicated cluster of **32x–64x NVIDIA H100 80GB** GPUs interconnected via 400 Gbps InfiniBand running PyTorch FSDP.
- The repository defaults to the **150M prototype** to ensure all code, training loops, agent workflows, and API routes run locally on consumer hardware without out-of-memory crashes.

---

## 📚 Documentation Index

- [`docs/REPOSITORY_AUDIT.md`](docs/REPOSITORY_AUDIT.md) — Comprehensive repository audit, verified bugs, and remediation roadmap.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Mathematical specification of RoPE, GQA, SwiGLU, and MoE routing.
- [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) — Pretraining, SFT, DPO, and distributed FSDP cloud execution.
- [`docs/DATASET_GUIDE.md`](docs/DATASET_GUIDE.md) — Tokenization, code preservation, binary sharding format, and manifest schema.
- [`docs/API.md`](docs/API.md) — OpenAI-compatible FastAPI endpoints, SSE streaming, authentication, and CORS.
- [`docs/EVALUATION.md`](docs/EVALUATION.md) — HumanEval/MBPP-compatible sandboxed execution and JSON report generation.
- [`docs/SECURITY.md`](docs/SECURITY.md) — Sandbox security model, symlink escape defenses, and secrets sanitization.

---

## 📜 License

Licensed under the [Apache License 2.0](LICENSE).
