# AetherCoder-50B Model Architecture Specification

## Overview

AetherCoder is an autoregressive decoder-only Transformer tailored for multilingual code intelligence, repository reasoning, Fill-in-the-Middle (FIM) generation, and autonomous coding agents.

The architecture incorporates modern transformer engineering advancements:
- **Rotary Position Embeddings (RoPE)** with dynamic frequency extrapolation for long-context stability.
- **Grouped-Query Attention (GQA)** to minimize KV cache footprint during incremental decoding.
- **Root Mean Square Normalization (RMSNorm)** for numerical stability across deep layers.
- **SwiGLU Feed-Forward Networks** for expressive non-linear activation.
- **Sparse Mixture-of-Experts (MoE)** routing with dedicated **Shared Experts** and auxiliary load balancing.

---

## Architectural Configurations

| Parameter | Prototype (150M) | Full MoE (50B Nom.) | Dense (31B) |
| :--- | :--- | :--- | :--- |
| **Model Type** | `moe_causal_lm` | `moe_causal_lm` | `dense_causal_lm` |
| **Hidden Dimension ($d_{\text{model}}$)** | 512 | 6,144 | 6,144 |
| **Layers** | 8 | 48 | 56 |
| **Attention Heads ($H_Q$)** | 8 | 48 | 48 |
| **Key/Value Heads ($H_{KV}$)** | 2 (4:1 GQA) | 8 (6:1 GQA) | 8 (6:1 GQA) |
| **Head Dimension ($d_{\text{head}}$)** | 64 | 128 | 128 |
| **Intermediate Size ($d_{\text{ffn}}$)** | 1,536 | 16,384 | 16,384 |
| **Vocabulary Size** | 65,536 | 65,536 | 65,536 |
| **Context Window** | 4,096 | 32,768 | 32,768 |
| **Total Experts ($E$)** | 4 | 16 | 1 (Dense) |
| **Routed Experts / Token ($k$)** | 2 | 2 | N/A |
| **Shared Experts ($E_{\text{shared}}$)**| 1 | 2 | N/A |
| **Total Parameters** | **156.92M** | **265.96B** | **31.11B** |
| **Active Params / Token** | **123.10M** | **63.02B** | **31.11B** |

---

## Key Components

### 1. Rotary Position Embeddings (RoPE)
RoPE applies complex rotation matrices to query and key states:
$$\mathbf{q}_m = \mathbf{R}_{\Theta, m}^d \mathbf{W}_q \mathbf{x}_m, \quad \mathbf{k}_n = \mathbf{R}_{\Theta, n}^d \mathbf{W}_k \mathbf{x}_n$$
Our implementation dynamically caches cosine and sine frequencies up to the maximum context length, gracefully expanding when sequences exceed initial buffer allocations without allocating unexpected CPU/GPU tensors.

### 2. Grouped-Query Attention (GQA) & KV Cache
To reduce memory bandwidth saturation during cached token-by-token generation, query heads are partitioned across key/value heads ($H_Q : H_{KV}$ ratio).
During incremental decoding:
- Past key and value states are retained in `DynamicKVCache`.
- Position offsets account for past tokens (`past_len = kv_caches[0][0].shape[2]`).
- Full sequence decoding and incremental KV cache decoding are numerically identical within float tolerance.

### 3. Sparse MoE Routing & Shared Experts
Each MoE layer contains $E$ routed experts and $E_{\text{shared}}$ always-active shared experts:
$$\mathbf{y} = \sum_{e \in \text{Top-}k} g_e(\mathbf{x}) \cdot \text{Expert}_e(\mathbf{x}) + \sum_{s=1}^{E_{\text{shared}}} \text{SharedExpert}_s(\mathbf{x})$$
- Routing weights $g_e(\mathbf{x})$ use softmax over Top-$k$ logits with renormalization.
- **Load-Balancing Loss**: Auxiliary loss penalizes uneven expert dispatch to prevent routing collapse:
  $$\mathcal{L}_{\text{balance}} = \alpha \cdot E \sum_{i=1}^E f_i \cdot P_i$$
- **Router Z-Loss**: Penalizes large router logits to maintain float stability.
- **Zero-Token Safety**: When an expert receives no tokens in a micro-batch, dispatch operations bypass execution cleanly without tensor dimension errors.

### 4. Dynamic Parameter Counting
Parameters are computed directly from PyTorch module graph inspections rather than hardcoded marketing numbers:
- Total parameters
- Active parameters per token
- Inactive parameters per token
- Routed expert parameters
- Shared expert parameters
- Embedding & head parameters
