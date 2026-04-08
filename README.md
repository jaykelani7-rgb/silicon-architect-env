---
title: SiliconArchitect v1
emoji: 💻
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# SiliconArchitect-v1

SiliconArchitect-v1 is a CPU-only OpenEnv environment for the Meta x Hugging Face OpenEnv Hackathon. It trains and evaluates agents on hardware-software co-design by asking them to optimize high-level mathematical workloads into low-level virtual kernel launch parameters.

Instead of executing real CUDA or Triton code, the environment uses a deterministic analytical hardware simulator. This makes the benchmark portable to the hackathon constraints of 2 vCPUs and 8GB RAM while still rewarding the kinds of decisions real systems agents need to make: occupancy, tiling, cache fit, shared-memory use, and conflict avoidance.

## Benchmark Motivation

This environment is designed as an LLM-to-kernel benchmark:

- Agents observe the target equation, a naive Python implementation, and a virtual hardware profile.
- Agents act by submitting kernel parameters: `block_size_x`, `block_size_y`, `unroll_factor`, and `use_shared_memory`.
- The simulator computes deterministic latency and efficiency metrics using analytical cost models.
- Programmatic graders return scores between `0.0` and `1.0` for all three tasks.

## Tasks

1. `vector_add_bandwidth`
   Focuses on bandwidth saturation for a 1D vector addition kernel.
2. `tiled_matmul_cache`
   Rewards tile choices that fit the 64KB L1 cache and reduce DRAM traffic.
3. `flash_attention_shared`
   Simulates flash-attention-style optimization with shared-memory bank conflict avoidance.

## OpenEnv Compliance

The implementation includes:

- Pydantic typed `Observation`, `Action`, and `Reward` models.
- `reset()`, `step(action)`, and `state()` methods in the environment.
- Shaped rewards for syntax correctness, memory safety, and latency improvement.
- Deterministic task graders in [`tasks.py`](./tasks.py).
- Metadata in [`openenv.yaml`](./openenv.yaml).
- A lightweight CPU-only Docker image in [`Dockerfile`](./Dockerfile).

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install openai==1.77.0 pydantic==1.10.15 PyYAML==6.0.2
```

## Running Inference

The baseline runner is [`inference.py`](./inference.py). It reads credentials strictly from environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Then it runs a baseline model across all three tasks and logs environment progress with strict `[START]`, `[STEP]`, and `[END]` tags.

Example:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
export HF_TOKEN="hf_..."
python inference.py
```

## Docker

```bash
docker build -t siliconarchitect-v1 .
docker run --rm \
  -e API_BASE_URL="$API_BASE_URL" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e HF_TOKEN="$HF_TOKEN" \
  siliconarchitect-v1
```

## Notes

- The simulator is deterministic and GPU-free by design.
- Invalid thread configurations return `0.0`.
- Flash-attention shared-memory over-allocation is treated as OOM and also returns `0.0`.
- The environment is designed to complete well under five minutes on CPU-only infrastructure.
