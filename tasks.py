from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    l1_cache_kb: int
    max_threads_per_block: int
    shared_memory_kb: int
    memory_bandwidth_gbps: float
    peak_compute_tflops: float
    warp_size: int = 32


DEFAULT_HARDWARE_PROFILE = HardwareProfile(
    name="SiliconArchitect-v1-virtual-accelerator",
    l1_cache_kb=64,
    max_threads_per_block=256,
    shared_memory_kb=32,
    memory_bandwidth_gbps=900.0,
    peak_compute_tflops=18.0,
)


TASKS: Dict[str, Dict[str, Any]] = {
    "vector_add_bandwidth": {
        "id": "vector_add_bandwidth",
        "name": "Vector Addition (Bandwidth Bound)",
        "difficulty": "easy",
        "equation": "y[i] = a[i] + b[i]",
        "baseline_python": (
            "def vector_add(a, b):\n"
            "    return [x + y for x, y in zip(a, b)]"
        ),
        "description": (
            "Optimize a 1D vector addition kernel for maximum memory throughput "
            "under block-size and loop-unrolling constraints."
        ),
        "problem_size": {"num_elements": 1_048_576},
        "optimal_params": {
            "block_size_x": 256,
            "block_size_y": 1,
            "unroll_factor": 4,
            "use_shared_memory": False,
        },
    },
    "tiled_matmul_cache": {
        "id": "tiled_matmul_cache",
        "name": "Tiled Matrix Multiplication (Compute Bound)",
        "difficulty": "medium",
        "equation": "C[M, N] = A[M, K] @ B[K, N]",
        "baseline_python": (
            "def matmul(a, b):\n"
            "    m, k = len(a), len(a[0])\n"
            "    n = len(b[0])\n"
            "    out = [[0.0 for _ in range(n)] for _ in range(m)]\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            acc = 0.0\n"
            "            for kk in range(k):\n"
            "                acc += a[i][kk] * b[kk][j]\n"
            "            out[i][j] = acc\n"
            "    return out"
        ),
        "description": (
            "Choose tile dimensions that fit cleanly into the 64KB L1 cache and "
            "minimize simulated DRAM traffic for matrix multiplication."
        ),
        "problem_size": {"m": 1024, "n": 1024, "k": 1024},
        "optimal_params": {
            "block_size_x": 16,
            "block_size_y": 16,
            "unroll_factor": 8,
            "use_shared_memory": True,
        },
    },
    "flash_attention_shared": {
        "id": "flash_attention_shared",
        "name": "Flash Attention Mock (Shared Memory Conflicts)",
        "difficulty": "hard",
        "equation": "softmax(Q @ K^T / sqrt(d)) @ V",
        "baseline_python": (
            "def attention(q, k, v):\n"
            "    scores = q @ k.T\n"
            "    probs = softmax(scores)\n"
            "    return probs @ v"
        ),
        "description": (
            "Balance tile dimensions, loop unrolling, and shared-memory use to "
            "minimize latency while avoiding bank conflicts and OOM."
        ),
        "problem_size": {"seq_len": 1024, "head_dim": 64, "num_heads": 8},
        "optimal_params": {
            "block_size_x": 16,
            "block_size_y": 16,
            "unroll_factor": 4,
            "use_shared_memory": True,
        },
    },
}


def _clamp_score(score: float) -> float:
    """Clamp score to the open interval (0, 1) — never exactly 0.0 or 1.0."""
    return round(min(0.999, max(0.001, score)), 4)


def _base_metrics(task: Dict[str, Any], params: Dict[str, Any], hw: HardwareProfile) -> Dict[str, Any]:
    block_x = params["block_size_x"]
    block_y = params["block_size_y"]
    unroll = params["unroll_factor"]
    use_shared = params["use_shared_memory"]
    threads = block_x * block_y
    shared_bytes = block_x * block_y * max(unroll, 1) * 4
    invalid = (
        block_x <= 0
        or block_y <= 0
        or unroll <= 0
        or threads > hw.max_threads_per_block
        or shared_bytes > hw.shared_memory_kb * 1024
    )
    return {
        "task_id": task["id"],
        "threads_per_block": threads,
        "shared_memory_bytes": shared_bytes if use_shared else 0,
        "requested_shared_memory_bytes": shared_bytes,
        "invalid": invalid,
    }


def grade_vector_add(task: Dict[str, Any], params: Dict[str, Any], hw: HardwareProfile) -> Dict[str, Any]:
    metrics = _base_metrics(task, params, hw)
    if metrics["invalid"]:
        metrics.update(
            {
                "bandwidth_utilization": 0.0,
                "latency_ns": 1_000_000.0,
                "score": _clamp_score(0.0),
                "oom": metrics["requested_shared_memory_bytes"] > hw.shared_memory_kb * 1024,
            }
        )
        return metrics

    block_x = params["block_size_x"]
    unroll = params["unroll_factor"]

    occupancy = min(metrics["threads_per_block"] / hw.max_threads_per_block, 1.0)
    coalescing_bonus = 1.0 if block_x % hw.warp_size == 0 else 0.72
    unroll_efficiency = max(0.55, 1.0 - abs(unroll - 4) * 0.12)
    bandwidth_utilization = max(
        0.0,
        min(0.99, occupancy * coalescing_bonus * unroll_efficiency),
    )
    latency_ns = 100_000.0 / max(bandwidth_utilization, 0.05)
    partial_score = 0.1 + min(0.8, bandwidth_utilization * 0.8)
    score = _clamp_score(1.0 if bandwidth_utilization > 0.95 else partial_score)

    metrics.update(
        {
            "bandwidth_utilization": round(bandwidth_utilization, 4),
            "latency_ns": round(latency_ns, 2),
            "score": score,
            "oom": False,
        }
    )
    return metrics


def grade_matmul(task: Dict[str, Any], params: Dict[str, Any], hw: HardwareProfile) -> Dict[str, Any]:
    metrics = _base_metrics(task, params, hw)
    if metrics["invalid"]:
        metrics.update(
            {
                "cache_miss_rate": 1.0,
                "dram_reads": float("inf"),
                "latency_ns": 2_000_000.0,
                "score": _clamp_score(0.0),
                "oom": metrics["requested_shared_memory_bytes"] > hw.shared_memory_kb * 1024,
            }
        )
        return metrics

    block_x = params["block_size_x"]
    block_y = params["block_size_y"]
    unroll = params["unroll_factor"]
    use_shared = params["use_shared_memory"]

    tile_bytes = (block_x * block_y + block_x * unroll + block_y * unroll) * 4
    cache_capacity = hw.l1_cache_kb * 1024
    fit_ratio = tile_bytes / cache_capacity
    fit_penalty = max(0.0, fit_ratio - 1.0)

    ideal_tile_x = 16
    ideal_tile_y = 16
    tile_distance = (abs(block_x - ideal_tile_x) / 32.0) + (abs(block_y - ideal_tile_y) / 32.0)
    unroll_penalty = abs(unroll - 8) / 16.0
    shared_bonus = 0.08 if use_shared else 0.0

    cache_miss_rate = min(
        1.0,
        max(0.0, 0.08 + fit_penalty * 0.65 + tile_distance * 0.22 + unroll_penalty - shared_bonus),
    )
    dram_reads = round((1.0 + cache_miss_rate * 4.5) * 1_000_000, 2)
    latency_ns = round(220_000.0 * (1.0 + cache_miss_rate * 3.0), 2)
    score = _clamp_score(1.0 - cache_miss_rate)

    metrics.update(
        {
            "tile_bytes": tile_bytes,
            "cache_fit_ratio": round(min(fit_ratio, 10.0), 4),
            "cache_miss_rate": round(cache_miss_rate, 4),
            "dram_reads": dram_reads,
            "latency_ns": latency_ns,
            "score": score,
            "oom": False,
        }
    )
    return metrics


def grade_flash_attention(task: Dict[str, Any], params: Dict[str, Any], hw: HardwareProfile) -> Dict[str, Any]:
    metrics = _base_metrics(task, params, hw)
    if metrics["requested_shared_memory_bytes"] > hw.shared_memory_kb * 1024:
        metrics.update(
            {
                "bank_conflicts": 999,
                "latency_ns": 5_000_000.0,
                "score": _clamp_score(0.0),
                "oom": True,
            }
        )
        return metrics

    if metrics["invalid"]:
        metrics.update(
            {
                "bank_conflicts": 999,
                "latency_ns": 5_000_000.0,
                "score": _clamp_score(0.0),
                "oom": False,
            }
        )
        return metrics

    block_x = params["block_size_x"]
    block_y = params["block_size_y"]
    unroll = params["unroll_factor"]
    use_shared = params["use_shared_memory"]

    shared_factor = 0.62 if use_shared else 1.0
    bank_conflicts = 0
    if use_shared:
        bank_conflicts = (block_x // 8 + block_y // 8 + unroll) % 8
    else:
        bank_conflicts = 6

    tile_distance = (abs(block_x - 16) / 32.0) + (abs(block_y - 16) / 32.0)
    unroll_penalty = abs(unroll - 4) / 6.0
    conflict_penalty = bank_conflicts * 0.14
    latency_ns = round(
        180_000.0 * shared_factor * (1.0 + tile_distance + unroll_penalty + conflict_penalty),
        2,
    )
    raw_score = 1.0 - min(1.0, tile_distance * 0.35 + unroll_penalty * 0.2 + conflict_penalty)
    if bank_conflicts == 0 and use_shared and block_x == 16 and block_y == 16 and unroll == 4:
        raw_score = 1.0
    score = _clamp_score(raw_score)

    metrics.update(
        {
            "bank_conflicts": bank_conflicts,
            "latency_ns": latency_ns,
            "score": score,
            "oom": False,
        }
    )
    return metrics


def grade_task(task_id: str, params: Dict[str, Any], hw: HardwareProfile = DEFAULT_HARDWARE_PROFILE) -> Dict[str, Any]:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id: {task_id}")

    task = TASKS[task_id]
    if task_id == "vector_add_bandwidth":
        return grade_vector_add(task, params, hw)
    if task_id == "tiled_matmul_cache":
        return grade_matmul(task, params, hw)
    if task_id == "flash_attention_shared":
        return grade_flash_attention(task, params, hw)
    raise KeyError(f"Unsupported task_id: {task_id}")
