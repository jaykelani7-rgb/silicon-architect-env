from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from tasks import DEFAULT_HARDWARE_PROFILE, TASKS, grade_task


class Action(BaseModel):
    block_size_x: int = Field(..., ge=1, le=256)
    block_size_y: int = Field(..., ge=1, le=256)
    unroll_factor: int = Field(..., ge=1, le=16)
    use_shared_memory: bool


class Reward(BaseModel):
    total: float = Field(..., ge=0.0, le=1.0)
    task_score: float = Field(..., ge=0.0, le=1.0)
    syntax_correctness: float = Field(..., ge=0.0, le=1.0)
    memory_safety: float = Field(..., ge=0.0, le=1.0)
    latency_improvement: float = Field(..., ge=0.0, le=1.0)
    explanation: str


class Observation(BaseModel):
    task_id: str
    task_name: str
    difficulty: str
    equation: str
    baseline_python: str
    hardware_profile: Dict[str, Any]
    action_schema: Dict[str, str]
    last_metrics: Optional[Dict[str, Any]] = None
    step_index: int = 0
    remaining_steps: int = 0


class EnvironmentState(BaseModel):
    current_task_id: str
    step_index: int
    max_steps: int
    best_score: float
    best_latency_ns: Optional[float]
    done: bool
    history: List[Dict[str, Any]]

    @field_validator("history", mode="before")
    @classmethod
    def _default_history(cls, value: Any) -> List[Dict[str, Any]]:
        return value or []


class AnalyticalHardwareSimulator:
    def __init__(self) -> None:
        self.hardware_profile = DEFAULT_HARDWARE_PROFILE

    def evaluate(self, task_id: str, action: Action) -> Dict[str, Any]:
        action_payload = action.model_dump() if hasattr(action, "model_dump") else action.dict()
        return grade_task(task_id, action_payload, self.hardware_profile)


class SiliconArchitectOpenEnv:
    def __init__(self, task_id: str = "vector_add_bandwidth", max_steps: int = 6) -> None:
        if task_id not in TASKS:
            raise KeyError(f"Unknown task_id: {task_id}")
        self.simulator = AnalyticalHardwareSimulator()
        self.max_steps = max_steps
        self._task_id = task_id
        self._step_index = 0
        self._best_score = 0.0
        self._best_latency_ns: Optional[float] = None
        self._done = False
        self._history: List[Dict[str, Any]] = []
        self._last_metrics: Optional[Dict[str, Any]] = None

    @property
    def task(self) -> Dict[str, Any]:
        return TASKS[self._task_id]

    def reset(self, task_id: Optional[str] = None) -> Observation:
        if task_id is not None:
            if task_id not in TASKS:
                raise KeyError(f"Unknown task_id: {task_id}")
            self._task_id = task_id
        self._step_index = 0
        self._best_score = 0.0
        self._best_latency_ns = None
        self._done = False
        self._history = []
        self._last_metrics = None
        return self._observation()

    def state(self) -> EnvironmentState:
        return EnvironmentState(
            current_task_id=self._task_id,
            step_index=self._step_index,
            max_steps=self.max_steps,
            best_score=self._best_score,
            best_latency_ns=self._best_latency_ns,
            done=self._done,
            history=self._history,
        )

    def step(self, action: Action) -> Dict[str, Any]:
        if self._done:
            raise RuntimeError("Environment is done. Call reset() before stepping again.")

        metrics = self.simulator.evaluate(self._task_id, action)
        reward = self._shape_reward(metrics)

        self._step_index += 1
        self._last_metrics = metrics
        self._best_score = max(self._best_score, reward.task_score)
        if self._best_latency_ns is None or metrics["latency_ns"] < self._best_latency_ns:
            self._best_latency_ns = metrics["latency_ns"]

        transition = {
            "action": action.model_dump() if hasattr(action, "model_dump") else action.dict(),
            "metrics": metrics,
            "reward": reward.model_dump() if hasattr(reward, "model_dump") else reward.dict(),
        }
        self._history.append(transition)
        self._done = self._step_index >= self.max_steps or reward.task_score >= 0.9999

        return {
            "observation": self._observation(),
            "reward": reward,
            "done": self._done,
            "info": {
                "metrics": metrics,
                "best_score": self._best_score,
                "best_latency_ns": self._best_latency_ns,
            },
        }

    def _observation(self) -> Observation:
        return Observation(
            task_id=self.task["id"],
            task_name=self.task["name"],
            difficulty=self.task["difficulty"],
            equation=self.task["equation"],
            baseline_python=self.task["baseline_python"],
            hardware_profile={
                "name": self.simulator.hardware_profile.name,
                "l1_cache_kb": self.simulator.hardware_profile.l1_cache_kb,
                "max_threads_per_block": self.simulator.hardware_profile.max_threads_per_block,
                "shared_memory_kb": self.simulator.hardware_profile.shared_memory_kb,
                "memory_bandwidth_gbps": self.simulator.hardware_profile.memory_bandwidth_gbps,
                "peak_compute_tflops": self.simulator.hardware_profile.peak_compute_tflops,
                "warp_size": self.simulator.hardware_profile.warp_size,
            },
            action_schema={
                "block_size_x": "int",
                "block_size_y": "int",
                "unroll_factor": "int",
                "use_shared_memory": "bool",
            },
            last_metrics=self._last_metrics,
            step_index=self._step_index,
            remaining_steps=max(self.max_steps - self._step_index, 0),
        )

    def _shape_reward(self, metrics: Dict[str, Any]) -> Reward:
        syntax_correctness = 1.0 if not metrics.get("invalid", False) else 0.0
        memory_safety = 0.0 if metrics.get("oom", False) else 1.0

        baseline_latency_map = {
            "vector_add_bandwidth": 250_000.0,
            "tiled_matmul_cache": 900_000.0,
            "flash_attention_shared": 750_000.0,
        }
        baseline_latency = baseline_latency_map[self._task_id]
        latency = float(metrics["latency_ns"])
        latency_improvement = max(0.0, min(1.0, (baseline_latency - latency) / baseline_latency))
        task_score = float(metrics["score"])

        total = (
            task_score * 0.65
            + syntax_correctness * 0.1
            + memory_safety * 0.1
            + latency_improvement * 0.15
        )
        total = max(0.0, min(1.0, round(total, 4)))

        explanation_bits = [
            f"task_score={task_score:.4f}",
            f"syntax_correctness={syntax_correctness:.2f}",
            f"memory_safety={memory_safety:.2f}",
            f"latency_improvement={latency_improvement:.4f}",
        ]
        if metrics.get("oom"):
            explanation_bits.append("OOM penalty applied")
        if metrics.get("invalid"):
            explanation_bits.append("invalid configuration")

        return Reward(
            total=total,
            task_score=round(task_score, 4),
            syntax_correctness=syntax_correctness,
            memory_safety=memory_safety,
            latency_improvement=round(latency_improvement, 4),
            explanation=", ".join(explanation_bits),
        )


OpenEnv = SiliconArchitectOpenEnv
