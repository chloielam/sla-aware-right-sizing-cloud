from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Observation:
    step: int
    timestamp: int
    demand_qps: float
    queue_len: float
    latency_p99_ms: float
    utilization: float
    active_instances: int
    min_instances: int
    max_instances: int
    sla_threshold_ms: float
    predicted_demand_qps: float
    predicted_latency_ms: float
    recent_violation_rate: float
    external_queue_signal: float
    external_latency_signal_ms: float
    gpu_util_pct: float
    memory_util: float


class BasePolicy:
    def __init__(self, name: str, params: dict | None = None) -> None:
        self.name = name
        self.params = params or {}

    def decide_target_instances(self, obs: Observation) -> int:
        raise NotImplementedError

    @staticmethod
    def clamp(value: int, low: int, high: int) -> int:
        return max(low, min(high, value))
