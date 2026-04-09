from __future__ import annotations

from collections import deque

from src.policies.base_policy import BasePolicy, Observation


class SLAAwarePolicy(BasePolicy):
    algorithm_name = "heuristic_risk"

    def __init__(self, name: str, params: dict | None = None) -> None:
        super().__init__(name, params)
        self._risk_hist: deque[float] = deque(maxlen=int(self.params.get("history_window", 8)))

    def decide_target_instances(self, obs: Observation) -> int:
        queue_budget = float(self.params.get("queue_budget", 12.0))
        latency_floor_ratio = float(self.params.get("latency_floor_ratio", 0.70))
        risk_up_threshold = float(self.params.get("risk_up_threshold", 0.32))
        risk_down_threshold = float(self.params.get("risk_down_threshold", 0.12))
        latency_weight = float(self.params.get("latency_weight", 0.50))
        queue_weight = float(self.params.get("queue_weight", 0.30))
        demand_weight = float(self.params.get("demand_weight", 0.20))
        hard_latency_ratio = float(self.params.get("hard_latency_ratio", 0.92))
        hard_queue_threshold = float(self.params.get("hard_queue_threshold", queue_budget * 0.65))
        hard_demand_rush_ratio = float(self.params.get("hard_demand_rush_ratio", 0.30))
        downscale_latency_ratio = float(self.params.get("downscale_latency_ratio", 0.78))
        downscale_queue_threshold = float(self.params.get("downscale_queue_threshold", queue_budget * 0.25))
        downscale_util_threshold = float(self.params.get("downscale_util_threshold", 0.55))
        up_step = int(self.params.get("scale_up_step", 1))
        down_step = int(self.params.get("scale_down_step", 1))

        latency_ratio = obs.latency_p99_ms / max(1.0, obs.sla_threshold_ms)
        latency_pressure = max(0.0, latency_ratio - latency_floor_ratio)
        queue_pressure = min(2.0, obs.queue_len / max(1.0, queue_budget))
        demand_rush = max(0.0, obs.predicted_demand_qps - obs.demand_qps) / max(1.0, obs.demand_qps)
        bounded_demand_rush = min(1.5, demand_rush)

        risk = (
            latency_weight * latency_pressure
            + queue_weight * queue_pressure
            + demand_weight * bounded_demand_rush
        )
        self._risk_hist.append(risk)
        smooth_risk = sum(self._risk_hist) / len(self._risk_hist)

        hard_signal = (
            obs.latency_p99_ms >= obs.sla_threshold_ms * hard_latency_ratio
            or obs.queue_len >= hard_queue_threshold
            or demand_rush >= hard_demand_rush_ratio
        )

        target = obs.active_instances
        if smooth_risk >= risk_up_threshold and hard_signal:
            target += up_step
        elif (
            smooth_risk <= risk_down_threshold
            and obs.latency_p99_ms < obs.sla_threshold_ms * downscale_latency_ratio
            and obs.queue_len <= downscale_queue_threshold
            and obs.utilization <= downscale_util_threshold
        ):
            target -= down_step

        return self.clamp(target, obs.min_instances, obs.max_instances)
