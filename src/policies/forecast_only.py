from __future__ import annotations

from collections import deque

from src.policies.base_policy import BasePolicy, Observation


class ForecastOnlyPolicy(BasePolicy):
    algorithm_name = "forecast_only"

    def __init__(self, name: str, params: dict | None = None) -> None:
        super().__init__(name, params)
        self._forecast_hist: deque[float] = deque(maxlen=int(self.params.get("history_window", 8)))

    def decide_target_instances(self, obs: Observation) -> int:
        demand_gap_ratio = max(0.0, obs.predicted_demand_qps - obs.demand_qps) / max(1.0, obs.demand_qps)
        predicted_utilization = (
            obs.utilization * (obs.predicted_demand_qps / max(1e-6, obs.demand_qps))
            if obs.demand_qps > 0
            else obs.utilization
        )
        predicted_latency_ratio = obs.predicted_latency_ms / max(1.0, obs.sla_threshold_ms)
        forecast_score = 0.45 * predicted_latency_ratio + 0.35 * demand_gap_ratio + 0.20 * predicted_utilization
        self._forecast_hist.append(forecast_score)
        smooth_forecast = sum(self._forecast_hist) / len(self._forecast_hist)

        up_step = int(self.params.get("scale_up_step", 2))
        down_step = int(self.params.get("scale_down_step", 1))
        forecast_up_threshold = float(self.params.get("forecast_up_threshold", 0.95))
        forecast_down_threshold = float(self.params.get("forecast_down_threshold", 0.60))
        predicted_utilization_up = float(self.params.get("predicted_utilization_up", 0.82))
        predicted_utilization_down = float(self.params.get("predicted_utilization_down", 0.50))

        target = obs.active_instances
        if (
            smooth_forecast >= forecast_up_threshold
            or predicted_latency_ratio >= 0.95
            or predicted_utilization >= predicted_utilization_up
            or demand_gap_ratio >= 0.30
        ):
            target += up_step
        elif (
            smooth_forecast <= forecast_down_threshold
            and predicted_utilization <= predicted_utilization_down
            and obs.latency_p99_ms < obs.sla_threshold_ms * 0.75
        ):
            target -= down_step

        return self.clamp(target, obs.min_instances, obs.max_instances)
