from __future__ import annotations

from collections import deque

from src.policies.base_policy import BasePolicy, Observation


class SLAAwarePolicy(BasePolicy):
    def __init__(self, name: str, params: dict | None = None) -> None:
        super().__init__(name, params)
        self._risk_hist: deque[float] = deque(maxlen=int(self.params.get("history_window", 12)))
        self._demand_hist: deque[float] = deque(maxlen=int(self.params.get("trend_window", 6)))

    def decide_target_instances(self, obs: Observation) -> int:
        self._demand_hist.append(obs.demand_qps)
        queue_budget = float(self.params.get("queue_budget", 15.0))
        risk_up = float(self.params.get("risk_up_threshold", 1.0))
        risk_down = float(self.params.get("risk_down_threshold", 0.45))
        up_step = int(self.params.get("scale_up_step", 1))
        down_step = int(self.params.get("scale_down_step", 1))
        prediction_weight = float(self.params.get("prediction_weight", 0.55))

        if len(self._demand_hist) >= 2:
            trend = (self._demand_hist[-1] - self._demand_hist[0]) / max(1e-6, self._demand_hist[0])
        else:
            trend = 0.0

        latency_risk = obs.predicted_latency_ms / max(1.0, obs.sla_threshold_ms)
        queue_risk = obs.queue_len / max(1.0, queue_budget)
        trend_risk = max(0.0, trend)
        demand_agg_risk = (obs.predicted_demand_qps - obs.demand_qps) / max(1e-6, obs.demand_qps)
        external_risk = obs.external_queue_signal / max(1.0, queue_budget * 1.5)
        latency_pressure = obs.latency_p99_ms / max(1.0, obs.sla_threshold_ms)
        blended_risk = (
            prediction_weight * max(0.0, min(2.0, latency_pressure))
            + 0.30 * queue_risk
            + 0.10 * trend_risk
            + 0.08 * min(1.0, max(0.0, demand_agg_risk))
            + 0.05 * obs.recent_violation_rate
            + 0.08 * external_risk
            + 0.10 * (obs.external_latency_signal_ms / max(1.0, obs.sla_threshold_ms))
        )
        self._risk_hist.append(blended_risk)
        smooth_risk = sum(self._risk_hist) / len(self._risk_hist)

        target = obs.active_instances
        recent_violation_spike = obs.recent_violation_rate >= 0.10
        near_threshold = obs.latency_p99_ms >= obs.sla_threshold_ms * 0.95
        sustained_risk = smooth_risk >= risk_up or trend_risk >= 0.55
        demand_rush = demand_agg_risk >= 0.65
        queue_pressure = obs.queue_len >= queue_budget * 0.75
        high_util = obs.utilization >= 0.72
        external_pressure = obs.external_latency_signal_ms >= obs.sla_threshold_ms * 0.90

        if (
            near_threshold
            or recent_violation_spike
            or demand_rush
            or queue_pressure
            or high_util
            or external_pressure
            or sustained_risk
        ):
            target += up_step
        elif (
            smooth_risk <= risk_down
            and obs.latency_p99_ms < obs.sla_threshold_ms * 0.70
            and obs.utilization < 0.55
            and obs.recent_violation_rate <= 0.05
            and obs.queue_len < queue_budget * 0.25
        ):
            target -= down_step

        return self.clamp(target, obs.min_instances, obs.max_instances)
