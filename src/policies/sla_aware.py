from __future__ import annotations

from collections import deque

from src.policies.base_policy import BasePolicy, Observation


class SLAAwarePolicy(BasePolicy):
    algorithm_name = "heuristic_risk"

    def __init__(self, name: str, params: dict | None = None) -> None:
        super().__init__(name, params)
        self._risk_hist: deque[float] = deque(maxlen=int(self.params.get("history_window", 12)))
        self._demand_hist: deque[float] = deque(maxlen=int(self.params.get("trend_window", 6)))
        self._quiet_hist: deque[float] = deque(maxlen=int(self.params.get("quiet_steps_before_hold", 4)))

    def decide_target_instances(self, obs: Observation) -> int:
        self._demand_hist.append(obs.demand_qps)
        queue_budget = float(self.params.get("queue_budget", 15.0))
        risk_up = float(self.params.get("risk_up_threshold", 1.15))
        risk_down = float(self.params.get("risk_down_threshold", 0.50))
        up_step = int(self.params.get("scale_up_step", 1))
        down_step = int(self.params.get("scale_down_step", 1))
        prediction_weight = float(self.params.get("prediction_weight", 0.45))
        queue_weight = float(self.params.get("queue_weight", 0.18))
        trend_weight = float(self.params.get("trend_weight", 0.06))
        demand_weight = float(self.params.get("demand_weight", 0.06))
        violation_weight = float(self.params.get("violation_weight", 0.05))
        external_queue_weight = float(self.params.get("external_queue_weight", 0.04))
        external_latency_weight = float(self.params.get("external_latency_component_weight", 0.05))
        max_slack_ratio_for_scale_up = float(self.params.get("max_slack_ratio_for_scale_up", 2.5))
        min_scale_up_utilization = float(self.params.get("min_scale_up_utilization", 0.40))
        require_hard_signal_for_risk_scale = bool(self.params.get("require_hard_signal_for_risk_scale", True))
        min_demand_rush = float(self.params.get("min_demand_rush_ratio", 0.40))
        downscale_latency_ratio = float(self.params.get("downscale_latency_ratio", 0.78))
        downscale_utilization = float(self.params.get("downscale_utilization", 0.62))
        downscale_queue_ratio = float(self.params.get("downscale_queue_ratio", 0.35))
        quiet_violation_rate = float(self.params.get("quiet_violation_rate", 0.03))

        if len(self._demand_hist) >= 2:
            trend = (self._demand_hist[-1] - self._demand_hist[0]) / max(1e-6, self._demand_hist[0])
        else:
            trend = 0.0

        queue_risk = obs.queue_len / max(1.0, queue_budget)
        trend_risk = max(0.0, trend)
        demand_agg_risk = (obs.predicted_demand_qps - obs.demand_qps) / max(1e-6, obs.demand_qps)
        external_risk = obs.external_queue_signal / max(1.0, queue_budget * 1.5)
        latency_pressure = obs.latency_p99_ms / max(1.0, obs.sla_threshold_ms)
        softened_latency_pressure = max(0.0, latency_pressure - 0.70)
        predicted_utilization = obs.utilization * (1.0 + max(0.0, demand_agg_risk))
        capacity_slack_ratio = max(0.0, (1.0 / max(1e-6, obs.utilization)) - 1.0) if obs.utilization > 0 else 99.0
        blended_risk = (
            prediction_weight * max(0.0, min(2.0, softened_latency_pressure))
            + queue_weight * queue_risk
            + trend_weight * trend_risk
            + demand_weight * min(1.0, max(0.0, demand_agg_risk))
            + violation_weight * obs.recent_violation_rate
            + external_queue_weight * external_risk
            + external_latency_weight * (obs.external_latency_signal_ms / max(1.0, obs.sla_threshold_ms))
        )
        self._risk_hist.append(blended_risk)
        smooth_risk = sum(self._risk_hist) / len(self._risk_hist)

        target = obs.active_instances
        recent_violation_spike = obs.recent_violation_rate >= 0.10
        near_threshold = obs.latency_p99_ms >= obs.sla_threshold_ms * 0.95
        sustained_risk = smooth_risk >= risk_up or trend_risk >= 0.70
        demand_rush = demand_agg_risk >= min_demand_rush
        queue_pressure = obs.queue_len >= queue_budget * 0.75
        high_util = obs.utilization >= 0.82
        external_pressure = obs.external_latency_signal_ms >= obs.sla_threshold_ms * 0.95
        hard_signal = near_threshold or queue_pressure or demand_rush
        quiet_state = (
            obs.utilization <= min_scale_up_utilization
            and obs.queue_len <= queue_budget * 0.15
            and obs.recent_violation_rate <= quiet_violation_rate
            and obs.latency_p99_ms <= obs.sla_threshold_ms * 0.55
        )
        self._quiet_hist.append(1.0 if quiet_state else 0.0)
        suppress_for_slack = (
            capacity_slack_ratio >= max_slack_ratio_for_scale_up
            and not hard_signal
            and not recent_violation_spike
            and predicted_utilization < 0.72
        )
        suppress_for_quiet = len(self._quiet_hist) == self._quiet_hist.maxlen and sum(self._quiet_hist) == len(self._quiet_hist)
        risk_scale_allowed = sustained_risk and (
            not require_hard_signal_for_risk_scale or hard_signal or predicted_utilization >= 0.85
        )

        if (
            not suppress_for_slack
            and not suppress_for_quiet
            and (
                near_threshold
                or recent_violation_spike
                or demand_rush
                or queue_pressure
                or high_util
                or external_pressure
                or risk_scale_allowed
            )
        ):
            target += up_step
        elif (
            smooth_risk <= risk_down
            and obs.latency_p99_ms < obs.sla_threshold_ms * downscale_latency_ratio
            and obs.utilization < downscale_utilization
            and obs.recent_violation_rate <= quiet_violation_rate
            and obs.queue_len < queue_budget * downscale_queue_ratio
            and predicted_utilization < downscale_utilization
        ):
            target -= down_step

        return self.clamp(target, obs.min_instances, obs.max_instances)
