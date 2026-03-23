from __future__ import annotations

from collections import deque

from src.policies.base_policy import BasePolicy, Observation


class SLAAwareEMA(BasePolicy):
    algorithm_name = "ema_forecast"

    def __init__(self, name: str, params: dict | None = None) -> None:
        super().__init__(name, params)
        self._risk_hist: deque[float] = deque(maxlen=int(self.params.get("history_window", 12)))
        self._demand_hist: deque[float] = deque(maxlen=int(self.params.get("trend_window", 8)))
        self._quiet_hist: deque[float] = deque(maxlen=int(self.params.get("quiet_steps_before_hold", 4)))
        self._ema_fast: float | None = None
        self._ema_slow: float | None = None
        self._prev_pred_err: float | None = None

    def _std_ratio(self, values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(1, len(values))
        return var**0.5 / max(1e-6, mean)

    def decide_target_instances(self, obs: Observation) -> int:
        self._demand_hist.append(obs.demand_qps)
        queue_budget = float(self.params.get("queue_budget", 15.0))
        risk_up = float(self.params.get("risk_up_threshold", 1.12))
        risk_down = float(self.params.get("risk_down_threshold", 0.48))
        up_step = int(self.params.get("scale_up_step", 1))
        down_step = int(self.params.get("scale_down_step", 1))
        ema_alpha = float(self.params.get("ema_alpha", 0.28))
        trend_gain = float(self.params.get("trend_gain", 0.95))
        volatility_gain = float(self.params.get("volatility_gain", 0.22))
        forecast_weight = float(self.params.get("forecast_weight", 0.38))
        latency_weight = float(self.params.get("latency_weight", 0.26))
        queue_weight = float(self.params.get("queue_weight", 0.16))
        predicted_utilization_weight = float(self.params.get("predicted_utilization_weight", 0.12))
        external_latency_weight = float(self.params.get("external_latency_component_weight", 0.04))
        external_queue_weight = float(self.params.get("external_queue_weight", 0.03))
        violation_weight = float(self.params.get("violation_weight", 0.07))
        volatility_weight = float(self.params.get("volatility_weight", 0.05))
        prediction_error_delta_weight = float(self.params.get("prediction_error_delta_weight", 0.05))
        max_slack_ratio_for_scale_up = float(self.params.get("max_slack_ratio_for_scale_up", 2.2))
        min_scale_up_utilization = float(self.params.get("min_scale_up_utilization", 0.42))
        require_hard_signal_for_risk_scale = bool(self.params.get("require_hard_signal_for_risk_scale", True))
        min_demand_rush = float(self.params.get("min_demand_rush_ratio", 0.32))
        downscale_latency_ratio = float(self.params.get("downscale_latency_ratio", 0.80))
        downscale_utilization = float(self.params.get("downscale_utilization", 0.60))
        downscale_queue_ratio = float(self.params.get("downscale_queue_ratio", 0.30))
        quiet_violation_rate = float(self.params.get("quiet_violation_rate", 0.03))

        alpha = max(1e-3, min(0.99, ema_alpha))
        demand_now = obs.demand_qps
        if self._ema_fast is None:
            self._ema_fast = demand_now
            self._ema_slow = demand_now
        else:
            self._ema_fast = alpha * demand_now + (1 - alpha) * self._ema_fast
            self._ema_slow = alpha * demand_now * 0.65 + (1 - alpha) * self._ema_slow

        trend = (self._ema_fast - self._ema_slow) * trend_gain
        forecast_demand = obs.predicted_demand_qps + trend
        demand_gap = max(0.0, forecast_demand - obs.demand_qps)
        demand_gap_ratio = demand_gap / max(1.0, obs.demand_qps)
        forecast_gain = forecast_weight * demand_gap_ratio
        predicted_utilization = (
            obs.utilization * (forecast_demand / max(1e-6, obs.demand_qps))
            if obs.demand_qps > 0
            else obs.utilization
        )

        latency_pressure = obs.latency_p99_ms / max(1.0, obs.sla_threshold_ms)
        softened_latency_pressure = max(0.0, latency_pressure - 0.72)
        queue_pressure = obs.queue_len / max(1.0, queue_budget)
        external_risk = obs.external_queue_signal / max(1.0, queue_budget * 1.3)
        ext_latency_ratio = obs.external_latency_signal_ms / max(1.0, obs.sla_threshold_ms)
        violation_pressure = obs.recent_violation_rate
        capacity_slack_ratio = max(0.0, (1.0 / max(1e-6, obs.utilization)) - 1.0) if obs.utilization > 0 else 99.0

        pred_err = max(0.0, obs.predicted_demand_qps - obs.demand_qps) / max(1.0, obs.demand_qps)
        if self._prev_pred_err is None:
            pred_err_delta = 0.0
        else:
            pred_err_delta = max(0.0, pred_err - self._prev_pred_err)
        self._prev_pred_err = pred_err

        vol_window = list(self._demand_hist)
        demand_volatility = self._std_ratio(vol_window)
        vol_risk = min(1.0, demand_volatility * volatility_gain)

        blended_risk = (
            latency_weight * softened_latency_pressure
            + queue_weight * queue_pressure
            + 0.20 * forecast_gain
            + predicted_utilization_weight * min(1.0, predicted_utilization)
            + external_latency_weight * ext_latency_ratio
            + external_queue_weight * external_risk
            + violation_weight * violation_pressure
            + volatility_weight * vol_risk
            + prediction_error_delta_weight * pred_err_delta
        )
        self._risk_hist.append(blended_risk)
        smooth_risk = sum(self._risk_hist) / len(self._risk_hist)

        target = obs.active_instances
        near_threshold = obs.latency_p99_ms >= obs.sla_threshold_ms * 0.95
        queue_alarm = queue_pressure >= 0.85
        demand_alarm = demand_gap_ratio >= min_demand_rush
        util_alarm = predicted_utilization >= 0.90 or obs.utilization >= 0.82
        hard_signal = near_threshold or queue_alarm or demand_alarm
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
            and predicted_utilization < 0.78
            and obs.recent_violation_rate <= quiet_violation_rate
        )
        suppress_for_quiet = len(self._quiet_hist) == self._quiet_hist.maxlen and sum(self._quiet_hist) == len(self._quiet_hist)
        risk_scale_allowed = smooth_risk >= risk_up and (
            not require_hard_signal_for_risk_scale or hard_signal or util_alarm
        )

        if (
            not suppress_for_slack
            and not suppress_for_quiet
            and (
                near_threshold
                or demand_alarm
                or predicted_utilization >= 0.90
                or queue_alarm
                or obs.utilization >= 0.82
                or external_risk >= 0.65
                or risk_scale_allowed
            )
        ):
            target += up_step
        elif (
            smooth_risk <= risk_down
            and obs.latency_p99_ms <= obs.sla_threshold_ms * downscale_latency_ratio
            and obs.queue_len <= queue_budget * downscale_queue_ratio
            and obs.recent_violation_rate <= quiet_violation_rate
            and obs.utilization <= downscale_utilization
            and predicted_utilization <= downscale_utilization
        ):
            target -= down_step

        return self.clamp(target, obs.min_instances, obs.max_instances)
