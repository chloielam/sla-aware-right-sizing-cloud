from __future__ import annotations

from collections import deque

from src.policies.base_policy import BasePolicy, Observation


class SLAAwareEMA(BasePolicy):
    algorithm_name = "ema_forecast"

    def __init__(self, name: str, params: dict | None = None) -> None:
        super().__init__(name, params)
        self._risk_hist: deque[float] = deque(maxlen=int(self.params.get("history_window", 12)))
        self._demand_hist: deque[float] = deque(maxlen=int(self.params.get("trend_window", 8)))
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
        risk_up = float(self.params.get("risk_up_threshold", 1.05))
        risk_down = float(self.params.get("risk_down_threshold", 0.40))
        up_step = int(self.params.get("scale_up_step", 1))
        down_step = int(self.params.get("scale_down_step", 1))
        ema_alpha = float(self.params.get("ema_alpha", 0.35))
        trend_gain = float(self.params.get("trend_gain", 1.4))
        volatility_gain = float(self.params.get("volatility_gain", 0.35))
        forecast_weight = float(self.params.get("forecast_weight", 0.70))

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
        queue_pressure = obs.queue_len / max(1.0, queue_budget)
        external_risk = obs.external_queue_signal / max(1.0, queue_budget * 1.3)
        ext_latency_ratio = obs.external_latency_signal_ms / max(1.0, obs.sla_threshold_ms)
        violation_pressure = obs.recent_violation_rate

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
            0.38 * latency_pressure
            + 0.30 * queue_pressure
            + 0.20 * forecast_gain
            + 0.15 * min(1.0, predicted_utilization)
            + 0.08 * ext_latency_ratio
            + 0.05 * external_risk
            + 0.07 * violation_pressure
            + 0.05 * vol_risk
            + 0.05 * pred_err_delta
        )
        self._risk_hist.append(blended_risk)
        smooth_risk = sum(self._risk_hist) / len(self._risk_hist)

        target = obs.active_instances
        if (
            obs.latency_p99_ms >= obs.sla_threshold_ms * 0.95
            or demand_gap_ratio >= 0.40
            or predicted_utilization >= 0.88
            or queue_pressure >= 0.85
            or obs.utilization >= 0.78
            or external_risk >= 0.50
            or smooth_risk >= risk_up
        ):
            target += up_step
        elif (
            smooth_risk <= risk_down
            and obs.latency_p99_ms <= obs.sla_threshold_ms * 0.72
            and obs.queue_len <= queue_budget * 0.20
            and obs.recent_violation_rate <= 0.04
            and obs.utilization <= 0.55
            and predicted_utilization <= 0.45
        ):
            target -= down_step

        return self.clamp(target, obs.min_instances, obs.max_instances)
