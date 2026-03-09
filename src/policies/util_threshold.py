from __future__ import annotations

from collections import deque

from src.policies.base_policy import BasePolicy, Observation


class UtilThresholdPolicy(BasePolicy):
    def __init__(self, name: str, params: dict | None = None) -> None:
        super().__init__(name, params)
        self._lat_hist: deque[float] = deque(maxlen=int(self.params.get("history_window", 8)))

    def decide_target_instances(self, obs: Observation) -> int:
        self._lat_hist.append(obs.latency_p99_ms)
        up_util = float(self.params.get("scale_up_util", 0.75))
        down_util = float(self.params.get("scale_down_util", 0.40))
        up_step = int(self.params.get("scale_up_step", 1))
        down_step = int(self.params.get("scale_down_step", 1))
        queue_up = float(self.params.get("queue_up_threshold", 20.0))
        queue_down = float(self.params.get("queue_down_threshold", 5.0))

        target = obs.active_instances
        if obs.utilization > up_util or obs.queue_len > queue_up:
            target += up_step
        elif (
            obs.utilization < down_util
            and obs.queue_len < queue_down
            and obs.latency_p99_ms < obs.sla_threshold_ms * 0.8
        ):
            target -= down_step
        return self.clamp(target, obs.min_instances, obs.max_instances)


class ReactivePolicy(BasePolicy):
    def __init__(self, name: str, params: dict | None = None) -> None:
        super().__init__(name, params)
        self._viol_hist: deque[float] = deque(maxlen=int(self.params.get("history_window", 10)))

    def decide_target_instances(self, obs: Observation) -> int:
        violation = 1.0 if obs.latency_p99_ms > obs.sla_threshold_ms else 0.0
        self._viol_hist.append(violation)
        recent = sum(self._viol_hist) / max(1, len(self._viol_hist))

        up_step = int(self.params.get("scale_up_step", 2))
        down_step = int(self.params.get("scale_down_step", 1))
        recovery_ratio = float(self.params.get("recovery_latency_ratio", 0.65))
        quiet_violation_rate = float(self.params.get("quiet_violation_rate", 0.05))

        target = obs.active_instances
        if obs.latency_p99_ms > obs.sla_threshold_ms or recent > 0.25:
            target += up_step
        elif (
            obs.latency_p99_ms < obs.sla_threshold_ms * recovery_ratio
            and obs.utilization < 0.55
            and recent <= quiet_violation_rate
        ):
            target -= down_step

        return self.clamp(target, obs.min_instances, obs.max_instances)
