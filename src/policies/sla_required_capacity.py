from __future__ import annotations

from math import ceil

from src.policies.base_policy import BasePolicy, Observation


class SLARequiredCapacityPolicy(BasePolicy):
    algorithm_name = "required_capacity"

    def decide_target_instances(self, obs: Observation) -> int:
        target_utilization = float(self.params.get("target_utilization", 0.80))
        demand_mix = float(self.params.get("demand_mix", 0.50))
        latency_headroom_ratio = float(self.params.get("latency_headroom_ratio", 0.78))
        scale_up_cap = int(self.params.get("scale_up_cap", 2))
        scale_down_cap = int(self.params.get("scale_down_cap", 2))
        queue_budget = float(self.params.get("queue_budget", 16.0))

        planning_demand = max(
            obs.demand_qps,
            demand_mix * obs.predicted_demand_qps + (1.0 - demand_mix) * obs.demand_qps,
        )
        safe_capacity_per_instance = max(1e-6, obs.service_rate_qps_per_instance * target_utilization)
        raw_target = ceil(planning_demand / safe_capacity_per_instance)

        if obs.latency_p99_ms >= latency_headroom_ratio * obs.sla_threshold_ms:
            raw_target += 1

        raw_target = self.clamp(raw_target, obs.min_instances, obs.max_instances)

        if raw_target > obs.active_instances:
            return min(raw_target, obs.active_instances + scale_up_cap)

        can_scale_down = (
            obs.queue_len <= queue_budget * 0.15
            and obs.latency_p99_ms <= obs.sla_threshold_ms * 0.70
            and obs.recent_violation_rate <= 0.03
            and obs.utilization <= target_utilization * 0.80
        )
        if not can_scale_down:
            return obs.active_instances

        floor_target = max(obs.active_instances - scale_down_cap, raw_target)
        return self.clamp(floor_target, obs.min_instances, obs.max_instances)
