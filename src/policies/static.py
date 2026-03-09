from __future__ import annotations

from src.policies.base_policy import BasePolicy, Observation


class StaticPolicy(BasePolicy):
    def decide_target_instances(self, obs: Observation) -> int:
        target = int(self.params.get("target_instances", obs.active_instances))
        return self.clamp(target, obs.min_instances, obs.max_instances)
