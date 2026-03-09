from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean

from src.policies.base_policy import BasePolicy, Observation


@dataclass
class TracePoint:
    timestamp: int
    demand_qps: float
    external_queue: float
    external_latency_ms: float
    gpu_util_pct: float
    memory_util: float


@dataclass
class SimulationConfig:
    min_instances: int
    max_instances: int
    initial_instances: int
    service_rate_qps_per_instance: float
    step_seconds: int
    boot_delay_steps: int
    cooldown_steps: int
    base_latency_ms: float
    queue_latency_factor_ms: float
    utilization_penalty_ms: float
    external_latency_weight: float
    sla_threshold_ms: float
    cost_per_instance_hour: float
    max_steps: int
    lookahead_steps: int


class CloudSimulator:
    def __init__(self, config: SimulationConfig, policy: BasePolicy, trace: list[TracePoint]) -> None:
        if not trace:
            raise ValueError("trace cannot be empty")
        self.config = config
        self.policy = policy
        self.trace = trace[: config.max_steps]

    def run(self) -> tuple[dict, list[dict]]:
        cfg = self.config
        active = max(cfg.min_instances, min(cfg.max_instances, cfg.initial_instances))
        pending_scaleups: list[tuple[int, int]] = []  # (ready_step, count)
        queue_len = 0.0
        cooldown_until = -1
        p99_hist: list[float] = []
        util_hist: list[float] = []
        inst_hist: list[int] = []
        recent_viol = deque(maxlen=20)
        time_series: list[dict] = []
        total_requests = 0.0
        violating_requests = 0.0
        billed_instance_hours = 0.0

        for step, point in enumerate(self.trace):
            newly_ready = 0
            still_pending: list[tuple[int, int]] = []
            for ready_step, count in pending_scaleups:
                if ready_step <= step:
                    newly_ready += count
                else:
                    still_pending.append((ready_step, count))
            pending_scaleups = still_pending
            active = max(cfg.min_instances, min(cfg.max_instances, active + newly_ready))

            arrivals = max(0.0, point.demand_qps) * cfg.step_seconds
            capacity = active * cfg.service_rate_qps_per_instance * cfg.step_seconds
            queue_len = max(0.0, queue_len + arrivals - capacity)
            utilization = arrivals / max(1e-6, capacity)
            util_hist.append(min(utilization, 1.5))
            inst_hist.append(active)

            model_p99 = (
                cfg.base_latency_ms
                + cfg.queue_latency_factor_ms * queue_len
                + cfg.utilization_penalty_ms * max(0.0, utilization - 1.0)
            )
            external_tail_increase = max(0.0, point.external_latency_ms - cfg.base_latency_ms)
            latency_p99 = model_p99 + cfg.external_latency_weight * external_tail_increase
            latency_p99 = max(1.0, latency_p99)
            p99_hist.append(latency_p99)

            lookahead = self.trace[step + 1 : step + 1 + cfg.lookahead_steps]
            predicted_demand = mean([x.demand_qps for x in lookahead]) if lookahead else point.demand_qps
            predicted_latency = (
                model_p99
                + cfg.external_latency_weight * max(0.0, point.external_latency_ms - cfg.base_latency_ms)
                + max(0.0, predicted_demand - point.demand_qps) * 120.0
                + point.external_queue * 0.2
            )
            recent_violation_rate = sum(recent_viol) / max(1, len(recent_viol))

            obs = Observation(
                step=step,
                timestamp=point.timestamp,
                demand_qps=point.demand_qps,
                queue_len=queue_len,
                latency_p99_ms=latency_p99,
                utilization=utilization,
                active_instances=active,
                min_instances=cfg.min_instances,
                max_instances=cfg.max_instances,
                sla_threshold_ms=cfg.sla_threshold_ms,
                predicted_demand_qps=predicted_demand,
                predicted_latency_ms=predicted_latency,
                recent_violation_rate=recent_violation_rate,
                external_queue_signal=point.external_queue,
                external_latency_signal_ms=point.external_latency_ms,
                gpu_util_pct=point.gpu_util_pct,
                memory_util=point.memory_util,
            )
            target = self.policy.decide_target_instances(obs)

            if target > active:
                pending_scaleups.append((step + cfg.boot_delay_steps, target - active))
            elif target < active and step >= cooldown_until:
                active = target
                cooldown_until = step + cfg.cooldown_steps

            violated = latency_p99 > cfg.sla_threshold_ms
            recent_viol.append(1.0 if violated else 0.0)
            total_requests += arrivals
            if violated:
                violating_requests += arrivals

            billed_instance_hours += active * (cfg.step_seconds / 3600.0)
            time_series.append(
                {
                    "step": step,
                    "timestamp": point.timestamp,
                    "policy": self.policy.name,
                    "prediction_algorithm": getattr(self.policy, "algorithm_name", self.policy.name),
                    "instances": active,
                    "target_instances": target,
                    "arrivals": round(arrivals, 4),
                    "queue_len": round(queue_len, 4),
                    "utilization": round(utilization, 4),
                    "latency_p99_ms": round(latency_p99, 4),
                    "sla_violation": int(violated),
                }
            )

        overall_p99 = sorted(p99_hist)[int(0.99 * (len(p99_hist) - 1))]
        summary = {
            "policy": self.policy.name,
            "prediction_algorithm": getattr(self.policy, "algorithm_name", self.policy.name),
            "steps": len(self.trace),
            "step_seconds": cfg.step_seconds,
            "simulated_hours": round(len(self.trace) * cfg.step_seconds / 3600.0, 3),
            "avg_instances": round(sum(inst_hist) / max(1, len(inst_hist)), 4),
            "cost_instance_hours": round(billed_instance_hours, 4),
            "cost_estimate": round(billed_instance_hours * cfg.cost_per_instance_hour, 4),
            "sla_violation_rate": round(violating_requests / max(1e-6, total_requests), 6),
            "tail_latency_p99_ms": round(overall_p99, 4),
            "resource_efficiency": round(min(1.0, sum(util_hist) / max(1, len(util_hist))), 4),
        }
        return summary, time_series
