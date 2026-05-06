"""Parallel strategies and their per-step communication footprints.

A strategy returns a StepCost (raw modeled comm time + latency component)
for one optimizer step. The training loop applies the OverlapModel on top,
so overlap is orthogonal to the base strategy.
"""

from dataclasses import dataclass

from comm_model import (
    CommModel, Topology,
    allreduce_seconds, compressed_allreduce_seconds, p2p_seconds,
)


@dataclass
class StepCost:
    t_comm_total: float
    t_comm_latency: float


@dataclass
class StrategyConfig:
    name: str = "dp_allreduce"
    topology: Topology = Topology.RING_AR
    bytes_per_param: int = 2

    # dp_local_sgd
    local_steps: int = 1

    # dp_compressed_ar
    compression_ratio: float = 1.0

    # tp_allreduce
    tp_act_bytes_per_layer: int = 0
    tp_layers_per_step: int = 0

    # pp_p2p
    pp_microbatches: int = 1
    pp_act_bytes: int = 0


def _grad_bytes(num_params: int, bpp: int) -> int:
    return num_params * bpp


def step_cost(cfg: StrategyConfig, model: CommModel, num_params: int,
              step_idx: int) -> StepCost:
    """Modeled comm cost for one optimizer step under the given strategy."""
    if cfg.name == "dp_allreduce":
        n = _grad_bytes(num_params, cfg.bytes_per_param)
        t, lat = allreduce_seconds(model, n, cfg.topology)
        return StepCost(t, lat)

    if cfg.name == "dp_local_sgd":
        if (step_idx + 1) % max(1, cfg.local_steps) != 0:
            return StepCost(0.0, 0.0)
        n = _grad_bytes(num_params, cfg.bytes_per_param)
        t, lat = allreduce_seconds(model, n, cfg.topology)
        return StepCost(t, lat)

    if cfg.name == "dp_compressed_ar":
        n = _grad_bytes(num_params, cfg.bytes_per_param)
        t, lat = compressed_allreduce_seconds(
            model, n, cfg.compression_ratio, cfg.topology)
        return StepCost(t, lat)

    if cfg.name == "tp_allreduce":
        per_t, per_lat = allreduce_seconds(
            model, cfg.tp_act_bytes_per_layer, cfg.topology)
        return StepCost(cfg.tp_layers_per_step * per_t,
                        cfg.tp_layers_per_step * per_lat)

    if cfg.name == "pp_p2p":
        per_t, per_lat = p2p_seconds(model, cfg.pp_act_bytes)
        n_msgs = 2 * max(1, cfg.pp_microbatches)
        return StepCost(n_msgs * per_t, n_msgs * per_lat)

    if cfg.name == "single_gpu":
        return StepCost(0.0, 0.0)

    raise ValueError(f"unknown strategy: {cfg.name}")


STRATEGIES = [
    "single_gpu", "dp_allreduce", "dp_local_sgd", "dp_compressed_ar",
    "tp_allreduce", "pp_p2p",
]
