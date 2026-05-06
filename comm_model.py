"""Hockney alpha-beta-gamma communication model with explicit overlap.

Pure-Python (no torch). Closed-form per-collective times for the standard
topologies plus a separable overlap-with-compute accounting so the training
loop can simulate multi-node wall-clock on a single GPU.

References:
  Hockney, "The communication challenge for MPP" (Parallel Comput., 1994)
  Thakur, Rabenseifner, Gropp, "Optimization of collective communication
    operations in MPICH" (IJHPCA, 2005)
  Chan et al., "Collective communication: theory, practice, experience"
    (CCPE, 2007)
"""

from dataclasses import dataclass
from enum import Enum
import math


class Topology(str, Enum):
    RING_AR = "ring_ar"
    RD_AR = "rd_ar"
    HD_AR = "hd_ar"
    TREE_AR = "tree_ar"


@dataclass
class CommModel:
    alpha: float        # latency, seconds per message
    beta: float         # inverse bandwidth, seconds per byte
    gamma: float = 0.0  # per-byte codec cost (compressed AR)
    P: int = 1          # world size


def allreduce_seconds(model: CommModel, n_bytes: int,
                      topology: Topology = Topology.RING_AR) -> tuple[float, float]:
    """Return (total_seconds, latency_seconds) for an all-reduce of n_bytes."""
    P = model.P
    if P <= 1 or n_bytes <= 0:
        return 0.0, 0.0
    a, b = model.alpha, model.beta
    if topology == Topology.RING_AR:
        n_msgs = P - 1
        bw = 2.0 * (P - 1) / P * b * n_bytes
    elif topology == Topology.RD_AR:
        n_msgs = math.ceil(math.log2(P))
        bw = math.log2(P) * b * n_bytes
    elif topology == Topology.HD_AR:
        n_msgs = 2 * math.ceil(math.log2(P))
        bw = 2.0 * (P - 1) / P * b * n_bytes
    elif topology == Topology.TREE_AR:
        n_msgs = 2 * math.ceil(math.log2(P))
        bw = 2.0 * math.log2(P) * b * n_bytes
    else:
        raise ValueError(f"unknown topology: {topology}")
    lat = n_msgs * a
    return lat + bw, lat


def compressed_allreduce_seconds(model: CommModel, n_bytes: int, ratio: float,
                                 topology: Topology = Topology.RING_AR
                                 ) -> tuple[float, float]:
    n_after = max(1, int(n_bytes * max(0.0, min(1.0, ratio))))
    codec = model.gamma * max(0, n_bytes)
    t_ar, lat = allreduce_seconds(model, n_after, topology)
    return codec + t_ar, lat


def p2p_seconds(model: CommModel, n_bytes: int) -> tuple[float, float]:
    if n_bytes <= 0:
        return 0.0, 0.0
    return model.alpha + model.beta * n_bytes, model.alpha


@dataclass
class OverlapModel:
    """Communication-compute overlap.

    factor in [0, 1]: fraction of the bandwidth-dominated portion of T_comm
    that can be hidden behind a compute window. Latency is on the critical
    path by default (message setup is hard to hide in practice).

    window_fraction: compute-window size, as a fraction of per-step compute
    time. Typical 0.5-0.7 for backward-pass overlap in DDP-style overlap.
    """
    factor: float = 0.0
    window_fraction: float = 0.6
    hide_latency: bool = False


def apply_overlap(t_total: float, t_latency: float,
                  step_compute_seconds: float,
                  overlap: OverlapModel) -> float:
    """On-critical-path comm time after applying the overlap model."""
    f = max(0.0, min(1.0, overlap.factor))
    window = max(0.0, overlap.window_fraction) * max(0.0, step_compute_seconds)
    t_bw = max(0.0, t_total - t_latency)
    hidden_bw = min(t_bw, f * window)
    on_path_bw = t_bw - hidden_bw
    if overlap.hide_latency:
        remaining = max(0.0, f * window - hidden_bw)
        on_path_lat = max(0.0, t_latency - remaining)
    else:
        on_path_lat = t_latency
    return on_path_bw + on_path_lat


# Convenience presets for common HPC regimes. Use as starting points; sweep.
PRESETS = {
    # ~Single rack of NDR InfiniBand, ~50 GB/s per NIC, P=8
    "ndr_ib_8": CommModel(alpha=5e-6, beta=1.0 / 50e9, gamma=0.0, P=8),
    # 200 GbE inter-Spark, 25 GB/s, P=2
    "spark_200gbe_2": CommModel(alpha=10e-6, beta=1.0 / 25e9, gamma=0.0, P=2),
    # Slow ethernet, 1 GB/s, P=8 (commodity cluster floor)
    "gbe_8": CommModel(alpha=50e-6, beta=1.0 / 1e9, gamma=0.0, P=8),
    # Idealized intra-node NVLink, 450 GB/s, P=8 (for control runs)
    "nvlink_8": CommModel(alpha=1e-6, beta=1.0 / 450e9, gamma=0.0, P=8),
}


OVERLAP_PRESETS = {
    "none": OverlapModel(factor=0.0),
    "ddp_typical": OverlapModel(factor=0.5, window_fraction=0.6),
    "fsdp_aggressive": OverlapModel(factor=0.85, window_fraction=0.7,
                                    hide_latency=False),
    "ideal": OverlapModel(factor=1.0, window_fraction=1.0, hide_latency=True),
}
