# commresearch

Autonomous-research harness for **s-step / communication-avoiding methods**
in LLM training, run on a single GPU with **modeled** multi-node
communication (Hockney α–β–γ) and explicit compute–comm overlap.

This is a hard pivot of [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
It keeps autoresearch's autonomous-loop discipline (commit / run / measure
/ keep-or-revert / log) and 5-minute training budget, but reinterprets
that budget as **simulated multi-node wall-clock** under a configurable
α–β–γ comm model. The goal is to give a single-GPU box (e.g. a single
DGX Spark) a coherent way to study communication-avoidance ideas — local
SGD, compressed all-reduce, polynomial-basis Krylov / NS variants,
randomized sketched orthogonalization, overlap strategies, etc. — by
ranking them under modeled regimes you can sweep at will.

## What "comm-aware" actually means here

- The 5-minute `TIME_BUDGET` is no longer pure compute. Each training
  step's modeled communication time (after overlap) is added to the
  budget counter.
- Real GPU compute runs at full speed; the simulated comm time advances
  the budget so methods that reduce comm fit more steps in the same
  simulated budget. Real wall-clock per experiment is therefore ≤ 5 min
  in practice.
- `val_bpb` (validation bits per byte) remains the ground-truth metric.
  Lower is better, vocab-independent.
- Each run reports `val_bpb` together with `(α, β, γ, P)`, strategy,
  overlap settings, and `comm_fraction` so its regime is fully declared.

This is a **consistency** tool, not a **prediction** tool. It rank-orders
methods consistently across modeled regimes; absolute numbers won't match
any specific cluster.

## What's modeled

- **α–β–γ collective times** (closed-form): ring AR, recursive doubling
  AR, halving-doubling AR, tree AR, compressed AR, point-to-point.
- **Overlap as a first-class, orthogonal axis**: `OverlapModel(factor,
  window_fraction, hide_latency)` applies on top of any base strategy.
- **Parallel strategies**: data-parallel all-reduce, local-SGD,
  compressed AR, tensor-parallel AR, pipeline-parallel P2P, single-GPU
  baseline. Add new strategies in `strategies.py`.

## What's NOT modeled (be honest with yourself)

- Worker-divergence / staleness convergence cost — local-SGD-style
  methods will be over-credited because their *quality* loss isn't
  simulated.
- Network congestion, NUMA effects, message-size knees.
- Reduce-scatter / all-gather pipelining beyond the first-order overlap
  model.

Calibrate your conclusions accordingly — the harness is for ranking
methods, not for predicting cluster numbers.

## Files

```
prepare.py        — read-only constants, data prep + runtime utilities
train.py          — model, optimizer, training loop, comm config block
comm_model.py     — α–β–γ Hockney model + overlap (pure Python)
strategies.py     — parallel strategy registry, per-step comm volume
program.md        — agent instructions / "skill"
CLAUDE.md         — Claude Code repository guidance
pyproject.toml    — frozen dependencies
```

## Quick start

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
uv run prepare.py                  # one-time data + tokenizer (~2 min)
uv run train.py > run.log 2>&1     # one experiment, ≤5 min wall-clock
grep "^val_bpb:\|^comm_fraction:\|^sim_\|^strategy:\|^overlap_" run.log
```

## Configuring the comm regime

Edit the `COMM` / `OVERLAP` / `STRATEGY` block at the top of `train.py`.
Convenience presets ship in `comm_model.PRESETS` and
`comm_model.OVERLAP_PRESETS`:

```python
COMM = PRESETS["ndr_ib_8"]              # alpha=5us, beta=1/50GB/s, P=8
OVERLAP = OVERLAP_PRESETS["ddp_typical"] # factor=0.5, window_fraction=0.6
STRATEGY = StrategyConfig(
    name="dp_allreduce",                 # or dp_local_sgd, dp_compressed_ar, ...
    topology=Topology.RING_AR,
    bytes_per_param=2,
    local_steps=1,
    compression_ratio=1.0,
)
```

## Running the loop

Spin up Claude Code (or your agent of choice) at the repo root:

```
Read program.md and let's set up a new commresearch run.
```

`program.md` is the skill — it walks the agent through branch setup,
baseline, the phased experiment plan (strategy → overlap → α-β
sensitivity → s-step variants → reinvestment), and the keep / discard
discipline.

## Lineage

- Forked-in-spirit from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
  The training stack (GPT model, Muon+AdamW, `prepare.py`) is essentially
  unchanged. The pivot is the comm model, the overlap accounting, the
  strategy registry, and the simulated-time budget.
- The communication model follows Hockney 1994 (α–β) and the standard
  references on collective topologies (Thakur–Rabenseifner–Gropp 2005;
  Chan et al. 2007).

## License

MIT
