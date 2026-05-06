# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is (commresearch)

`commresearch` is a comm-aware autonomous-research harness for **s-step
methods and their generalizations** (CA-Krylov, polynomial-basis optimizer
inner solves, randomized sketching, communication-avoiding outer
optimizers, compressed/pipelined collectives, overlap strategies). It is a
hard pivot of [karpathy/autoresearch](https://github.com/karpathy/autoresearch):
it keeps autoresearch's loop discipline (commit / run / measure / keep-or-revert
/ log) and 5-minute training budget, but reinterprets that budget as
**simulated multi-node wall-clock** under an explicit α–β–γ communication
model with separable compute–comm overlap.

The point: a single Spark (or any single GPU) can now produce coherent,
reproducible rankings of communication-avoiding methods, sweeping over
modeled `(α, β, γ, P)` regimes that no single physical fabric can give you.
It is a **consistency** tool, not a **prediction** tool — it rank-orders
methods correctly across modeled regimes; absolute numbers won't match any
specific cluster.

## How the loop works

1. The agent edits `train.py` (and `comm_model.py` / `strategies.py` when
   adding new collectives or strategies). The optimizer, training loop,
   and parallel strategy are all fair game.
2. It runs `uv run train.py > run.log 2>&1`. The training loop simulates
   a 5-minute multi-node wall-clock by adding modeled communication time
   (after overlap) to the budget counter on every step. Real GPU compute
   runs at full speed; the simulated "comm" time advances the budget so
   methods that reduce comm fit more steps in the same simulated budget.
3. It greps `val_bpb` (validation bits per byte) plus the new comm-aware
   diagnostics from the log. Lower `val_bpb` is better.
4. If the metric improved, the commit is kept and the branch advances. If
   not, it's reverted. Crashes are logged and skipped.
5. Each result is appended to `results.tsv` with the comm-aware schema
   (see `program.md`).
6. The loop runs forever — the agent does not stop to ask whether to continue.

## Files that matter

- `prepare.py` — **read-only.** Constants (incl. `TIME_BUDGET`), data
  download, BPE tokenizer, dataloader, and the `evaluate_bpb` ground-truth
  metric. Do not modify.
- `train.py` — main training loop, GPT model, Muon+AdamW optimizer.
  Now also wires in the comm model: `COMM`, `OVERLAP`, `STRATEGY` config
  block plus simulated-time budget accounting.
- `comm_model.py` — α–β–γ Hockney model, closed-form collective times for
  ring/RD/HD/tree all-reduce and compressed AR, plus the OverlapModel.
  Pure Python, no torch deps. Edit to add new collectives or refine the
  overlap accounting.
- `strategies.py` — per-step communication footprint for each parallel
  strategy. Edit to add new strategies (FSDP-style, ZeRO-3, async, etc.).
- `program.md` — agent instructions for the comm-aware research loop. The
  "skill" handed to the agent at run start.
- `pyproject.toml` — dependencies are frozen. The agent cannot add new ones.

## Hard constraints

- Do **not** edit `prepare.py` or the eval harness (`evaluate_bpb`). The
  ground-truth metric must be unchanged across experiments.
- Do **not** add dependencies. Stay within `pyproject.toml`.
- Each run targets ~5 min *simulated* wall-clock. Real wall-clock may be
  shorter (if modeled comm is high) or up to ~5 min (if comm fraction is
  small). The 10-min real-wall-clock kill switch always applies.
- Work on a dedicated `commresearch/<tag>` branch. Never run on `main`.
- VRAM is a soft constraint; meaningful `val_bpb` gains under the chosen
  comm regime justify some growth, but no blow-ups.
- Simpler is better. Numerical-noise gains that add hacky code do not pass.

## What's new vs. autoresearch

- **Communication is modeled, not measured.** A configurable `CommModel`
  defines `(α, β, γ, P)` and a topology; closed-form formulas in
  `comm_model.py` give per-collective time. The `OverlapModel` separately
  controls how much of comm is hidden behind compute.
- **Strategy is a first-class axis.** `STRATEGIES` in `strategies.py`
  registers DP-AR, local-SGD, compressed-AR, TP-AR, PP-P2P, and a
  `single_gpu` baseline. New strategies plug in by adding a `step_cost`
  case.
- **Overlap is orthogonal to base strategy.** `OverlapModel(factor,
  window_fraction, hide_latency)` applies to whatever the base strategy
  produced. Sweep overlap independently of the strategy.
- **Simulated-time budget.** `total_training_time + simulated_comm_time`
  is checked against `TIME_BUDGET` instead of just compute time. `val_bpb`
  rankings reflect comm-aware progress per second.
- **Extended summary.** Each run reports `sim_world_size`, `alpha_us`,
  `beta_GBs`, `gamma_per_byte_ns`, `overlap_factor`, `comm_fraction`,
  `sim_comm_seconds_raw/eff`, `overlap_hidden_frac` alongside `val_bpb`.

## Use case: s-step methods research

The harness was built for s-step / communication-avoiding research on a
single Spark. Concretely:

- **Polynomial-basis NS in Muon.** The Muon orthogonalization is an
  s-step-style polynomial iteration; sweep monomial → Newton/Leja →
  Chebyshev bases for stability and FLOP-per-step trade-offs. Single-GPU,
  measurable in `val_bpb`.
- **Randomized sketched orthogonalization.** Sketch + NS-polish variants;
  particularly informative under low-bandwidth `CommModel` regimes
  because the sketch reduces effective data motion.
- **Local-SGD / s-step outer optimizer.** Sweep `local_steps` against `α`
  and `β` to find the optimal s under each modeled regime.
- **Compressed all-reduce.** Sweep `compression_ratio` and `gamma`; pick
  Pareto front in (`val_bpb`, `comm_fraction`) space.
- **Overlap strategy ablations.** For any base strategy, sweep
  `OverlapModel.factor ∈ {0, 0.5, 0.85, 1.0}`. The finding "method X wins
  only at high overlap" is itself a research result.
- **(α, β) sensitivity sweeps.** Same algorithm, scan a grid of comm
  regimes to produce an operating envelope per method. This is uniquely
  achievable via the modeled harness — a real fabric pins you to one point.

## Limitations to keep in mind

1. **No staleness modeling.** Local-SGD/async methods will be over-credited
   here because their convergence-quality cost from worker divergence is
   not simulated. Cap `local_steps` and treat headline numbers as
   directional.
2. **First-order overlap.** `apply_overlap` linearly hides bandwidth
   behind a compute window; real reduce-scatter / all-gather pipelines are
   richer. Refine if the conclusions hinge on the overlap model.
3. **No congestion / NUMA / message-size knees.** β is constant.
   Acceptable for ranking; not for absolute prediction.
4. **`val_bpb` is still a small-scale proxy.** At ~50M params the optimizer
   inner loop may not be the bottleneck; flat results are a falsifying
   outcome and should be reported as such.

## Useful commands

```bash
uv sync                                              # install deps
uv run prepare.py                                    # one-time data + tokenizer (~2 min)
uv run train.py > run.log 2>&1                       # one experiment
grep "^val_bpb:\|^comm_fraction:\|^sim_" run.log     # extract comm-aware metrics
tail -n 80 run.log                                   # on crash
```
