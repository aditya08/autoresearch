# commresearch

Autonomous research loop for s-step / communication-avoiding methods,
under a *modeled* α–β–γ communication regime with explicit overlap.

## Setup

To set up a new run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date and topic
   (e.g. `mar5-localsgd`, `apr2-cheb-ns`). The branch
   `commresearch/<tag>` must not already exist — fresh runs only.
2. **Create the branch**: `git checkout -b commresearch/<tag>` from
   current `main`.
3. **Read the in-scope files**:
   - `README.md` — repository context and the pivot from autoresearch.
   - `CLAUDE.md` — repository guidance.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader,
     evaluation. **Do not modify.**
   - `train.py` — the file you modify. Model architecture, optimizer,
     training loop, comm config block.
   - `comm_model.py` — α–β–γ model + overlap. You may extend
     (new collectives, refined overlap), but preserve the existing API.
   - `strategies.py` — parallel strategy registry. Add new strategies
     here as new branches in `step_cost`.
4. **Verify data exists**: Check `~/.cache/autoresearch/` for shards and a
   tokenizer. If missing, ask the human to run `uv run prepare.py`.
5. **Initialize results.tsv** with the header below.
6. **Confirm and go.**

Once confirmed, kick off the experimentation.

## Experimentation

The training script runs for a **fixed simulated-time budget of 5 minutes**
(`TIME_BUDGET` in `prepare.py`, reinterpreted as compute + modeled comm).
Real wall-clock will typically be ≤ 5 minutes; methods with high modeled
comm finish *faster* in real time because fewer steps fit in the simulated
budget. A 10-min real-wall-clock kill switch is enforced.

Launch: `uv run train.py > run.log 2>&1`

**What you CAN do:**
- Modify `train.py`: model, optimizer, training loop, the `COMM` /
  `OVERLAP` / `STRATEGY` config block at the top.
- Extend `comm_model.py` and `strategies.py` to add new collectives or
  parallel strategies. Preserve existing public functions
  (`allreduce_seconds`, `apply_overlap`, `step_cost`).

**What you CANNOT do:**
- Modify `prepare.py` or `evaluate_bpb`. Ground-truth metric is fixed.
- Add or remove dependencies.
- Change `TIME_BUDGET`. Hold the budget constant so `val_bpb` is comparable.

**The goal: lowest `val_bpb` under a *declared* comm regime.**
You declare `(COMM, OVERLAP, STRATEGY)` and report `val_bpb` for that
regime. Comparisons are valid within the same declared regime. Cross-regime
comparisons are a separate study (see "Sensitivity sweep" below).

**Simplicity criterion**: All else equal, simpler is better. A 0.001
`val_bpb` improvement that adds 20 lines of hacky code is not worth it. A
0.001 improvement from deletion is. An ~0 change with simpler code is
also a keep.

**The first run**: baseline. Run `train.py` unmodified. Defaults are
`COMM=PRESETS["ndr_ib_8"]`, `OVERLAP=OVERLAP_PRESETS["ddp_typical"]`,
`STRATEGY=dp_allreduce`. Record this as the regime baseline.

## Output format

End-of-run summary now includes:

```
---
val_bpb:                  0.997900
training_seconds:         210.4
total_seconds:            232.0
peak_vram_mb:             45060.2
mfu_percent:              39.80
total_tokens_M:           430.1
num_steps:                820
num_params_M:             50.3
depth:                    8
strategy:                 dp_allreduce
topology:                 ring_ar
sim_world_size:           8
alpha_us:                 5.000
beta_GBs:                 50.000
gamma_per_byte_ns:        0.000
overlap_factor:           0.50
overlap_window_fraction:  0.60
overlap_hide_latency:     False
sim_comm_seconds_raw:     180.5
sim_comm_seconds_eff:     89.6
overlap_hidden_frac:      50.4
comm_fraction:            29.85
sim_total_seconds:        300.0
```

Extract:

```bash
grep "^val_bpb:\|^comm_fraction:\|^sim_\|^strategy:\|^overlap_" run.log
```

## Logging results

Append every run to `results.tsv` (tab-separated). Schema:

```
commit  val_bpb  memory_gb  strategy  alpha_us  beta_GBs  P  overlap_factor  comm_frac  status  description
```

1. git commit hash (short, 7 chars)
2. `val_bpb` (e.g. 0.997900) — `0.000000` for crashes
3. peak memory in GB, `.1f` (peak_vram_mb / 1024) — `0.0` for crashes
4. `strategy` (e.g. `dp_allreduce`)
5. `alpha_us`
6. `beta_GBs`
7. world size `P`
8. `overlap_factor`
9. `comm_fraction` percentage as `.2f`
10. status: `keep` / `discard` / `crash`
11. short text description of what this experiment tried

Example:

```
commit   val_bpb   memory_gb  strategy          alpha_us  beta_GBs  P  overlap_factor  comm_frac  status   description
a1b2c3d  0.997900  44.0       dp_allreduce      5.000     50.000    8  0.50            29.85      keep     baseline ndr_ib_8 ddp_typical
b2c3d4e  0.985300  44.2       dp_local_sgd      5.000     50.000    8  0.50            8.10       keep     local_steps=4
c3d4e5f  1.001000  44.0       dp_compressed_ar  5.000     50.000    8  0.50            12.40      discard  topk-compress 0.1, val_bpb regress
d4e5f6g  0.000000  0.0        dp_local_sgd      5.000     50.000    8  0.50            0.00       crash    local_steps=64 OOM in eval
```

## The experiment loop

Run on a dedicated `commresearch/<tag>` branch.

LOOP FOREVER:

1. Look at git state.
2. Tune `train.py` (or `strategies.py` / `comm_model.py`) with one
   experimental idea. **One variable at a time.**
3. `git commit`
4. `uv run train.py > run.log 2>&1`
5. `grep "^val_bpb:\|^comm_fraction:\|^sim_\|^strategy:\|^overlap_\|^peak_vram_mb:" run.log`
6. Empty output → crash. `tail -n 80 run.log` and try a small fix; if no
   easy fix, log `crash` and move on.
7. Append to `results.tsv` (do **not** commit `results.tsv`; leave
   untracked).
8. If `val_bpb` improved (lower) under the same declared regime, advance
   the branch. If equal-or-worse, `git reset` to where you started.

**Timeout**: ~5-min target, 10-min hard cap. Real wall-clock may be much
shorter at high `comm_fraction`.

**Crashes**: typo / OOM-in-eval → fix and re-run. Idea fundamentally
broken → log `crash`, move on.

**NEVER STOP**: Do not pause to ask whether to continue once setup is done.
The human may be asleep. Run until manually interrupted.

## Recommended phases

Use the regime baseline (Phase 0) as the reference for every later
keep/discard decision *within the same regime*.

### Phase 0 — Regime baseline (1 run)
Default config. Captures `val_bpb`, `comm_fraction`, `sim_*` for
`ndr_ib_8 + ddp_typical + dp_allreduce`.

### Phase 1 — Strategy sweep at the baseline regime (4–6 runs)
- `dp_local_sgd` with `local_steps ∈ {4, 16, 64}`
- `dp_compressed_ar` with `compression_ratio ∈ {0.5, 0.1, 0.01}`
- Each gets one row in `results.tsv` per setting.

### Phase 2 — Overlap sweep (3 runs)
Pin the Phase-1 winner. Sweep
`OverlapModel.factor ∈ {0.0, 0.5, 0.85, 1.0}` (factor=0.5 is the Phase 1
baseline overlap). Records the method's overlap-sensitivity.

### Phase 3 — α–β regime sweep (4–6 runs)
Pin Phase-2 winner. Vary `COMM` across `PRESETS`:
`spark_200gbe_2`, `gbe_8`, `nvlink_8`. Plus a "double β" point
(simulate 2× worse bandwidth). Goal: an operating envelope per method.

### Phase 4 — s-step / NS variants (open-ended)
Polynomial-basis Muon NS, randomized sketched orthogonalization, etc.
Held under the Phase-2 winning regime; declare the regime in the
description. See `CLAUDE.md` "Use case: s-step methods research" for
candidates.

### Phase 5 — Reinvestment (1–2 runs)
Take the cheapest winner that matches Phase-0 `val_bpb` and feed the
freed simulated-time budget back into more steps. This is the
communication-avoidance payoff in a single number.

## Stop conditions

- Per-experiment: 10-min real-wall-clock hard cap (already in `train.py`).
- Per-phase: 3 consecutive variants regress `val_bpb` by ≥ 0.005 → abort
  phase, move on.
- Whole study: declared done when Phase 5 yields a meaningful
  reinvestment win, OR ~30 runs in with no winner (a real negative
  result — record it).
