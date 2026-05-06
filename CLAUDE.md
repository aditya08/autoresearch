# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What autoresearch is

Autoresearch is an autonomous LLM-research harness built around a small,
single-GPU language-model training setup (a simplified single-file fork of
[nanochat](https://github.com/karpathy/nanochat)). The premise: hand an AI
agent a real training pipeline and let it run experiments on its own —
modifying code, training, measuring, keeping or discarding the change, and
moving on — overnight, while the human sleeps.

### How the loop works

1. The agent edits `train.py` (model, optimizer, training loop — everything is
   fair game).
2. It runs `uv run train.py > run.log 2>&1`, which trains for a **fixed
   5-minute wall-clock budget** (excluding startup/compilation).
3. It greps `val_bpb` (validation bits per byte) out of the log. Lower is
   better, and `val_bpb` is vocab-size-independent so architectural changes
   compare fairly.
4. If the metric improved, the commit is kept and the branch advances. If not,
   it's reverted. Crashes are logged and skipped.
5. Each result is appended to `results.tsv` as
   `commit / val_bpb / memory_gb / status / description`.
6. The loop runs forever — the agent does not stop to ask whether to continue.

### Files that matter

- `prepare.py` — **read-only.** Constants, data download, BPE tokenizer,
  dataloader, and the `evaluate_bpb` ground-truth metric. Do not modify.
- `train.py` — **the only file the agent edits.** Full GPT model
  (Muon + AdamW, GQA, RoPE, FlashAttention-3, banded `SSSL` window pattern,
  value-embedding gates), and training loop.
- `program.md` — the instructions the human iterates on; the "skill" handed to
  the agent at the start of a run.
- `pyproject.toml` — dependencies are frozen. The agent cannot add new ones.

### Hard constraints when running experiments here

- Do **not** edit `prepare.py` or the eval harness.
- Do **not** add dependencies.
- Each run targets ~5 min training + a little overhead. If a run exceeds
  10 min, kill it and treat as failure.
- Work on a dedicated `autoresearch/<tag>` branch. Never run the loop on
  `master`.
- VRAM is a soft constraint; meaningful `val_bpb` gains justify some growth,
  but no blow-ups.
- Simpler is better — a 0.001 `val_bpb` gain that adds 20 lines of hacky code
  is not worth it; the same gain via deletion is.

## Use case: step-methods research

The primary research direction this repo supports is **step-methods research**
— studying how the *update step* itself influences training under a fixed
compute budget. Because the wall-clock budget is fixed at 5 minutes, anything
that changes how each step is taken (the optimizer, its preconditioner, the
schedule, gradient accumulation, the curvature approximation, etc.) is
directly comparable across experiments by their final `val_bpb`.

Concretely, a step-methods sweep in this repo looks like:

- **Optimizer variants on the matmul params.** The baseline uses Muon
  (Newton–Schulz orthogonalization step) on 2D weights and AdamW on
  embeddings/heads. Swap in alternatives — Shampoo, SOAP, Lion, PSGD, Adafactor,
  signSGD, plain SGD+momentum — and measure under the same budget.
- **Inner-step count.** Vary the number of Newton–Schulz iterations in Muon
  (or analogous inner solves in other preconditioned methods) to trade cost
  per step against step quality.
- **Multi-step / lookahead schemes.** Lookahead, EMA of weights, slow-fast
  outer/inner loops, multi-step extragradient — does spending compute on
  outer-loop bookkeeping beat just doing more raw steps?
- **Step-size schedules.** Warmup length, decay shape (cosine vs. linear vs.
  WSD), peak LR, group-wise LR ratios between Muon and AdamW.
- **Effective step size knobs.** Gradient accumulation vs. larger
  `device_batch_size`, weight decay coupling, gradient clipping, update
  normalization (per-tensor RMS, spectral norms).
- **Precision of the step.** bf16 vs. fp32 master weights, stochastic
  rounding, momentum buffer dtype.

Each of these maps cleanly onto `train.py` edits and produces a single number
(`val_bpb`) for ranking. Because the time budget — not the step count — is
fixed, the harness naturally rewards step methods that achieve more *progress
per second of wall clock*, which is the quantity that actually matters in
practice. The `results.tsv` log accumulates a directly comparable ladder of
step-method variants over a single overnight run.

When starting such a run, follow the setup in `program.md`: agree on a tag,
cut `autoresearch/<tag>` from `master`, verify `~/.cache/autoresearch/` has
data + tokenizer (else the human runs `uv run prepare.py`), initialize
`results.tsv` with the header, baseline once unmodified, then enter the loop.

## Useful commands

```bash
uv sync                               # install deps
uv run prepare.py                     # one-time data + tokenizer (~2 min)
uv run train.py > run.log 2>&1        # one experiment (~5 min)
grep "^val_bpb:\|^peak_vram_mb:" run.log
tail -n 50 run.log                    # on crash
```
