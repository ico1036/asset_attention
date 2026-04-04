# Program

Read `docs/philosophy.md` first. THE MISSION is non-negotiable.

## THE RULE

**Every experiment MUST use an attention-based model that preserves the time dimension.**

- No mean/sum/last pooling over time BEFORE the attention layer.
- Attention must see sequential time steps (patches, raw days, etc.).
- MLP, Linear, MinVar are BENCHMARKS computed in prepare.py. You don't build them. You compare against them.
- If your attention model scores lower than a benchmark: that's information. Fix the model. Do NOT switch to the benchmark.

## Lock Protocol
```bash
if [ -f LOCK ]; then echo "LOCKED"; exit 1; fi
echo "$(date)" > LOCK
# ... run experiments ...
rm LOCK
```

## Setup (once per session)
```bash
uv sync
uv run prepare.py   # builds tensors from parquet, ~10s
```

## Experiment Loop

1. Read `docs/philosophy.md`, `docs/insights.md`, `docs/experiments.md`.
2. Form a hypothesis about how attention can learn regimes. Write it at the top of `train.py`:
   ```python
   # Hypothesis: PatchTST with 5-day patches lets attention see weekly patterns
   # Expected: val_sharpe [0.3, 1.0], train_time [60, 300]
   # Regime check: attention weights should differ between 2008 crisis and 2017 calm
   ```
3. Modify `train.py`. The model MUST have attention over a time axis.
4. Run: `uv run guard.py` (never train.py directly).
5. Record in `docs/experiments.md`:
   ```
   ## Exp N: [description]
   - Hypothesis: ...
   - Architecture: [attention type, d_model, n_heads, patch_size]
   - val_sharpe: X.XX | params: XXX
   - Regime signal: [did attention weights change across periods? Y/N + detail]
   - Verdict: KEEP / DISCARD
   ```
6. KEEP: `git add -A && git commit && git push`. DISCARD: `git checkout HEAD -- train.py`.
7. Repeat.

## What train.py Must Do

```python
from prepare import (
    load_data, make_samples, split_data,
    evaluate_and_print, write_card,
    N_ASSETS, REBAL_FREQ, MAX_PARAMS,
)
```

1. Load data via `load_data()`, create samples via `make_samples()`.
2. Build features from raw price data (NO collapsing time axis).
3. Define an attention-based model.
4. Train with Sharpe loss (or CVaR, Sortino — your choice).
5. Evaluate with `evaluate_and_print()`.
6. Write card with `write_card()`. Include `regime_signal` in config.
7. NEVER redefine sharpe/sortino/MDD. Use prepare.py's versions.

## Session Limits
- 20 experiments or 3 hours, then stop.
- Always remove LOCK on exit.

## Self-Respawn (mandatory on session end)
1. Remove LOCK.
2. Update `docs/insights.md`.
3. `git add -A && git commit -m "round N complete" && git push`
4. `sessions_send(sessionKey="main", message="Round complete. Spawn next.")`

## Self-Diagnosis (before every experiment)
- Read `docs/insights.md` first.
- If val-test gap > 1.5: overfitting → reduce model or add regularization.
- If training finishes in < 30s with >0 params: data pipeline or model too small.
- If attention weights are static across all periods: the model isn't learning regimes. Change architecture.
- NEVER give up on attention. If stuck, GET CREATIVE:
  - Invent new attention variants (sparse, linear, cross-attention, prototype queries)
  - Hybrid architectures (state-space + attention gate, conv + attention)
  - Novel input representations (signatures, wavelet patches, learned tokenization)
  - Unconventional losses (contrastive regime loss, attention entropy regularization)
  - The constraint is "attention over time for regime learning." HOW is wide open.

## Rules
- Only modify `train.py`. Never touch `prepare.py` or `guard.py`.
- Max 25K parameters.
- Device: MPS (Apple Silicon).
- Walk-forward: train on past, test on future. No shuffling.
- Reproducible (set seed).
- New libraries: `uv add <package>` first.

## Socratic Self-Check (before committing to any new model direction)

Before implementing a new architecture or major design change, interrogate yourself:

1. **인풋 데이터의 여정을 말해봐.** Raw data → model → output까지 한 단계씩 설명. 설명 못 하면 이해 못 한 거다.
2. **피처는 뭐야?** 각 차원이 뭘 뜻하는지 명확히. "17차원"이면 왜 17인지.
3. **데이터포인트는 몇 개야?** 자산을 독립 취급했는가? 안 했으면 왜?
4. **시계열 학습이 되는 구조야?** 시간축이 mean/sum으로 사라지지 않았는가?
5. **이 설계의 전제는 뭐야?** 그 전제가 philosophy.md의 미션과 일치하는가?

이 질문에 명쾌하게 답할 수 없으면 코딩하지 마라. 먼저 생각하라.

## Dream Phase (every 10 experiments)

1. Read all cards and `docs/experiments.md`.
2. **Mission check**: How many experiments actually had attention over time? If < 100%, explain why and fix.
3. **Regime check**: In the best model so far, do attention weights change meaningfully across market periods? If not, what's missing?
4. Extract patterns. Generate 3 new hypotheses for how attention can capture regimes.
5. Update `docs/insights.md` with:
   - Mission progress (not just Sharpe numbers)
   - Best regime signal observed so far
   - Next hypotheses

## What to Explore
- Patch sizes (3, 5, 10, 20 days)
- Attention order: temporal→spatial, spatial→temporal, interleaved
- d_model (8, 16, 32), heads (1, 2, 4)
- Loss: -Sharpe, CVaR, Sortino, -Sharpe + turnover penalty
- Input: raw returns, normalized returns, return + vol features
- Position encoding: RoPE, learned, sinusoidal, none
- Data augmentation if samples insufficient
- Rebalancing frequency changes (daily for more samples)
